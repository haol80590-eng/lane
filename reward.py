"""
IPPO 奖励计算模块

==============================================================================
学术奖励函数设计 - 基于势能场理论 + TTC安全惩罚
==============================================================================

【总奖励】
    r = r_lane + r_safety

【车道势能 r_lane】紧迫度调制设计

    r_lane = r_target + r_deviation
    
    紧迫度因子 u(p)：
        u(p) = 0                        当 p < 0.2（入口区）
        u(p) = (p-0.2)/0.8 * (1+β·p)    当 p >= 0.2
    
    r_target = R_t · u(p) · 1_{d=0}              # 在目标车道时的正奖励
    r_deviation = -α · d · u(p)                  # 偏离目标车道的惩罚
    
    物理意义：
    - 入口区（p<0.2）：无换道激励，专注加速融入车流
    - 中后段（p>=0.2）：换道激励逐渐增强
    - 避免车辆在入口处急着换道、阻塞车流

【安全惩罚 r_safety】TTC 连续函数（有学术依据）

    r_safety = -k / TTC   when TTC < TTC_th
    r_safety = 0          when TTC >= TTC_th
    
    学术依据：
    - Hayward (1972), "Near-miss determination through use of a scale of danger"
    - arxiv:2405.01440 推荐使用 TTC 作为安全度量
    
    物理意义：
    - TTC（Time To Collision）直接度量碰撞风险
    - 越接近碰撞，惩罚越大（连续递增）
    - 前瞻性信号，优于加速度后果性信号

【参数设置】
    R_t = 0.3       在目标车道时的正奖励
    α = 0.2         基础偏离惩罚系数
    β = 2.0         进度放大系数
    k = 1.0         TTC 惩罚系数
    TTC_th = 3.0    TTC 阈值（秒）

【学术特性】
    1. 车道偏离惩罚：线性距离 × 进度加权（论文常见）
    2. TTC 安全惩罚：有明确学术引用
    3. 正向激励：在目标车道时有正奖励
    4. 物理可解释：基于势能场和碰撞时间
==============================================================================
"""

import numpy as np
import traci


class RewardCalculator:
    """IPPO 个体奖励计算器 - 势能场 + TTC 版本"""

    def __init__(self, qminx_network):
        """初始化奖励计算器"""
        self.qminx = qminx_network

        # ==================================================================
        # 奖励函数参数（学术规范设计）
        # ==================================================================
        #
        # r_lane = R_t · 1_{d=0} - α · d · (1 + β · p)
        # r_safety = -k / TTC  when TTC < TTC_th
        #
        self.reward_config = {
            # --------------------------------------------------------------
            # 车道势能参数（简洁设计）
            # --------------------------------------------------------------
            'R_target': 0.3,    # 在目标车道时的正奖励（解决正向激励缺失）
            'alpha': 0.2,       # 基础偏离惩罚系数
            'beta': 2.0,        # 进度放大系数（后期惩罚更重）

            # --------------------------------------------------------------
            # TTC 安全惩罚参数（学术依据：arxiv:2405.01440）
            # --------------------------------------------------------------
            'k_ttc': 1.0,       # TTC 惩罚系数
            'ttc_threshold': 3.0,   # TTC 阈值（秒），Euro NCAP 标准
            'ttc_min': 0.5,     # 最小 TTC（防止除零，对应最大惩罚 -2.0）
        }

        # 状态跟踪
        self.exited_vehicles = {}
        self.previous_active_vehicles = set()
        self.all_seen_vehicles = set()
        self.vehicle_target_status = {}
        self.newly_exited_this_step = {}
        self.dangerous_approach_count = 0

    def calculate_progress_ratio(self, vehicle_id):
        """
        计算车辆在栅格区域内的进度比例 (0.0 - 1.0)

        用于紧迫度因子计算
        """
        try:
            grid_position = None
            for grid in self.qminx.grid_generator.grids:
                if vehicle_id in grid['vehicle_ids']:
                    grid_position = grid['coordinate']
                    break

            if grid_position is None:
                if vehicle_id in self.exited_vehicles:
                    return 1.0
                return 0.0

            current_j = grid_position[1]
            total_grid_length = self.qminx.grid_generator.grid_length_count

            if total_grid_length > 0:
                progress = current_j / total_grid_length
                return min(progress, 1.0)

            return 0.0

        except Exception as e:
            return 0.0

    def reset_mission_tracking(self):
        """重置所有状态"""
        self.exited_vehicles = {}
        self.previous_active_vehicles = set()
        self.all_seen_vehicles = set()
        self.vehicle_target_status = {}
        self.newly_exited_this_step = {}
        self.dangerous_approach_count = 0

        # 清除 QminxNetwork 的横向位置历史
        if hasattr(self.qminx, 'vehicle_lateral_positions'):
            self.qminx.vehicle_lateral_positions = {}

    def update_exited_vehicles(self, current_step):
        """更新已驶离栅格区域的车辆记录"""
        current_active = set(self.qminx.get_active_vehicle_ids())

        # 记录当前活跃车辆的状态
        for vehicle_id in current_active:
            try:
                in_target = self.qminx.is_vehicle_in_target_lane(vehicle_id)
                self.vehicle_target_status[vehicle_id] = in_target
            except:
                self.vehicle_target_status[vehicle_id] = False

        newly_exited = self.previous_active_vehicles - current_active

        self.newly_exited_this_step = {}

        for vehicle_id in newly_exited:
            if vehicle_id not in self.exited_vehicles:
                in_target = self.vehicle_target_status.get(vehicle_id, False)

                exit_info = {
                    'in_target_lane': in_target,
                    'exit_step': current_step,
                }
                self.exited_vehicles[vehicle_id] = exit_info
                self.newly_exited_this_step[vehicle_id] = exit_info

                self.vehicle_target_status.pop(vehicle_id, None)

                # 恢复 SUMO 控制
                try:
                    if vehicle_id in traci.vehicle.getIDList():
                        traci.vehicle.setSpeedMode(vehicle_id, 31)
                        traci.vehicle.setLaneChangeMode(vehicle_id, 256)
                        traci.vehicle.setSpeed(vehicle_id, -1)
                except Exception as e:
                    pass

        self.previous_active_vehicles = current_active.copy()
        self.all_seen_vehicles.update(current_active)

    def calculate_individual_rewards_for_ippo(self, vehicle_ids: list,
                                              include_exited: bool = False,
                                              actions: list = None) -> dict:
        """
        计算 IPPO 个体奖励（简洁势能场 + TTC 安全惩罚）

        数学公式：
            r = r_lane + r_safety

            r_lane = R_t · 1_{d=0} - α · d · (1 + β · p)
            
            物理意义（论文常见形式）：
            - 在目标车道时有正奖励（解决正向激励缺失）
            - 偏离目标车道时有惩罚，且随进度增加

            r_safety = -k / TTC  when TTC < TTC_th
            
            学术依据：
            - Hayward (1972), TTC 作为碰撞风险度量
            - arxiv:2405.01440 推荐 TTC 连续函数

        Args:
            vehicle_ids: 车辆 ID 列表
            include_exited: 是否包含刚离开的车辆
            actions: 动作列表（未使用）

        Returns:
            rewards: {vehicle_id: reward} 字典
        """
        rewards = {}

        # 获取参数
        R_target = self.reward_config.get('R_target', 0.3)  # 在目标车道时的正奖励
        alpha = self.reward_config.get('alpha', 0.2)        # 基础偏离惩罚
        beta = self.reward_config.get('beta', 2.0)          # 进度放大系数

        for vehicle_id in vehicle_ids:
            reward = 0.0

            try:
                if vehicle_id in getattr(self, 'newly_exited_this_step', {}):
                    reward = 0.0
                else:
                    current_lane = self.qminx.get_vehicle_current_lane(vehicle_id)
                    optimal_lane = self.qminx.get_optimal_target_lane(vehicle_id)
                    p = self.calculate_progress_ratio(vehicle_id)

                    if current_lane is None or optimal_lane is None:
                        rewards[vehicle_id] = 0.0
                        continue

                    # 车道偏离距离
                    d = abs(current_lane - optimal_lane)

                    # ==================================================
                    # r_lane = R_t · 1_{d=0} - α · d · u(p)
                    # 
                    # 紧迫度因子 u(p)：
                    # - p < 0.2: u = 0（入口区，专注加速，不急换道）
                    # - p >= 0.2: u = (p - 0.2) / 0.8 * (1 + β*p)
                    #
                    # 物理意义：刚进入道路应先加速融入车流，
                    # 到达中后段再考虑换道
                    # ==================================================
                    
                    # 入口区（前20%路程）：无换道激励，让车辆专注前进
                    if p < 0.2:
                        r_lane = 0.0  # 不惩罚也不奖励，车辆自由前进
                    else:
                        # 紧迫度因子：从0.2开始逐渐增加
                        urgency = (p - 0.2) / 0.8  # 0到1之间
                        
                        if d == 0:
                            # 在目标车道：正奖励
                            r_lane = R_target * (1 + beta * urgency)
                        else:
                            # 偏离目标车道：惩罚（后期更重）
                            r_lane = -alpha * d * (1 + beta * urgency)

                    # ==================================================
                    # r_safety = -k / TTC  (TTC < threshold)
                    # 
                    # 学术依据：arxiv:2405.01440
                    # ==================================================
                    r_safety = self._calculate_safety_penalty_ttc(vehicle_id)

                    reward = r_lane + r_safety

            except Exception as e:
                reward = 0.0

            rewards[vehicle_id] = reward

        return rewards

    def _calculate_safety_penalty_ttc(self, vehicle_id: str) -> float:
        """
        基于 TTC 的安全惩罚（有学术依据）

        数学公式：
            r_safety = -k / TTC   when TTC < TTC_th
            r_safety = 0          when TTC >= TTC_th

        学术依据：
            - Hayward (1972), "Near-miss determination through use of a scale of danger"
            - arxiv:2405.01440: "TTC is considered more suitable than other risk heuristics"
            
        参数设置依据：
            - TTC_th = 3.0s: Euro NCAP 安全标准
            - k = 1.0: 当 TTC=1s 时惩罚 -1.0，TTC=0.5s 时惩罚 -2.0
            - ttc_min = 0.5s: 防止除零，对应最大惩罚 -2.0

        Returns:
            float: 惩罚值 ∈ [-k/ttc_min, 0]
        """
        try:
            # 获取参数
            k_ttc = self.reward_config.get('k_ttc', 1.0)
            ttc_threshold = self.reward_config.get('ttc_threshold', 3.0)
            ttc_min = self.reward_config.get('ttc_min', 0.5)

            # 获取与前车的 TTC
            ttc = self._get_ttc_to_front_vehicle(vehicle_id)

            if ttc is None or ttc >= ttc_threshold:
                return 0.0  # 安全，无惩罚

            # TTC 连续惩罚：r = -k / TTC
            # TTC 越小，惩罚越大
            effective_ttc = max(ttc, ttc_min)  # 防止除零
            r_safety = -k_ttc / effective_ttc

            # 统计危险事件
            if ttc < 2.0:
                self.dangerous_approach_count += 1

            return r_safety

        except Exception:
            return 0.0

    def _get_ttc_to_front_vehicle(self, vehicle_id: str) -> float:
        """
        获取与前车的 TTC（Time To Collision）

        TTC = distance / relative_speed

        Returns:
            float: TTC 值（秒），如果无法计算则返回 None
        """
        try:
            my_speed = traci.vehicle.getSpeed(vehicle_id)

            # 获取前车信息
            leader_info = traci.vehicle.getLeader(vehicle_id, dist=50.0)
            if leader_info is None or leader_info[0] == '':
                return None  # 没有前车

            leader_id, distance = leader_info
            leader_speed = traci.vehicle.getSpeed(leader_id)

            # 计算相对速度
            relative_speed = my_speed - leader_speed

            # 只有当我比前车快时才有碰撞风险
            if relative_speed <= 0.1:
                return None  # 不会追尾

            ttc = distance / relative_speed
            return ttc

        except Exception:
            return None

    def get_individual_reward_for_vehicle(self, vehicle_id: str) -> float:
        """获取单个车辆的个体奖励"""
        return self.calculate_individual_rewards_for_ippo([vehicle_id]).get(vehicle_id, 0.0)

    def update_reward_config(self, new_config):
        """更新奖励配置"""
        self.reward_config.update(new_config)
        print(f"Updated reward config: {self.reward_config}")

    def get_mission_statistics(self):
        """获取任务统计信息"""
        total = len(self.exited_vehicles)
        success = sum(1 for v in self.exited_vehicles.values() if v.get('in_target_lane', False))
        return {
            'total_vehicles': total,
            'success_count': success,
            'success_rate': success / total if total > 0 else 0.0,
            'dangerous_approach_count': self.dangerous_approach_count
        }

    def calculate_qmix_rewards(self, vehicle_ids, action_results=None, agent_mask=None,
                               current_step=None, all_vehicles_exited=False):
        """
        计算 Qmix 格式的奖励（兼容环境接口）

        对于 IPPO 训练，返回全局奖励（个体奖励的平均值）
        实际的个体奖励通过 calculate_individual_rewards_for_ippo 获取

        Args:
            vehicle_ids: 车辆 ID 列表
            action_results: 动作执行结果（未使用）
            agent_mask: 智能体掩码
            current_step: 当前步数
            all_vehicles_exited: 是否所有车辆已离开

        Returns:
            float: 全局奖励值
        """
        # 计算所有活跃车辆的个体奖励
        individual_rewards = self.calculate_individual_rewards_for_ippo(vehicle_ids)

        # 计算平均奖励作为全局奖励
        if individual_rewards:
            valid_rewards = [r for r in individual_rewards.values() if r != 0.0]
            if valid_rewards:
                global_reward = np.mean(valid_rewards)
            else:
                global_reward = 0.0
        else:
            global_reward = 0.0

        return global_reward
