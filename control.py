"""
动作控制模块
负责将Qmix的动作ID转换为SUMO中的实际车辆控制
"""

import traci
import numpy as np
from .action_space import ActionSpace
from .planner import TrajectoryPlanner


class VehicleController:
    def __init__(self, grid_generator, action_space, verbose=False):#true变成False
        """
        初始化车辆控制器

        Args:
            grid_generator: 栅格生成器实例
            action_space: 动作空间实例
            verbose: 是否打印详细日志（False 可提速）
        """
        self.grid_generator = grid_generator
        self.action_space = action_space
        self.planner = TrajectoryPlanner()  # 初始化轨迹规划器
        self.verbose = verbose  # 控制打印输出

        # 动作执行历史记录
        self.action_history = {}

        # 记录车辆首次出现的步数，用于warmup期
        self.vehicle_first_seen = {}
        self.warmup_steps = 3  # 前3步不使用moveToXY

        # 动作统计（用于诊断决策vs控制问题）
        self.action_statistics = {
            'forward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stay_count': 0,
            'lane_change_attempts': 0,
            'lane_change_success': 0
        }
        
        # 换道冷却期物理约束（2025-12-03 新增）
        # 模拟真实车辆换道后需要稳定时间，防止连续换道和反复横跳
        # 
        # 参数计算依据（基于用户场景）：
        #   - 栅格区域：500米，87列，每列≈5.75米
        #   - 车速：~12m/s，每步前进~3.6米
        #   - 真实换道时间：车道宽3.5米 / 横向速度1.5m/s ≈ 2.3秒 ≈ 7步
        #   - 若需两次换道（如2→0），冷却期太长会导致来不及
        #   - 5步冷却期：两次换道需10步≈36米≈6个栅格，可在81列前开始换道
        #
        self.vehicle_last_lc_step = {}    # {vehicle_id: step_when_lane_change_started}
        self.lane_change_cooldown = 5     # 换道冷却期（步数），约 1.5 秒（学术优化）
        self.current_step = 0             # 当前仿真步数

    def convert_qmix_action_to_sumo(self, qmix_action_id):
        """
        将Qmix动作ID(0-6)转换为ActionSpace动作ID(1-7)

        Args:
            qmix_action_id: Qmix动作ID (0-6)

        Returns:
            int: ActionSpace动作ID (1-7)
        """
        return qmix_action_id + 1

    def get_vehicle_current_position(self, vehicle_id):
        """
        获取车辆当前位置和车道信息

        Args:
            vehicle_id: 车辆ID

        Returns:
            dict: 车辆位置信息
        """
        try:
            if not self.grid_generator.traci_connected:
                return None

            position = traci.vehicle.getPosition(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)

            return {
                'position': position,
                'lane_id': lane_id,
                'speed': speed,
                'x': position[0],
                'y': position[1]
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting position for vehicle {vehicle_id}: {e}")
            return None

    def execute_action(self, vehicle_id, qmix_action_id):
        """
        执行单个车辆的动作

        Args:
            vehicle_id: 车辆ID
            qmix_action_id: Qmix动作ID (0-6)

        Returns:
            bool: 执行是否成功
        """
        try:
            if not self.grid_generator.traci_connected:
                return False

            # 转换动作ID
            action_id = self.convert_qmix_action_to_sumo(qmix_action_id)

            # 获取动作信息
            action_info = self.action_space.get_action_info(action_id)
            if not action_info:
                return False

            # 获取车辆当前状态
            vehicle_pos = self.get_vehicle_current_position(vehicle_id)
            if not vehicle_pos:
                return False

            # 🔍 添加详细诊断日志：区分决策问题vs控制问题
            current_lane = vehicle_pos['lane_id']
            if self.verbose:
                print(f"🎮 [CONTROL] 车辆{vehicle_id}: 执行动作{action_info['name']}(ID:{qmix_action_id})")
                print(f"    当前车道: {current_lane}, 当前速度: {vehicle_pos['speed']:.2f}m/s")
                print(f"    网格变化: Δi={action_info['delta_i']}, Δj={action_info['delta_j']}")

            # 显示目标车道信息（帮助判断AI决策是否合理）
            try:
                route_edges = traci.vehicle.getRoute(vehicle_id)
                next_edge = traci.vehicle.getRouteIndex(vehicle_id)
                if next_edge < len(route_edges) - 1:
                    target_edge = route_edges[-1]  # 目标路段
                    if self.verbose:
                        print(f"    🎯 路线信息: 当前路段={current_lane.split('_')[0]}, 目标路段={target_edge}")
            except:
                pass

            # 更新动作统计
            action_name = action_info['name']
            if 'Forward' in action_name:
                self.action_statistics['forward_count'] += 1
            elif 'Turn_Left' in action_name:
                self.action_statistics['turn_left_count'] += 1
                self.action_statistics['lane_change_attempts'] += 1
            elif 'Turn_Right' in action_name:
                self.action_statistics['turn_right_count'] += 1
                self.action_statistics['lane_change_attempts'] += 1
            elif 'Stay' in action_name:
                self.action_statistics['stay_count'] += 1

            # 执行具体的控制动作
            success = self._execute_sumo_control(vehicle_id, action_info, vehicle_pos)

            # 记录换道成功情况
            if success and ('Turn_Left' in action_name or 'Turn_Right' in action_name):
                self.action_statistics['lane_change_success'] += 1

            # 记录执行后的实际速度
            if success:
                try:
                    actual_speed = traci.vehicle.getSpeed(vehicle_id)
                    if self.verbose:
                        print(f"    ✅ 执行成功，新速度: {actual_speed:.2f}m/s")
                except:
                    if self.verbose:
                        print(f"    ⚠️ 无法获取执行后速度")
            else:
                if self.verbose:
                    print(f"    ❌ 执行失败")

            # 记录动作历史
            if vehicle_id not in self.action_history:
                self.action_history[vehicle_id] = []
            self.action_history[vehicle_id].append({
                'qmix_action': qmix_action_id,
                'sumo_action': action_id,
                'action_name': action_info['name'],
                'success': success
            })

            return success

        except Exception as e:
            if self.verbose:
                print(f"Error executing action for vehicle {vehicle_id}: {e}")
            return False

    def _execute_sumo_control(self, vehicle_id, action_info, vehicle_pos):
        """
        在SUMO中执行具体的控制动作
        使用五次多项式轨迹规划实现平滑运动

        Args:
            vehicle_id: 车辆ID
            action_info: 动作信息字典
            vehicle_pos: 车辆位置信息

        Returns:
            bool: 执行是否成功
        """
        try:
            # 记录车辆首次出现
            if vehicle_id not in self.vehicle_first_seen:
                self.vehicle_first_seen[vehicle_id] = traci.simulation.getTime()

            # 检查是否在warmup期内
            current_time = traci.simulation.getTime()
            time_since_spawn = current_time - self.vehicle_first_seen[vehicle_id]
            in_warmup = time_since_spawn < (self.warmup_steps * 0.3)  # 0.3s per step

            step_length = 0.3  # 仿真步长(秒) - 已更新为0.3s
            current_pos = vehicle_pos['position']  # (x, y)
            current_speed = vehicle_pos['speed']

            # 1. 计算目标状态
            target_pos = self._calculate_target_position(
                vehicle_id, action_info, current_pos
            )

            # 根据动作确定目标速度
            # 如果是Stay，目标速度为0；如果是Forward/Turn，目标速度根据距离计算
            target_speed = self._calculate_target_speed(vehicle_id, current_pos, target_pos, step_length, action_info)

            # 🚫 轨迹规划(moveToXY)在此场景中不稳定，完全禁用
            # 只使用传统的 setSpeed + changeLane 控制，稳定可靠
            return self._fallback_control(vehicle_id, action_info, target_speed)

            # ===== 以下轨迹规划代码已禁用 =====
            # 2. 构建起始和目标状态向量 [x, y, vx, vy, ax, ay]
            # 简化假设：当前航向角与车道平行，vx=speed, vy=0 (相对车道坐标系，但这里用的是绝对坐标)
            # 为了准确，我们需要获取车辆的航向角
            angle = traci.vehicle.getAngle(vehicle_id)  # 角度: 0是北, 90是东
            angle_rad = np.radians((90 - angle) % 360)  # 转换为标准数学坐标系 (0是东, 逆时针)

            vx = current_speed * np.cos(angle_rad)
            vy = current_speed * np.sin(angle_rad)

            start_state = [current_pos[0], current_pos[1], vx, vy, 0, 0]  # 假设起始加速度为0

            # 计算目标状态的vx, vy
            # 目标航向角：如果是转弯，需要估算；直行则保持
            # 简单起见，我们假设到达目标点时速度方向指向目标点（如果距离远），或者保持车道方向
            # 这里使用简化的终点状态：位置确定，速度大小确定，加速度为0
            # 终点速度方向：近似为起点到终点的向量方向
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist > 0.1:
                target_vx = target_speed * (dx / dist)
                target_vy = target_speed * (dy / dist)
            else:
                target_vx = 0
                target_vy = 0

            end_state = [target_pos[0], target_pos[1], target_vx, target_vy, 0, 0]

            # 3. 生成轨迹
            # 规划时长等于仿真步长，因为每个step我们都要重新规划
            # 但为了平滑，我们可以规划未来一段时间的轨迹，然后取第一个点
            planning_horizon = step_length
            trajectory = self.planner.generate_trajectory(start_state, end_state, planning_horizon, dt=step_length)

            if not trajectory or len(trajectory) < 2:
                if self.verbose:
                    print(f"    ⚠️ 轨迹生成失败或太短")
                return False

            # 4. 执行控制：取轨迹的终点（因为dt=step_length）
            # trajectory[0]是起点，trajectory[-1]是step_length时刻的点
            next_pt = trajectory[-1]
            next_x, next_y, next_v, next_yaw = next_pt

            # 安全检查1：确保移动距离不会太大
            move_dist = np.sqrt((next_x - current_pos[0]) ** 2 + (next_y - current_pos[1]) ** 2)
            if move_dist > 5.0:  # 降低到5米，更保守
                if self.verbose:
                    print(f"    ⚠️ 单步移动距离过大({move_dist:.1f}m)，使用传统控制")
                return self._fallback_control(vehicle_id, action_info, target_speed)

            # 安全检查2：限制速度
            next_v = min(next_v, 12.0)  # 最大速度 12 m/s (43 km/h)

            # 使用moveToXY强制移动车辆到下一个轨迹点
            # keepRoute=2 表示更严格地遵循路网
            # 这里的angle需要转换回SUMO的角度标准 (0是北, 顺时针增加)
            sumo_angle = (90 - np.degrees(next_yaw)) % 360

            try:
                traci.vehicle.moveToXY(vehicle_id, "", 0, next_x, next_y, angle=sumo_angle, keepRoute=2)
                traci.vehicle.setSpeed(vehicle_id, next_v)  # 同时设定速度以匹配内部状态

                if self.verbose:
                    print(
                        f"      📍 轨迹规划执行: ({current_pos[0]:.1f}, {current_pos[1]:.1f}) -> ({next_x:.1f}, {next_y:.1f}), v={next_v:.1f}")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️ moveToXY失败: {e}，使用保守控制")
                return self._fallback_control(vehicle_id, action_info, target_speed)

        except Exception as e:
            if self.verbose:
                print(f"Error in SUMO control for vehicle {vehicle_id}: {e}")
            return False

    def _calculate_target_speed(self, vehicle_id, current_pos, target_pos, step_length, action_info):
        """
        计算目标速度
        
        【学术设计】：
        - 纵向控制（加减速）完全交给SUMO的跟驰模型（IDM/Krauss）
        - RL只负责横向控制（换道决策）
        - 通过设置较高的期望速度，让SUMO的跟驰模型自行决定实际速度
        """
        import traci
        
        # 设置较高的期望速度，让SUMO跟驰模型自行决定
        # SUMO会根据前车距离、安全约束等自动调整实际速度
        desired_speed = 13.89  # m/s = 50 km/h（城市道路常见限速）
        
        try:
            # 提高车辆的最大速度限制
            traci.vehicle.setMaxSpeed(vehicle_id, desired_speed)
        except:
            pass
        
        # 返回 -1.0 表示让SUMO的跟驰模型控制速度
        # SUMO会使用IDM/Krauss等成熟模型自动处理：
        # - 前方无车 → 加速到期望速度
        # - 前方有车 → 根据跟驰模型减速保持安全距离
        return -1.0

    def _fallback_control(self, vehicle_id, action_info, target_speed):
        """
        保守的fallback控制方法（当moveToXY失败时使用）
        使用传统的setSpeed和changeLane

        Args:
            vehicle_id: 车辆ID
            action_info: 动作信息
            target_speed: 目标速度

        Returns:
            bool: 是否成功
        """
        try:
            # 启用手动速度控制模式
            # speedMode: 31 = 允许更多控制权限，但保留基本安全机制
            traci.vehicle.setSpeedMode(vehicle_id, 31)

            # 【学术设计】换道模式：允许 Agent 自由尝试，通过奖励学习
            # laneChangeMode: 512 = 只保留最基本的碰撞避免
            # 
            # 学术原理：
            # - 让 Agent 可以尝试危险换道
            # - 碰撞/紧急刹车会触发负奖励（在 reward.py 中实现）
            # - Agent 通过"尝试→失败→惩罚→学习"的循环掌握安全换道
            # 
            # 这符合强化学习的核心范式：从经验中学习，而非被硬性约束
            traci.vehicle.setLaneChangeMode(vehicle_id, 512)

            # 让SUMO的跟驰模型控制速度
            # target_speed = -1.0 表示交给SUMO控制
            if target_speed < 0:
                traci.vehicle.setSpeed(vehicle_id, -1)  # SUMO自动控制
            else:
                traci.vehicle.setSpeed(vehicle_id, target_speed)

            # 根据动作执行换道
            action_name = action_info['name']
            step_length = 0.3

            if 'Turn_Left' in action_name:
                self._change_lane_left_with_timing(vehicle_id)
            elif 'Turn_Right' in action_name:
                self._change_lane_right_with_timing(vehicle_id)

            if self.verbose:
                print(f"      🔧 传统控制: 动作={action_name}, 速度={target_speed:.1f}m/s")
            return True

        except Exception as e:
            if self.verbose:
                print(f"    ❌ Fallback控制也失败: {e}")
            return False

    def _calculate_target_position(self, vehicle_id, action_info, current_pos):
        """
        根据动作计算目标位置

        Args:
            vehicle_id: 车辆ID
            action_info: 动作信息字典
            current_pos: 当前位置 (x, y)

        Returns:
            tuple: 目标位置 (x, y)
        """
        # 获取当前网格坐标
        current_grid = None
        for grid in self.grid_generator.grids:
            if vehicle_id in grid['vehicle_ids']:
                current_grid = grid['coordinate']
                break
        if not current_grid:
            return current_pos

        i, j = current_grid
        delta_i = action_info['delta_i']  # 横向变化
        delta_j = action_info['delta_j']  # 纵向变化

        # 计算目标网格坐标
        target_grid = (i + delta_i, j + delta_j)

        # 转换为物理坐标
        target_physical = self.grid_generator.get_physical_position(*target_grid)

        return target_physical if target_physical else current_pos

    def _calculate_required_speed(self, current_pos, target_pos, step_length, action_info):
        """
        计算达到目标位置所需的速度

        Args:
            current_pos: 当前位置 (x, y)
            target_pos: 目标位置 (x, y)
            step_length: 仿真步长(秒)
            action_info: 动作信息字典

        Returns:
            float: 所需速度 (m/s)
        """
        if not target_pos:
            return 8.0  # 默认速度

        # 计算欧氏距离
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # 计算所需速度（包括Stay动作距离为0的情况）
        if distance == 0:
            required_speed = 0.0  # Stay动作或其他原地动作
        else:
            required_speed = distance / step_length

        return required_speed

    def _apply_speed_constraints(self, required_speed):
        """
        应用速度约束，选择最合适的速度

        Args:
            required_speed: 计算得出的所需速度

        Returns:
            float: 约束后的最终速度
        """
        # 速度限制配置
        min_speed = 2.0  # 最小速度 2m/s
        max_speed = 30.0  # 最大速度 30m/s

        # 应用约束
        if required_speed < min_speed:
            # 如果计算速度太小，使用最小速度
            final_speed = min_speed
        elif required_speed > max_speed:
            # 如果计算速度太大，使用最大速度
            final_speed = max_speed
        else:
            # 计算速度在合理范围内
            final_speed = required_speed

        return final_speed

    def _execute_control_commands(self, vehicle_id, action_info, final_speed):
        """
        执行具体的控制命令

        Args:
            vehicle_id: 车辆ID
            action_info: 动作信息字典
            final_speed: 最终速度

        Returns:
            bool: 执行是否成功
        """
        try:
            # 🔍 详细控制日志
            if self.verbose:
                print(f"      � 计算得出目标速度: {final_speed:.2f}m/s")

            # 1. 设置速度（速度约束已在上层处理）
            traci.vehicle.setSpeed(vehicle_id, final_speed)
            if self.verbose:
                print(f"      🚗 已设置SUMO车辆速度: {final_speed:.2f}m/s")

            # 2. 根据动作类型执行相应的换道命令
            action_name = action_info['name']
            if 'Turn_Left' in action_name:
                if self.verbose:
                    print(f"      ⬅️ 执行左换道命令")
                self._change_lane_left_with_timing(vehicle_id)
            elif 'Turn_Right' in action_name:
                if self.verbose:
                    print(f"      ➡️ 执行右换道命令")
                self._change_lane_right_with_timing(vehicle_id)
            elif action_name == 'Stay':
                if self.verbose:
                    print(f"      ⏸️ 执行原地等待（最小速度）")
            else:
                if self.verbose:
                    print(f"      ⬆️ 执行直行命令（无换道）")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Error executing control commands: {e}")
            return False

    def get_action_history(self, vehicle_id=None):
        """
        获取动作执行历史

        Args:
            vehicle_id: 车辆ID，None表示获取所有车辆的历史

        Returns:
            dict or list: 动作历史记录
        """
        if vehicle_id:
            return self.action_history.get(vehicle_id, [])
        else:
            return self.action_history

    def _is_lane_change_allowed(self, vehicle_id):
        """
        检查车辆是否可以换道（冷却期物理约束）
        
        模拟真实车辆换道后需要稳定时间，防止连续换道（2→1→0）和反复横跳（2→3→2）
        
        学术特性：
            1. 物理可解释：真实车辆换道需要2-3秒稳定
            2. 非奖励设计：是物理约束，不是人为打分
            3. 自然防止不合理行为：连续换道、反复横跳
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            bool: True 表示可以换道，False 表示在冷却期内
        """
        if vehicle_id not in self.vehicle_last_lc_step:
            return True  # 首次换道，允许
        
        steps_since_lc = self.current_step - self.vehicle_last_lc_step[vehicle_id]
        return steps_since_lc >= self.lane_change_cooldown
    
    def update_step(self, step):
        """更新当前仿真步数"""
        self.current_step = step
    
    def _change_lane_left_with_timing(self, vehicle_id):
        """
        考虑时间同步的左换道（带冷却期物理约束）

        Args:
            vehicle_id: 车辆ID
        """
        try:
            # 【物理约束】检查换道冷却期
            if not self._is_lane_change_allowed(vehicle_id):
                if self.verbose:
                    cooldown_remaining = self.lane_change_cooldown - (self.current_step - self.vehicle_last_lc_step.get(vehicle_id, 0))
                    print(f"      ⏳ 车辆{vehicle_id}换道冷却中，剩余{cooldown_remaining}步")
                return  # 在冷却期内，忽略换道请求
            
            current_lane_id = traci.vehicle.getLaneID(vehicle_id)

            # 计算目标车道
            target_lane_index = -1
            if 'E0_0' in current_lane_id:
                target_lane_index = 1
            elif 'E0_1' in current_lane_id:
                target_lane_index = 2
            elif 'E0_2' in current_lane_id:
                target_lane_index = 3
            else:
                if self.verbose:
                    print(f"      ⚠️ 车辆{vehicle_id}已在最左侧或不在E0路段({current_lane_id})，无法左转")
                return  # 已在最左侧

            # 【学术设计】安全检查已移至奖励函数（r_safety 组件）
            # 让 Agent 通过学习决定何时换道是安全的
            # SUMO 内置碰撞避免机制作为物理层保护

            if self.verbose:
                print(f"      🔄 车辆{vehicle_id}尝试左转: {current_lane_id} -> Lane {target_lane_index}")

            # 记录换道时间（用于冷却期计算）
            self.vehicle_last_lc_step[vehicle_id] = self.current_step
            
            # 换道持续时间（给车辆足够的时间完成换道）
            lane_change_duration = 3.0
            traci.vehicle.changeLane(vehicle_id, target_lane_index, lane_change_duration)

        except Exception as e:
            if self.verbose:
                print(f"Error in timed lane change left: {e}")

    def _change_lane_right_with_timing(self, vehicle_id):
        """
        考虑时间同步的右换道（带冷却期物理约束）

        Args:
            vehicle_id: 车辆ID
        """
        try:
            # 【物理约束】检查换道冷却期
            if not self._is_lane_change_allowed(vehicle_id):
                if self.verbose:
                    cooldown_remaining = self.lane_change_cooldown - (self.current_step - self.vehicle_last_lc_step.get(vehicle_id, 0))
                    print(f"      ⏳ 车辆{vehicle_id}换道冷却中，剩余{cooldown_remaining}步")
                return  # 在冷却期内，忽略换道请求
            
            current_lane_id = traci.vehicle.getLaneID(vehicle_id)

            # 计算目标车道
            target_lane_index = -1
            if 'E0_3' in current_lane_id:
                target_lane_index = 2
            elif 'E0_2' in current_lane_id:
                target_lane_index = 1
            elif 'E0_1' in current_lane_id:
                target_lane_index = 0
            else:
                if self.verbose:
                    print(f"      ⚠️ 车辆{vehicle_id}已在最右侧或不在E0路段({current_lane_id})，无法右转")
                return  # 已在最右侧

            # 【学术设计】安全检查已移至奖励函数（r_safety 组件）
            # 让 Agent 通过学习决定何时换道是安全的
            # SUMO 内置碰撞避免机制作为物理层保护

            if self.verbose:
                print(f"      🔄 车辆{vehicle_id}尝试右转: {current_lane_id} -> Lane {target_lane_index}")
            
            # 记录换道时间（用于冷却期计算）
            self.vehicle_last_lc_step[vehicle_id] = self.current_step

            # 换道持续时间（给车辆足够的时间完成换道）
            lane_change_duration = 3.0
            traci.vehicle.changeLane(vehicle_id, target_lane_index, lane_change_duration)

        except Exception as e:
            if self.verbose:
                print(f"Error in timed lane change right: {e}")

    # 【已移除】_is_lane_change_safe 方法
    # 原因：硬阈值判断（if gap < threshold）不符合学术规范
    # 安全引导已通过奖励函数 r_safety 组件实现（全连续可微）
    # SUMO 内置碰撞避免机制作为物理层保护

    def execute_batch_actions(self, vehicle_ids, qmix_actions):
        """
        批量执行车辆动作
        简化版本：直接执行所有动作，不做冲突检测

        Args:
            vehicle_ids: 车辆ID列表
            qmix_actions: Qmix动作ID列表 (0-3)

        Returns:
            dict: 执行结果 {vehicle_id: success}
        """
        results = {}

        for i, vehicle_id in enumerate(vehicle_ids):
            if i < len(qmix_actions):
                action_id = qmix_actions[i]
                # 直接执行动作
                success = self.execute_action(vehicle_id, action_id)
                results[vehicle_id] = success
            else:
                results[vehicle_id] = False

        return results

    def reset_history(self):
        """
        重置车辆历史记录
        """
        self.action_history = {}
        self.vehicle_first_seen = {}  # 同时清空首次出现记录
        
        # 重置换道冷却期状态（2025-12-03 新增）
        self.vehicle_last_lc_step = {}
        self.current_step = 0

    def get_action_statistics_report(self):
        """
        生成动作统计报告（用于诊断决策vs控制问题）

        Returns:
            str: 格式化的统计报告
        """
        stats = self.action_statistics
        total_actions = sum([stats['forward_count'], stats['turn_left_count'],
                             stats['turn_right_count'], stats['stay_count']])

        if total_actions == 0:
            return "📊 [统计] 尚无动作执行"

        report = "\n" + "=" * 60 + "\n"
        report += "📊 动作执行统计报告（诊断工具）\n"
        report += "=" * 60 + "\n"

        # 动作分布
        report += f"🎮 动作分布:\n"
        report += f"   Forward:    {stats['forward_count']:4d} ({stats['forward_count'] / total_actions * 100:5.1f}%)\n"
        report += f"   Turn_Left:  {stats['turn_left_count']:4d} ({stats['turn_left_count'] / total_actions * 100:5.1f}%)\n"
        report += f"   Turn_Right: {stats['turn_right_count']:4d} ({stats['turn_right_count'] / total_actions * 100:5.1f}%)\n"
        report += f"   Stay:       {stats['stay_count']:4d} ({stats['stay_count'] / total_actions * 100:5.1f}%)\n"
        report += f"   总计:       {total_actions:4d}\n\n"

        # 换道成功率
        if stats['lane_change_attempts'] > 0:
            success_rate = stats['lane_change_success'] / stats['lane_change_attempts'] * 100
            report += f"🔀 换道执行情况:\n"
            report += f"   尝试次数: {stats['lane_change_attempts']}\n"
            report += f"   成功次数: {stats['lane_change_success']}\n"
            report += f"   成功率:   {success_rate:.1f}%\n\n"

        # 诊断建议
        report += "💡 诊断分析:\n"
        lane_change_ratio = (stats['turn_left_count'] + stats['turn_right_count']) / total_actions * 100

        if lane_change_ratio < 20:
            report += "   ⚠️ 换道动作比例过低 (<20%)，可能是**上层决策问题**\n"
            report += "      → AI没有选择足够的换道动作\n"
            report += "      → 建议检查：奖励函数、观测信息、训练状态\n"
        else:
            report += "   ✓ 换道动作比例合理 (>=20%)\n"

        if stats['lane_change_attempts'] > 0:
            success_rate = stats['lane_change_success'] / stats['lane_change_attempts'] * 100
            if success_rate < 80:
                report += "   ⚠️ 换道成功率过低 (<80%)，可能是**下层控制问题**\n"
                report += "      → 换道命令执行失败\n"
                report += "      → 建议检查：laneChangeMode、换道时间、交通密度\n"
            else:
                report += "   ✓ 换道成功率正常 (>=80%)\n"

        report += "=" * 60 + "\n"
        return report

    def reset(self):
        """
        重置环境
        """
        self.action_history = {}
        self.vehicle_first_seen = {}  # 同时清空首次出现记录

        # 重置统计
        self.action_statistics = {
            'forward_count': 0,
            'turn_left_count': 0,
            'turn_right_count': 0,
            'stay_count': 0,
            'lane_change_attempts': 0,
            'lane_change_success': 0
        }