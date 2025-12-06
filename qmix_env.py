"""
Qmix环境接口
整合所有组件，提供标准的多智能体强化学习环境接口
"""

import sys
import numpy as np
import torch
import traci

# 导入依赖模块
from .grid import SUMOGridGenerator
from .state import QminxNetwork
from .action_space import ActionSpace
from .control import VehicleController
from .reward import RewardCalculator


class QmixEnvironment:
    def __init__(self, config=None):
        """
        初始化Qmix环境

        Args:
            config: 环境配置字典
        """
        # 默认配置
        self.config = config or {
            'net_file': 'sumo/net.net.xml',
            'cfg_file': 'sumo/car.sumocfg',
            'max_agents': 10,
            'max_steps': 1000,
            'gui_mode': False
            #'gui_mode': True
        }

        # 初始化组件
        self.grid_generator = None
        self.qminx = None
        self.action_space = None
        self.controller = None
        self.reward_calculator = None

        # 环境状态
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 500)
        self.max_agents = self.config.get('max_agents', 10)
        self.is_initialized = False
        self.verbose = self.config.get('verbose', False)  # 控制打印输出

        # 数据缓存
        self.last_observations = None
        self.last_global_state = None
        self.last_agent_mask = None

        # ====== 学术评价指标跟踪（新增 2025-11-30）======
        self.episode_metrics = {
            'dangerous_approaches': 0,  # 危险接近次数（距离<安全阈值）
            'total_lane_changes': 0,  # 总换道次数
            'speed_samples': [],  # 速度采样（用于计算平均速度）
            'travel_times': {},  # 每辆车的通过时间 {vid: (enter_step, exit_step)}
        }
        self.d_safe_threshold = 8.0  # 安全距离阈值（米）

    def initialize(self):
        """
        初始化所有组件
        """
        if self.is_initialized:
            return True

        try:
            print("Initializing Qmix Environment...")

            # 1. 初始化栅格生成器
            self.grid_generator = SUMOGridGenerator(
                net_file_path=self.config['net_file'],
                sumo_cfg_path=self.config['cfg_file'],
                verbose=self.verbose
            )

            # 2. 初始化动作空间
            self.action_space = ActionSpace(max_grid_sequence=self.grid_generator.grid_length_count)

            # 3. 初始化控制器
            self.controller = VehicleController(self.grid_generator, self.action_space, verbose=self.verbose)

            # 4. 初始化状态网络（传入控制器引用用于冷却期状态）
            self.qminx = QminxNetwork(self.grid_generator, self.controller)

            # 5. 初始化奖励计算器
            self.reward_calculator = RewardCalculator(self.qminx)

            # 7. 生成栅格
            self.grid_generator.generate_grids()

            # 8. 启动SUMO
            gui_mode = self.config.get('gui_mode', False)
            self.grid_generator.start_sumo(gui=gui_mode)

            self.is_initialized = True
            print("Qmix Environment initialized successfully.")
            return True

        except Exception as e:
            print(f"Error initializing Qmix Environment: {e}")
            return False

    def reset(self):
        """
        重置环境到初始状态

        Returns:
            tuple: (observations, global_state, agent_mask)
        """
        try:
            # 确保环境已初始化
            if not self.is_initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize environment")

            # 重置步数
            self.current_step = 0

            # 重新加载SUMO仿真（不关闭GUI）
            if self.verbose:
                print("🔄 重新加载SUMO仿真...")
            try:
                # 使用 traci.load 重新加载配置，不关闭 SUMO
                traci.load(["-c", self.config['cfg_file']])
                if self.verbose:
                    print("   ✅ SUMO仿真已重新加载")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 重新加载失败，尝试完全重启: {e}")
                self.grid_generator.stop_sumo()
                gui_mode = self.config.get('gui_mode', False)
                self.grid_generator.start_sumo(gui=gui_mode)

            # 重置控制器历史
            self.controller.reset_history()

            # 重置任务跟踪状态
            self.reward_calculator.reset_mission_tracking()

            # 重置固定目标车道缓存（智能分配机制）
            self.qminx.reset_fixed_target_lanes()

            # 重置学术评价指标
            self.episode_metrics = {
                'dangerous_approaches': 0,
                'total_lane_changes': 0,
                'speed_samples': [],
                'travel_times': {},
            }

            # 等待车辆进入仿真和栅格区域
            if self.verbose:
                print("Waiting for vehicles to enter simulation...")
            max_wait_steps = 50  # 最多等待50步
            wait_step = 0

            while wait_step < max_wait_steps:
                self.grid_generator.simulate_step()
                self.grid_generator.update_vehicle_positions()

                # 检查是否有车辆进入栅格区域
                active_vehicles = self.qminx.get_active_vehicle_ids()
                if active_vehicles:
                    if self.verbose:
                        print(f"Found {len(active_vehicles)} vehicles in grid area after {wait_step + 1} steps")
                    
                    # 【修复】立即为所有车辆锁定目标车道（基于当前车道）
                    # 避免等待期间车辆换道导致目标分配错误
                    for vid in active_vehicles:
                        self.qminx.get_optimal_target_lane(vid)
                    if self.verbose:
                        print(f"  [FIX] 已为 {len(active_vehicles)} 辆车锁定目标车道")
                    break

                wait_step += 1

            if wait_step >= max_wait_steps:
                if self.verbose:
                    print("⚠️ Warning: No vehicles entered grid area after maximum wait time")
            else:
                if self.verbose:
                    print(f"✅ Vehicles successfully entered grid area")

            # 获取初始观测
            observations, global_state, agent_mask = self._get_observations()

            # 缓存观测数据
            self.last_observations = observations
            self.last_global_state = global_state
            self.last_agent_mask = agent_mask

            active_count = int(agent_mask.sum())
            if self.verbose:
                # 使用换行确保不被 SUMO 状态栏覆盖
                print(f"\n✅ Environment reset. Active vehicles: {active_count}", flush=True)

            # 调试信息：显示车辆状态
            if active_count == 0 and self.verbose:
                print("⚠️ No active vehicles found after reset!")
                all_vehicles = traci.vehicle.getIDList() if self.grid_generator.traci_connected else []
                print(f"Total SUMO vehicles: {len(all_vehicles)}")
                if all_vehicles:
                    print(f"Vehicle IDs: {all_vehicles}")

            return observations, global_state, agent_mask

        except Exception as e:
            print(f"Error resetting environment: {e}")
            return None, None, None

    def step(self, actions):
        """
        执行一步环境交互

        Args:
            actions: 动作数组 [max_agents] (Qmix动作ID: 0-6)

        Returns:
            tuple: (next_observations, global_state, rewards, done, info, agent_mask)
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Environment not initialized. Call reset() first.")

            # 更新步数
            self.current_step += 1
            
            # 同步控制器的步数（用于换道冷却期物理约束）
            self.controller.update_step(self.current_step)

            # 获取活跃车辆
            active_vehicles = self.qminx.get_active_vehicle_ids()

            # 执行动作
            action_results = self._execute_actions(active_vehicles, actions)

            # 推进SUMO仿真
            self.grid_generator.simulate_step()
            self.grid_generator.update_vehicle_positions()

            # 更新车辆状态
            self.qminx.update_vehicle_states()

            # ====== 更新学术评价指标 ======
            self._update_episode_metrics(active_vehicles, actions)

            # 获取下一步观测
            next_observations, next_global_state, agent_mask = self._get_observations()

            # 🔧 修复：在判断done之前，先更新exited_vehicles
            # 原因：如果最后一辆车离开导致栅格清空，_is_done()会立即返回True
            # 但如果update_exited_vehicles()在done判断之后，这辆车就不会被记录
            self.reward_calculator.update_exited_vehicles(self.current_step)

            # 判断是否结束
            done = self._is_done()

            # 计算奖励（集成主线任务奖励和冲突惩罚）
            # 注意：update_exited_vehicles已经在上面调用过了，这里不会重复调用
            reward = self.reward_calculator.calculate_qmix_rewards(
                active_vehicles, action_results, agent_mask,
                current_step=self.current_step, all_vehicles_exited=done
            )

            # 构建信息字典
            info = self._get_info(active_vehicles, action_results, reward, done)

            # 更新缓存
            self.last_observations = next_observations
            self.last_global_state = next_global_state
            self.last_agent_mask = agent_mask

            return next_observations, next_global_state, agent_mask, reward, done, info

        except Exception as e:
            print(f"Error in environment step: {e}")
            return None, None, None, 0.0, True, {}

    def _detect_action_conflicts(self, vehicle_ids, actions):
        """
        步骤1：检测动作冲突

        Args:
            vehicle_ids: 车辆ID列表
            actions: 动作列表

        Returns:
            dict: 冲突检测结果
        """
        conflicts = []
        grid_occupation = {}  # {grid_coord: [vehicle_ids]}

        # 计算每个车辆的目标网格
        for i, vehicle_id in enumerate(vehicle_ids):
            if i >= len(actions):
                continue

            action_id = actions[i]
            # 转换为ActionSpace的动作ID
            sumo_action_id = action_id + 1
            action_info = self.action_space.get_action_info(sumo_action_id)

            if not action_info:
                continue

            # 获取当前网格位置
            current_grid = None
            for grid in self.grid_generator.grids:
                if vehicle_id in grid['vehicle_ids']:
                    current_grid = grid['coordinate']
                    break

            if not current_grid:
                continue

            # 计算目标网格
            i_curr, j_curr = current_grid
            target_grid = (i_curr + action_info['delta_i'], j_curr + action_info['delta_j'])

            # 记录网格占用
            if target_grid not in grid_occupation:
                grid_occupation[target_grid] = []
            grid_occupation[target_grid].append(vehicle_id)

        # 检测冲突
        for grid_coord, occupying_vehicles in grid_occupation.items():
            if len(occupying_vehicles) > 1:
                conflicts.append({
                    'grid': grid_coord,
                    'vehicles': occupying_vehicles
                })

        return {
            'has_conflicts': len(conflicts) > 0,
            'conflicts': conflicts,
            'grid_occupation': grid_occupation
        }

    def _resolve_conflicts_by_priority(self, vehicle_ids, original_actions, conflict_result):
        """
        步骤2：优先级重选机制

        Args:
            vehicle_ids: 车辆ID列表
            original_actions: 原始动作列表
            conflict_result: 冲突检测结果

        Returns:
            tuple: (final_actions, reselection_info)
        """
        final_actions = original_actions.copy()
        reselection_info = {
            'reselected_vehicles': {},  # {vehicle_id: reselection_count}
            'reselected_count': 0
        }

        if not conflict_result['has_conflicts']:
            return final_actions, reselection_info

        # 处理每个冲突
        for conflict in conflict_result['conflicts']:
            conflicting_vehicles = conflict['vehicles']

            # 计算优先级
            priorities = []
            for vehicle_id in conflicting_vehicles:
                current_lane = self.qminx.get_vehicle_current_lane(vehicle_id)
                target_lanes = self.qminx.get_vehicle_target_lanes(vehicle_id)

                if current_lane is not None and target_lanes:
                    # 计算到最近目标车道的距离
                    min_distance = min(abs(current_lane - target_lane) for target_lane in target_lanes)
                    if min_distance == 0:
                        priority = float('inf')  # 已在目标车道，最高优先级
                    else:
                        priority = 1.0 / min_distance
                else:
                    priority = 0.1  # 默认低优先级

                priorities.append({
                    'vehicle_id': vehicle_id,
                    'priority': priority,
                    'vehicle_index': vehicle_ids.index(vehicle_id) if vehicle_id in vehicle_ids else -1
                })

            # 按优先级排序（高到低）
            priorities.sort(key=lambda x: x['priority'], reverse=True)

            # 高优先级车辆保持原动作，低优先级车辆重选
            confirmed_actions = set()

            for i, priority_info in enumerate(priorities):
                vehicle_id = priority_info['vehicle_id']
                vehicle_index = priority_info['vehicle_index']

                if vehicle_index == -1:
                    continue

                if i == 0:  # 最高优先级，保持原动作
                    confirmed_actions.add(vehicle_index)
                    if self.verbose:
                        print(f"   车辆{vehicle_id}: 优先级最高，保持原动作{original_actions[vehicle_index]}")
                else:  # 低优先级，需要重选
                    reselection_count = 0
                    max_reselections = 3

                    while reselection_count < max_reselections:
                        # 重选动作
                        available_actions = [0, 1, 2, 3]  # 所有可能动作 (已更新为4个)
                        import random
                        new_action = random.choice(available_actions)

                        # 检查是否与已确定动作冲突
                        if self._check_action_conflict_with_confirmed(
                                vehicle_id, new_action, vehicle_ids, final_actions, confirmed_actions
                        ):
                            reselection_count += 1
                            continue
                        else:
                            # 重选成功
                            final_actions[vehicle_index] = new_action
                            confirmed_actions.add(vehicle_index)
                            reselection_info['reselected_vehicles'][vehicle_id] = reselection_count + 1
                            reselection_info['reselected_count'] += 1
                            if self.verbose:
                                print(f"   车辆{vehicle_id}: 重选{reselection_count + 1}次，新动作{new_action}")
                            break

                    if reselection_count >= max_reselections:
                        # 重选失败，强制Stay (Stay现在是action 4, QMIX ID是3)
                        final_actions[vehicle_index] = 3
                        confirmed_actions.add(vehicle_index)
                        reselection_info['reselected_vehicles'][vehicle_id] = max_reselections
                        reselection_info['reselected_count'] += 1
                        if self.verbose:
                            print(f"   车辆{vehicle_id}: 重选超限，强制Stay")

        return final_actions, reselection_info

    def _check_action_conflict_with_confirmed(self, vehicle_id, action, vehicle_ids, actions, confirmed_indices):
        """
        检查动作是否与已确定的动作冲突
        """
        # 获取当前车辆的网格位置
        current_grid = None
        for grid in self.grid_generator.grids:
            if vehicle_id in grid['vehicle_ids']:
                current_grid = grid['coordinate']
                break

        if not current_grid:
            return False

        # 计算目标网格
        sumo_action_id = action + 1
        action_info = self.action_space.get_action_info(sumo_action_id)
        if not action_info:
            return False

        i_curr, j_curr = current_grid
        target_grid = (i_curr + action_info['delta_i'], j_curr + action_info['delta_j'])

        # 检查与已确定动作的冲突
        for confirmed_idx in confirmed_indices:
            if confirmed_idx >= len(vehicle_ids) or confirmed_idx >= len(actions):
                continue

            confirmed_vehicle_id = vehicle_ids[confirmed_idx]
            confirmed_action = actions[confirmed_idx]

            # 获取已确定车辆的目标网格
            confirmed_current_grid = None
            for grid in self.grid_generator.grids:
                if confirmed_vehicle_id in grid['vehicle_ids']:
                    confirmed_current_grid = grid['coordinate']
                    break

            if not confirmed_current_grid:
                continue

            confirmed_sumo_action_id = confirmed_action + 1
            confirmed_action_info = self.action_space.get_action_info(confirmed_sumo_action_id)
            if not confirmed_action_info:
                continue

            i_conf, j_conf = confirmed_current_grid
            confirmed_target_grid = (
                i_conf + confirmed_action_info['delta_i'], j_conf + confirmed_action_info['delta_j'])

            # 检查目标网格是否相同
            if target_grid == confirmed_target_grid:
                return True  # 有冲突

        return False  # 无冲突

    def _get_observations(self):
        """
        获取当前观测数据

        Returns:
            tuple: (observations, global_state, agent_mask) - 所有数据为torch.Tensor格式
        """
        try:
            # 获取批量观测数据
            observations, agent_mask = self.qminx.get_qmix_batch_data(self.max_agents)

            # 获取全局状态
            global_state = self.qminx.get_global_state_for_qmix()

            # 转换为torch.Tensor格式
            observations = torch.tensor(observations, dtype=torch.float32)
            global_state = torch.tensor(global_state, dtype=torch.float32)
            agent_mask = torch.tensor(agent_mask, dtype=torch.float32)

            return observations, global_state, agent_mask

        except Exception as e:
            print(f"Error getting observations: {e}")
            # 返回默认值（torch.Tensor格式）- 更新：22→23维，新增冷却期状态
            observations = torch.zeros((self.max_agents, 23), dtype=torch.float32)
            global_state = torch.zeros(4, dtype=torch.float32)
            agent_mask = torch.zeros(self.max_agents, dtype=torch.float32)
            return observations, global_state, agent_mask

    def _execute_actions(self, active_vehicles, actions):
        """
        执行批量动作，包含冲突检测和处理机制

        基于学习的冲突解决策略：
        - 不再使用人为规则重选动作
        - 当发生冲突时（多车争夺同一格），强制所有涉及车辆 Stay
        - 配合连续的拥挤度惩罚，引导网络自主学习避让

        Args:
            active_vehicles: 活跃车辆ID列表
            actions: 动作数组

        Returns:
            dict: 动作执行结果，包含冲突信息
        """
        try:
            # 只对活跃车辆执行动作
            original_actions = actions[:len(active_vehicles)]

            # 步骤1：冲突检测
            conflict_detection_result = self._detect_action_conflicts(active_vehicles, original_actions)

            # 步骤2：优先级重选机制
            # 使用基于距离优先级的冲突解决策略
            # 确保高优先级的车辆（如更接近目标的）能优先执行动作，避免死锁
            final_actions, reselection_info = self._resolve_conflicts_by_priority(
                active_vehicles, original_actions, conflict_detection_result
            )

            # 步骤3：批量执行最终动作
            action_results = self.controller.execute_batch_actions(
                active_vehicles, final_actions
            )

            # 构造结果，保留冲突信息供调试和统计
            enhanced_results = {
                'action_results': action_results,
                'conflict_info': {
                    'had_conflicts': conflict_detection_result['has_conflicts'],
                    'conflicts': conflict_detection_result['conflicts'],
                    'original_actions': original_actions,
                    'final_actions': final_actions,
                    'reselection_info': reselection_info
                }
            }

            return enhanced_results

        except Exception as e:
            print(f"Error executing actions: {e}")
            return {
                'action_results': {vid: False for vid in active_vehicles},
                'conflict_info': {
                    'had_conflicts': False,
                    'conflicts': [],
                    'original_actions': actions[:len(active_vehicles)],
                    'final_actions': actions[:len(active_vehicles)]
                }
            }

    def _is_done(self):
        """
        判断episode是否结束

        新逻辑：
        1. 如果栅格区域内没有活跃车辆
        2. 且已经有车辆通过过栅格区域（exited_vehicles不为空）
        3. 则认为Episode完成

        这避免了持续车流导致Episode永不结束的问题

        Returns:
            bool: 是否结束
        """
        # 获取栅格区域内的活跃车辆
        active_vehicles = self.qminx.get_active_vehicle_ids()

        # 检查是否有车辆已经离开过栅格区域
        has_exited_vehicles = len(self.reward_calculator.exited_vehicles) > 0

        # 如果栅格区域已经没有车了，且之前有车辆通过过，Episode结束
        if not active_vehicles and has_exited_vehicles:
            # 🔍 调试信息：检查车辆去向（仅 verbose 模式）
            if self.verbose:
                try:
                    all_sumo_vehicles = traci.vehicle.getIDList() if self.grid_generator.traci_connected else []
                    exited_ids = list(self.reward_calculator.exited_vehicles.keys())
                    all_seen_ids = list(self.reward_calculator.all_seen_vehicles)

                    print(f"🏁 栅格区域清空，已有 {len(self.reward_calculator.exited_vehicles)} 辆车完成")
                    print(f"   📊 车辆统计:")
                    print(f"      - 栅格内活跃: {len(active_vehicles)} 辆")
                    print(f"      - 已离开栅格: {len(exited_ids)} 辆 {exited_ids}")
                    print(f"      - SUMO路网中: {len(all_sumo_vehicles)} 辆 {all_sumo_vehicles}")
                    print(f"      - 历史见过: {len(all_seen_ids)} 辆 {all_seen_ids}")

                    # 找出"失踪"的车辆
                    missing = set(all_seen_ids) - set(exited_ids)
                    if missing:
                        print(f"   ⚠️ 未记录为离开的车辆: {missing}")
                    else:
                        print(f"   ✅ 所有车辆都已正确记录！")
                except Exception as e:
                    print(f"🏁 栅格区域清空，已有 {len(self.reward_calculator.exited_vehicles)} 辆车完成")

            return True

        # 兜底：如果达到最大步数，也结束
        if self.current_step >= self.max_steps:
            # 🔍 调试信息：为什么达到max_steps？（仅 verbose 模式）
            if self.verbose:
                try:
                    all_sumo_vehicles = traci.vehicle.getIDList() if self.grid_generator.traci_connected else []
                    exited_ids = list(self.reward_calculator.exited_vehicles.keys())
                    all_seen_ids = list(self.reward_calculator.all_seen_vehicles)

                    print(f"⏱️ 达到最大步数 {self.max_steps}，Episode结束")
                    print(f"   📊 车辆统计（超时结束）:")
                    print(f"      - 栅格内活跃: {len(active_vehicles)} 辆 {active_vehicles}")
                    print(f"      - 已离开栅格: {len(exited_ids)} 辆 {exited_ids}")
                    print(f"      - SUMO路网中: {len(all_sumo_vehicles)} 辆 {all_sumo_vehicles}")
                    print(f"      - 历史见过: {len(all_seen_ids)} 辆 {all_seen_ids}")

                    # 诊断：为什么没有触发栅格清空？
                    if active_vehicles:
                        print(f"   ⚠️ 原因：栅格内还有 {len(active_vehicles)} 辆车未离开")
                    elif not has_exited_vehicles:
                        print(f"   ⚠️ 原因：没有车辆离开过栅格（可能都卡在入口）")
                    else:
                        print(f"   ⚠️ 原因：未知（逻辑异常）")
                except Exception as e:
                    print(f"⏱️ 达到最大步数 {self.max_steps}，Episode结束")

            return True

        return False

    def _update_episode_metrics(self, active_vehicles, actions):
        """
        更新学术评价指标（每步调用，性能优化版）

        优化：
        - 删除 speed_samples 采样（数据量大，用处小）
        - 删除 getLeader 重复调用（已在奖励函数中调用）
        - 危险接近计数改为从奖励计算器获取（避免重复 traci 调用）
        """
        try:
            # 1. 记录车辆进入时间（仅首次进入时）
            for vehicle_id in active_vehicles:
                if vehicle_id not in self.episode_metrics['travel_times']:
                    self.episode_metrics['travel_times'][vehicle_id] = {
                        'enter_step': self.current_step,
                        'exit_step': None
                    }

            # 2. 统计换道动作（简单计数，开销极小）
            for i, action in enumerate(actions):
                if i < len(active_vehicles) and action in [1, 2]:  # Left or Right
                    self.episode_metrics['total_lane_changes'] += 1

            # 3. 更新已离开车辆的退出时间
            if hasattr(self.reward_calculator, 'newly_exited_this_step'):
                for vid in self.reward_calculator.newly_exited_this_step:
                    if vid in self.episode_metrics['travel_times']:
                        self.episode_metrics['travel_times'][vid]['exit_step'] = self.current_step

        except Exception as e:
            pass  # 指标收集失败不影响主流程

    def _get_info(self, active_vehicles, action_results, rewards, done=False):
        """
        构建信息字典

        Args:
            active_vehicles: 活跃车辆列表
            action_results: 动作执行结果
            rewards: 奖励数组

        Returns:
            dict: 信息字典
        """
        try:
            # 基础信息
            info = {
                'step': self.current_step,
                'active_vehicles': len(active_vehicles),
                'total_reward': rewards if isinstance(rewards, (int, float)) else np.sum(rewards),
                'avg_reward': rewards if isinstance(rewards, (int, float)) else (
                    np.mean(rewards[rewards != 0]) if np.any(rewards != 0) else 0.0)
            }

            # 动作执行统计
            if action_results and 'action_results' in action_results:
                actual_results = action_results['action_results']
                successful_actions = sum(1 for success in actual_results.values() if success)
                info['action_success_rate'] = successful_actions / len(actual_results) if actual_results else 0.0

                # 添加冲突信息
                if 'conflict_info' in action_results:
                    conflict_info = action_results['conflict_info']
                    info['had_conflicts'] = conflict_info.get('had_conflicts', False)

                    # 计算被强制Stay的车辆数 (物理阻挡)
                    # 通过比较 original_actions 和 final_actions
                    original = conflict_info.get('original_actions', [])
                    final = conflict_info.get('final_actions', [])
                    # 只要动作变了，就是被阻挡了
                    blocked_count = sum(1 for o, f in zip(original, final) if o != f)
                    info['reselected_vehicles'] = blocked_count
            else:
                info['action_success_rate'] = 0.0
                info['had_conflicts'] = False
                info['reselected_vehicles'] = 0

            # 添加本步新退出的车辆信息（用于 IPPO 个体奖励）
            if hasattr(self.reward_calculator, 'newly_exited_this_step'):
                info['newly_exited_vehicles'] = self.reward_calculator.newly_exited_this_step.copy()
            else:
                info['newly_exited_vehicles'] = {}

            # 车辆状态统计 - 基于任务完成状态
            total_exited = len(self.reward_calculator.exited_vehicles) if hasattr(self.reward_calculator,
                                                                                  'exited_vehicles') else 0
            vehicles_in_target = 0

            if done and total_exited > 0:
                # Episode 结束时：基于所有已退出车辆
                vehicles_in_target = sum(
                    1 for v_info in self.reward_calculator.exited_vehicles.values()
                    if v_info['in_target_lane']
                )
                info['target_lane_rate'] = vehicles_in_target / total_exited
                if self.verbose:
                    print(f"🎯 [INFO] Episode结束统计: {vehicles_in_target}/{total_exited} 车辆在目标车道离开")
            elif active_vehicles:
                # Episode 进行中：基于当前仍在栅格中的车辆
                vehicles_in_target = sum(
                    1 for vid in active_vehicles
                    if self.qminx.is_vehicle_in_target_lane(vid)
                )
                info['target_lane_rate'] = vehicles_in_target / len(active_vehicles)
            else:
                info['target_lane_rate'] = 0.0

            # 新增：严格的“10 车全部成功”统计，用于 IPPO / QMIX 成功率计算
            required_agents = self.max_agents
            info['total_exited_vehicles'] = total_exited
            info['vehicles_in_target_lane'] = vehicles_in_target
            info['required_agents'] = required_agents
            info['success_all_agents'] = (
                    total_exited >= required_agents and vehicles_in_target >= required_agents
            )
            # 使用固定分母（required_agents）计算一个更严格的成功率指标
            info['target_lane_rate_all'] = (
                vehicles_in_target / required_agents if required_agents > 0 else 0.0
            )

            # ====== 学术评价指标（新增 2025-11-30，性能优化版）======
            # 1. 危险接近次数（从奖励计算器获取，避免重复traci调用）
            info['dangerous_approaches'] = getattr(
                self.reward_calculator, 'dangerous_approach_count', 0
            )

            # 2. 换道次数
            info['total_lane_changes'] = self.episode_metrics.get('total_lane_changes', 0)

            # 3. 平均通过时间（仅在 episode 结束时计算）
            if done:
                travel_times = self.episode_metrics.get('travel_times', {})
                completed_times = []
                for vid, times in travel_times.items():
                    if times['exit_step'] is not None:
                        travel_time = times['exit_step'] - times['enter_step']
                        completed_times.append(travel_time)
                info['avg_travel_time'] = np.mean(completed_times) if completed_times else 0.0
                info['completed_vehicles'] = len(completed_times)
            else:
                info['avg_travel_time'] = 0.0
                info['completed_vehicles'] = 0

            return info

        except Exception as e:
            print(f"Error creating info dict: {e}")
            return {'step': self.current_step, 'error': str(e)}

    def get_observation_space(self):
        """
        获取观测空间信息

        Returns:
            dict: 观测空间描述
        """
        return {
            'local_obs_dim': 23,  # 每个智能体的观测维度（更新：22→23，新增冷却期状态）
            'global_state_dim': 4,  # 全局状态维度
            'n_agents': self.max_agents,  # 最大智能体数量
            'action_dim': 4  # 动作空间大小 (已更新)
        }

    def get_action_space(self):
        """
        获取动作空间信息

        Returns:
            dict: 动作空间描述
        """
        return {
            'n_actions': 4,
            'action_names': [
                'Forward', 'Turn_Left', 'Turn_Right', 'Stay'
            ],
            'discrete': True
        }

    def render(self, mode='human'):
        """
        渲染环境（可选）

        Args:
            mode: 渲染模式
        """
        if mode == 'human':
            # 打印当前状态信息
            active_vehicles = self.qminx.get_active_vehicle_ids()
            print(f"\nStep {self.current_step}: {len(active_vehicles)} active vehicles")

            if active_vehicles:
                stats = self.reward_calculator.get_reward_statistics(active_vehicles)
                print(f"Avg Reward: {stats['avg_reward']:.3f}")
                print(f"Vehicles in target: {stats.get('positive_rewards', 0)}/{len(active_vehicles)}")

    def close(self):
        """
        关闭环境，清理资源
        """
        try:
            if self.grid_generator:
                self.grid_generator.stop_sumo()
            print("✅ Qmix Environment closed.")
        except Exception as e:
            # 忽略关闭时的错误（SUMO 可能已经关闭）
            pass

    def get_env_info(self):
        """
        获取环境信息（兼容PyMARL格式）

        Returns:
            dict: 环境信息
        """
        return {
            'state_shape': 4,  # 全局状态维度
            'obs_shape': 18,  # 局部观测维度
            'n_actions': 4,  # 动作数量 (已更新)
            'n_agents': self.max_agents,  # 智能体数量
            'episode_limit': self.max_steps  # episode最大长度
        }


import traci
from .config_manager import ConfigManager  # 添加导入


def create_default_config():
    """
    创建默认配置
    """
    return {
        'net_file': 'sumo/net.net.xml',
        'cfg_file': ConfigManager.DEFAULT_SUMO_CONFIG_PATH,
        #'gui_mode': True,
        'gui_mode': False,
        # 批判性修改：
        # 原 max_steps=200 (60s) 太短，容易导致任务被强行截断。
        # 现改为 500 (150s)，给予智能体充足的时间完成任务。
        'max_steps': 500,
        'max_agents': 10
    }


def run_control_demo():
    """
    运行车辆控制策略演示
    """
    print("🚀 启动车辆控制策略演示...")

    # 创建环境
    config = create_default_config()
    env = QmixEnvironment(config)

    try:
        # 初始化环境
        print("🔧 初始化环境...")
        env.initialize()

        # 重置环境，让车辆进入
        env.reset()
        print("✅ 环境初始化完成，开始控制演示")

        # 持续运行控制演示
        step = 0
        while True:
            step += 1
            print(f"\n⏰ === Step {step} ===")

            # 检查SUMO中是否还有车辆
            all_vehicles = traci.vehicle.getIDList() if env.grid_generator.traci_connected else []
            if not all_vehicles:
                print("🏁 所有车辆已从路网中消失，演示结束")
                break

            # 获取栅格区域内的活跃车辆
            active_vehicles = env.qminx.get_active_vehicle_ids()

            if not active_vehicles:
                print("📍 栅格区域无车辆，但路网中还有 {} 辆车在行驶，继续仿真...".format(len(all_vehicles)))
                # 只推进仿真，不执行动作选择
                env.grid_generator.simulate_step()
                env.grid_generator.update_vehicle_positions()
                env.qminx.update_vehicle_states()
                continue

            print(f"🚗 当前活跃车辆: {active_vehicles}")

            # 获取观测数据
            observations, global_state, agent_mask = env._get_observations()

            # 使用Qmix模型选择动作
            actions_tensor, action_info = env.qmix_model.select_actions(
                observations=observations.unsqueeze(0),  # 添加batch维度
                global_state=global_state.unsqueeze(0),  # 添加batch维度
                agent_mask=agent_mask.unsqueeze(0),  # 添加batch维度
                epsilon=0.3,  # 探索率
                method='epsilon_greedy'
            )

            # 转换为列表格式
            actions = actions_tensor[0].numpy().tolist()  # 移除batch维度并转换为列表

            print(f"🤖 [QMIX] 动作选择完成: {actions[:len(active_vehicles)]}")
            print(f"   探索率: {action_info['epsilon']:.3f}, 方法: {action_info['method']}")

            # 执行动作（调用我们的控制系统）
            next_obs, next_global, agent_mask, rewards, done, info = env.step(actions)

            # 显示执行结果
            active_count = int(sum(agent_mask))  # 转换为 int
            target_lane_rate = info.get('target_lane_rate', 0.0)
            had_conflicts = info.get('had_conflicts', False)
            reselected_vehicles = info.get('reselected_vehicles', 0)
            total_reward = info.get('total_reward', 0.0)

            print(f"📊 执行结果: 活跃车辆{active_count}, 目标车道率{target_lane_rate:.1%}")
            print(f"   冲突情况: {'有冲突' if had_conflicts else '无冲突'}, 重选车辆{reselected_vehicles}辆")
            print(f"   本步奖励: {total_reward:.1f}")

            # 检查是否结束 - 所有车辆离开栅格区域时自动结束
            if done:  # 环境判断结束（所有车辆离开或到达目标）
                print(f"🏁 控制演示完成！所有车辆已离开栅格区域 (步数: {step})")
                break

        # 显示动作统计报告（诊断工具）
        print(env.controller.get_action_statistics_report())

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
        # 即使中断也显示统计
        try:
            print(env.controller.get_action_statistics_report())
        except:
            pass
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
    finally:
        # 清理资源
        env.close()
        print("🧹 资源已清理")


if __name__ == "__main__":
    # 修复Windows控制台编码问题
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    run_control_demo()