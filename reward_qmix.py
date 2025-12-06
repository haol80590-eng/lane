"""
QMIX 奖励计算模块（已弃用，保留备份）

此文件包含 QMIX 训练使用的全局奖励函数，
已从主 reward.py 中分离，仅供参考。
"""

import numpy as np
import traci


class QmixRewardCalculator:
    """QMIX 全局奖励计算器"""
    
    def __init__(self, qminx_network):
        """
        初始化奖励计算器
        
        Args:
            qminx_network: QminxNetwork实例
        """
        self.qminx = qminx_network
        
        # 🎯 奖励参数配置 - 基于势能场理论的连续奖励 (Potential-Based Reward Shaping)
        self.reward_config = {
            # 📈 纵向进度奖励权重
            'progress_reward': 50.0,
            
            # 🚗 车道偏离惩罚权重  
            'lane_deviation_penalty_scale': 1.0,
            
            # 🎯 目标车道完成奖励
            'target_lane_completion_bonus': 60.0,
            
            # ⏳ 时间惩罚（生存成本）
            'existence_cost': 0.005,
            
            # ⚠️ 无效动作惩罚
            'invalid_action_penalty': -0.1,
            
            # 💥 碰撞惩罚
            'collision_penalty': -10.0,
            
            # 🚗 速度奖励参数
            'speed_amplitude': 0.6,
            'speed_threshold': 10.0,
            'speed_transition_width': 3.0,
        }
        
        # 状态跟踪
        self.exited_vehicles = {}
        self.previous_active_vehicles = set()
        self.all_seen_vehicles = set()
        self.vehicle_target_status = {}
        self.previous_potentials = {}
        self.newly_exited_this_step = {}
        self.vehicle_lane_deviation_cache = {}

    def calculate_lane_distance_reward(self, vehicle_id, action_id=None):
        """
        🎯 渐进式换道压力 (Progressive Lane Change Pressure)
        """
        try:
            current_lane = self.qminx.get_vehicle_current_lane(vehicle_id)
            if current_lane is None: return 0.0
            
            target_lane = self.qminx.get_optimal_target_lane(vehicle_id)
            if target_lane is None: return 0.0
            
            lane_deviation = abs(target_lane - current_lane)
            
            if lane_deviation == 0:
                return 0.5
            
            progress = self.calculate_progress_ratio(vehicle_id)
            
            # Sigmoid 光滑压力系数
            p_min, p_max, p_center, k_steep = 0.3, 2.5, 0.4, 10.0
            sigmoid_val = 1.0 / (1.0 + np.exp(-k_steep * (progress - p_center)))
            pressure_factor = p_min + (p_max - p_min) * sigmoid_val
            
            penalty_scale = self.reward_config.get('lane_deviation_penalty_scale', 1.0)
            
            if not hasattr(self, 'vehicle_lane_deviation_cache'):
                self.vehicle_lane_deviation_cache = {}
            self.vehicle_lane_deviation_cache[vehicle_id] = lane_deviation
            
            return -penalty_scale * lane_deviation * pressure_factor
                
        except Exception as e:
            print(f"Error calculating lane distance reward for vehicle {vehicle_id}: {e}")
            return 0.0

    def calculate_progress_ratio(self, vehicle_id):
        """计算车辆在栅格区域内的进度比例 (0.0 - 1.0)"""
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

    def calculate_progress_reward_delta(self, vehicle_id):
        """计算进度势能差分奖励 (Delta Reward)"""
        current_progress = self.calculate_progress_ratio(vehicle_id)
        
        if vehicle_id not in self.previous_potentials:
            self.previous_potentials[vehicle_id] = {'progress': current_progress}
            return 0.0
        
        prev_progress = self.previous_potentials[vehicle_id].get('progress', 0.0)
        delta_progress = current_progress - prev_progress
        
        if delta_progress < 0:
            delta_progress = 0.0
            
        self.previous_potentials[vehicle_id]['progress'] = current_progress
        
        scale = self.reward_config.get('progress_reward', 50.0)
        return scale * delta_progress

    def calculate_speed_reward(self, vehicle_id):
        """🚗 计算速度奖励/惩罚（连续可微 Sigmoid 函数）"""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return 0.0
            
            v = traci.vehicle.getSpeed(vehicle_id)
            
            A = self.reward_config.get('speed_amplitude', 0.6)
            v0 = self.reward_config.get('speed_threshold', 10.0)
            k = self.reward_config.get('speed_transition_width', 3.0)
            
            x = (v - v0) / k
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            
            r_speed = A * sigmoid - A / 2.0
            
            return r_speed
            
        except Exception as e:
            return 0.0

    def reset_mission_tracking(self):
        """重置所有状态"""
        self.exited_vehicles = {}
        self.previous_active_vehicles = set()
        self.all_seen_vehicles = set()
        self.vehicle_target_status = {}
        self.previous_potentials = {}
        self.newly_exited_this_step = {}
        self.vehicle_lane_deviation_cache = {}

    def _calculate_global_reward(self, vehicle_ids, action_results, all_vehicles_exited):
        """
        🎯 三项势能场奖励 (Three-Term Potential-Based Reward)
        
        R_t = R_progress + R_lane + R_speed - Cost + Bonus
        """
        if not vehicle_ids:
            return 0.0
        
        total_progress_delta_reward = 0.0
        total_lane_potential_reward = 0.0
        total_speed_reward = 0.0
        
        w_cost = self.reward_config.get('existence_cost', 0.01)
        
        for vehicle_id in vehicle_ids:
            prog_reward = self.calculate_progress_reward_delta(vehicle_id)
            total_progress_delta_reward += prog_reward
            
            lane_reward = self.calculate_lane_distance_reward(vehicle_id)
            total_lane_potential_reward += lane_reward
            
            speed_reward = self.calculate_speed_reward(vehicle_id)
            total_speed_reward += speed_reward
            
        total_cost = w_cost * len(vehicle_ids)
        
        target_lane_bonus = 0.0
        
        if hasattr(self, 'newly_exited_this_step') and self.newly_exited_this_step:
            w_target = self.reward_config.get('target_lane_completion_bonus', 50.0)
            
            for vid, info in self.newly_exited_this_step.items():
                if info.get('in_target_lane', False):
                    target_lane_bonus += w_target
            
        total_reward_sum = (total_progress_delta_reward + total_lane_potential_reward + 
                          total_speed_reward - total_cost + target_lane_bonus)
        
        scaling_factor = 10.0
        normalized_reward = total_reward_sum / scaling_factor
        
        return normalized_reward

    def calculate_qmix_rewards(self, vehicle_ids, action_results=None, agent_mask=None, 
                              current_step=None, all_vehicles_exited=False):
        """计算Qmix格式的全局奖励"""
        max_agents = len(agent_mask) if agent_mask is not None else 10
        
        global_reward = self._calculate_global_reward(vehicle_ids, action_results, all_vehicles_exited)
        
        rewards = np.full(max_agents, global_reward, dtype=np.float32)
        
        if agent_mask is not None:
            if hasattr(agent_mask, 'cpu'):
                agent_mask = agent_mask.cpu().numpy()
            rewards = rewards * agent_mask
        
        return global_reward

    def update_exited_vehicles(self, current_step):
        """更新已驶离栅格区域的车辆记录"""
        current_active = set(self.qminx.get_active_vehicle_ids())
        
        for vehicle_id in current_active:
            try:
                in_target = self.qminx.is_vehicle_in_target_lane(vehicle_id)
                self.vehicle_target_status[vehicle_id] = in_target
                
                current_lane = self.qminx.get_vehicle_current_lane(vehicle_id)
                target_lane = self.qminx.get_optimal_target_lane(vehicle_id)
                if current_lane is not None and target_lane is not None:
                    lane_deviation = abs(target_lane - current_lane)
                else:
                    lane_deviation = 0.0
                    
                if not hasattr(self, 'vehicle_lane_deviation_cache'):
                    self.vehicle_lane_deviation_cache = {}
                self.vehicle_lane_deviation_cache[vehicle_id] = lane_deviation
                
            except: 
                self.vehicle_target_status[vehicle_id] = False
            
        newly_exited = self.previous_active_vehicles - current_active
        
        self.newly_exited_this_step = {}
        
        for vehicle_id in newly_exited:
            if vehicle_id not in self.exited_vehicles:
                in_target = self.vehicle_target_status.get(vehicle_id, False)
                lane_deviation = getattr(self, 'vehicle_lane_deviation_cache', {}).get(vehicle_id, 0.0 if in_target else 2.0)
                
                exit_info = {
                    'in_target_lane': in_target, 
                    'exit_step': current_step,
                    'lane_deviation': lane_deviation
                }
                self.exited_vehicles[vehicle_id] = exit_info
                self.newly_exited_this_step[vehicle_id] = exit_info
                
                self.vehicle_target_status.pop(vehicle_id, None)
                if hasattr(self, 'vehicle_lane_deviation_cache'):
                    self.vehicle_lane_deviation_cache.pop(vehicle_id, None)
                self.previous_potentials.pop(vehicle_id, None)
                
                try:
                    if vehicle_id in traci.vehicle.getIDList():
                        traci.vehicle.setSpeedMode(vehicle_id, 31)
                        traci.vehicle.setLaneChangeMode(vehicle_id, 256)
                        traci.vehicle.setSpeed(vehicle_id, -1)
                except Exception as e:
                    pass
                
        self.previous_active_vehicles = current_active.copy()
        self.all_seen_vehicles.update(current_active)

    def calculate_mission_completion_reward(self, all_vehicles_exited=False):
        if not all_vehicles_exited or not self.exited_vehicles: return 0.0
        total = len(self.exited_vehicles)
        success = sum(1 for v in self.exited_vehicles.values() if v['in_target_lane'])
        return 0.0

    def update_reward_config(self, new_config):
        self.reward_config.update(new_config)
        print(f"Updated reward config: {self.reward_config}")
