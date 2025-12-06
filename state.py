import numpy as np
import traci
from typing import Dict, List, Tuple, Optional, Any
from .grid import SUMOGridGenerator


class QminxNetwork:
    def __init__(self, grid_generator: 'SUMOGridGenerator', vehicle_controller=None):
        """
        初始化Qminx网络

        Args:
            grid_generator: 栅格生成器实例
            vehicle_controller: 车辆控制器实例（用于获取冷却期状态）
        """
        self.grid_generator = grid_generator
        self.vehicle_controller = vehicle_controller

        # 目标车道函数定义
        self.target_lane_function = self._build_target_lane_function()

        # 车辆状态存储
        self.vehicle_states: Dict[str, Dict[str, Any]] = {}

        # 横向位置历史记录（用于计算横向速度）
        self.vehicle_lateral_positions: Dict[str, List[float]] = {}

        # 固定目标车道缓存（智能分配，防止反复横跳）
        self.fixed_target_lanes: Dict[str, int] = {}

    # =====================================
    # 1. 初始化和配置方法
    # =====================================

    def _build_target_lane_function(self) -> Dict[str, Any]:
        """
        动态构建车辆目标车道函数 T(vehicle_id)
        基于车辆的路由目的地动态解析转向意图

        Returns:
            dict: 车辆ID到目标车道的映射
        """
        # 基础目标车道映射规则
        self.lane_mapping_rules = {
            'L': [3],  # 左转 -> 车道3 (最北侧)
            'R': [0],  # 右转 -> 车道0 (最南侧)
            'S': [1, 2]  # 直行 -> 车道1,2 (中间)
        }

        # 路由目的地到转向意图的映射
        self.destination_intent_map = {
            'E2': 'L',  # 到E2为左转
            'E3': 'R',  # 到E3为右转
            '-E1': 'S'  # 到-E1为直行
        }

        return {}

    def update_destination_mapping(self, destination_intent_map):
        """
        更新路由目的地到转向意图的映射规则

        Args:
            destination_intent_map: dict, 目的地到转向意图的映射
                                   例如: {'E2': 'L', 'E3': 'R', '-E1': 'S'}
        """
        self.destination_intent_map.update(destination_intent_map)
        print(f"Updated destination mapping: {self.destination_intent_map}")

    def update_lane_mapping_rules(self, lane_mapping_rules):
        """
        更新转向意图到目标车道的映射规则

        Args:
            lane_mapping_rules: dict, 转向意图到目标车道的映射
                               例如: {'L': [3], 'R': [0], 'S': [1, 2]}
        """
        self.lane_mapping_rules.update(lane_mapping_rules)
        print(f"Updated lane mapping rules: {self.lane_mapping_rules}")

    # =====================================
    # 2. 车辆基础信息获取
    # =====================================

    def get_vehicle_current_lane(self, vehicle_id: str) -> Optional[int]:
        """
        获取车辆当前车道索引

        Args:
            vehicle_id: 车辆ID

        Returns:
            int: 当前车道索引 (0-3)，如果未找到返回None
        """
        # 遍历所有栅格，找到包含该车辆的栅格
        for grid in self.grid_generator.grids:
            if vehicle_id in grid['vehicle_ids']:
                return grid['lane_index']
        return None

    def _get_vehicle_grid_position(self, vehicle_id):
        """
        获取车辆的栅格位置坐标

        Args:
            vehicle_id: 车辆ID

        Returns:
            tuple: 栅格坐标 (i, j)，如果未找到返回None
        """
        for grid in self.grid_generator.grids:
            if vehicle_id in grid['vehicle_ids']:
                return grid['coordinate']
        return None

    def _get_vehicle_route_destination(self, vehicle_id):
        """
        动态获取车辆路由目的地

        Args:
            vehicle_id: 车辆ID

        Returns:
            str: 目的地边ID，如果未找到返回None
        """
        try:
            if self.grid_generator.traci_connected:
                # 获取车辆路由
                route = traci.vehicle.getRoute(vehicle_id)
                if route and len(route) > 1:
                    return route[-1]  # 返回最后一个边作为目的地
        except Exception as e:
            print(f"Error getting route for vehicle {vehicle_id}: {e}")
        return None

    def _determine_vehicle_intent(self, vehicle_id):
        """
        根据车辆路由动态确定转向意图

        Args:
            vehicle_id: 车辆ID

        Returns:
            str: 转向意图 ('L', 'R', 'S', 'U')
        """
        destination = self._get_vehicle_route_destination(vehicle_id)
        if destination and destination in self.destination_intent_map:
            return self.destination_intent_map[destination]
        return 'U'  # Undefined

    def get_vehicle_target_lanes(self, vehicle_id):
        """
        动态获取车辆目标车道索引列表

        Args:
            vehicle_id: 车辆ID

        Returns:
            list: 目标车道索引列表，如果未找到返回空列表
        """
        intent = self._determine_vehicle_intent(vehicle_id)
        if intent in self.lane_mapping_rules:
            return self.lane_mapping_rules[intent]
        return []

    def get_optimal_target_lane(self, vehicle_id):
        """
        智能分配固定目标车道
        
        【学术设计原则】：最小换道原则 (Minimum Lane Change)
        - 如果车辆已在合法目标车道内，就以当前车道为目标（无需换道）
        - 如果车辆不在合法车道，选择最近的合法车道
        - 一旦分配，缓存不变（避免反复横跳）
        
        这符合真实驾驶行为：驾驶员倾向于保持当前车道，
        除非必须换道才能完成转弯。

        Args:
            vehicle_id: 车辆ID

        Returns:
            int: 目标车道索引，如果未找到返回None
        """
        # 1. 检查缓存：如果已分配过，直接返回
        if vehicle_id in self.fixed_target_lanes:
            return self.fixed_target_lanes[vehicle_id]

        # 2. 首次查询：智能分配目标车道
        target_lanes = self.get_vehicle_target_lanes(vehicle_id)
        if not target_lanes:
            return None

        # 左转/右转：只有一个目标车道
        if len(target_lanes) == 1:
            fixed_lane = target_lanes[0]
        else:
            # 直行车辆：目标车道 = [1, 2]
            # 【最小换道原则】：如果已在合法车道，保持当前车道
            try:
                current_lane = traci.vehicle.getLaneIndex(vehicle_id)

                if current_lane in target_lanes:
                    # 已在合法车道（1或2），保持当前车道，无需换道
                    fixed_lane = current_lane
                else:
                    # 不在合法车道（0或3），选择最近的合法车道
                    # 车道0 → 目标1，车道3 → 目标2
                    if current_lane < min(target_lanes):
                        fixed_lane = min(target_lanes)  # 车道0 → 1
                    else:
                        fixed_lane = max(target_lanes)  # 车道3 → 2

            except traci.exceptions.TraCIException:
                fixed_lane = target_lanes[0]

        # 3. 缓存并返回
        self.fixed_target_lanes[vehicle_id] = fixed_lane
        return fixed_lane

    def reset_fixed_target_lanes(self):
        """重置固定目标车道缓存和横向位置历史（在episode开始时调用）"""
        self.fixed_target_lanes = {}
        self.vehicle_lateral_positions = {}

    def get_vehicle_intent(self, vehicle_id):
        """
        动态获取车辆转向意图

        Args:
            vehicle_id: 车辆ID

        Returns:
            str: 转向意图 ('L', 'R', 'S')，如果未找到返回'U'
        """
        return self._determine_vehicle_intent(vehicle_id)

    def get_lateral_velocity(self, vehicle_id):
        """
        计算车辆的横向速度（用于连续换道惩罚）

        横向速度定义为车道内横向位置的变化率，单位为米/步

        Args:
            vehicle_id: 车辆ID

        Returns:
            float: 横向速度（米/步），首次调用返回0.0
        """
        try:
            # 获取当前横向位置（相对于车道中心线，单位：米）
            current_lateral_pos = traci.vehicle.getLateralLanePosition(vehicle_id)

            # 如果是首次记录，初始化并返回0
            if vehicle_id not in self.vehicle_lateral_positions:
                self.vehicle_lateral_positions[vehicle_id] = current_lateral_pos
                return 0.0

            # 计算横向速度（当前位置 - 上一步位置）
            previous_lateral_pos = self.vehicle_lateral_positions[vehicle_id]
            lateral_velocity = current_lateral_pos - previous_lateral_pos

            # 更新记录
            self.vehicle_lateral_positions[vehicle_id] = current_lateral_pos

            return lateral_velocity

        except traci.exceptions.TraCIException:
            # 车辆不存在或已离开，返回0
            return 0.0

    # =====================================
    # 3. 全局状态感知
    # =====================================

    def get_grid_occupancy_ratio(self, grid_i, grid_j):
        """
        获取指定网格的占用率，处理边界情况

        Args:
            grid_i: 车道索引
            grid_j: 纵向网格序号

        Returns:
            float: 占用率 [0,1]
        """
        # 车道边界检查
        if grid_i < 0 or grid_i > 3:
            return 1.0  # 边界外视为完全占用，阻止越界

        # 纵向边界检查
        if grid_j < 1 or grid_j > self.grid_generator.grid_length_count:
            return 0.0  # 道路端点视为空，允许通过

        # 查找对应网格
        for grid in self.grid_generator.grids:
            if grid['coordinate'] == (grid_i, grid_j):
                vehicle_count = len(grid['vehicle_ids'])
                max_capacity = grid.get('max_capacity', 1)  # 默认容量为1
                return min(vehicle_count / max_capacity, 1.0)

        return 0.0  # 未找到网格，视为空

    def get_grid_vehicle_count(self, grid_i, grid_j):
        """
        获取指定网格的车辆数量（原始数值，非占用率）

        Args:
            grid_i: 车道索引
            grid_j: 纵向网格序号

        Returns:
            int: 车辆数量 (0, 1, 2, ...)
        """
        # 车道边界检查
        if grid_i < 0 or grid_i > 3:
            return 999  # 边界外视为极大值，避免选择

        # 纵向边界检查
        if grid_j < 1 or grid_j > self.grid_generator.grid_length_count:
            return 0  # 道路端点视为空

        # 查找对应网格
        for grid in self.grid_generator.grids:
            if grid['coordinate'] == (grid_i, grid_j):
                return len(grid['vehicle_ids'])  # 直接返回车辆数量

        return 0  # 未找到网格，视为空

    def get_all_active_vehicles_routes(self):
        """
        获取所有活跃车辆的路由信息（用于调试和分析）

        Returns:
            dict: 车辆ID到路由信息的映射
        """
        vehicle_routes = {}
        if self.grid_generator.traci_connected:
            try:
                vehicle_ids = traci.vehicle.getIDList()
                for vehicle_id in vehicle_ids:
                    route = traci.vehicle.getRoute(vehicle_id)
                    destination = self._get_vehicle_route_destination(vehicle_id)
                    intent = self._determine_vehicle_intent(vehicle_id)
                    vehicle_routes[vehicle_id] = {
                        'route': route,
                        'destination': destination,
                        'intent': intent
                    }
            except Exception as e:
                print(f"Error getting vehicle routes: {e}")
        return vehicle_routes

    # =====================================
    # 4. 局部状态感知 (Qmix专用)
    # =====================================

    def get_local_grid_occupancy(self, vehicle_id):
        """
        获取车辆周围3×5网格的占用状态
        纵向5格：后2格 + 当前 + 前2格 (符合驾驶习惯)
        横向3格：左1格 + 当前车道 + 右1格

        Args:
            vehicle_id: 车辆ID

        Returns:
            list: 15维局部网格占用状态 [0,1]
        """
        current_pos = self._get_vehicle_grid_position(vehicle_id)
        if not current_pos:
            return [0] * 15

        i, j = current_pos  # i是车道索引(横向), j是纵向位置
        local_state = []

        # 3×5感知窗口布局
        # 纵向：j-2, j-1, j, j+1, j+2 (后→前)
        # 横向：i-1, i, i+1 (左→右)

        for dj in [-2, -1, 0, 1, 2]:  # 纵向偏移(后→前)
            for di in [-1, 0, 1]:  # 横向偏移(左→右)
                grid_i, grid_j = i + di, j + dj
                occupancy = self.get_grid_occupancy_ratio(grid_i, grid_j)
                local_state.append(occupancy)

        return local_state  # 15维

    # =====================================
    # 5. Qmix数据转换和管理
    # =====================================

    def convert_to_qmix_obs(self, vehicle_id):
        """
        1. 数据格式转换: 字典 → 数值向量
        将车辆状态转换为Qmix可用的23维观测向量

        更新（2025-12-03）：从22维扩展到23维，新增换道冷却期状态
        
        观测结构 (23维):
        - [0:3]   自身状态: 当前车道、目标车道、是否需换道
        - [3:18]  局部栅格: 3×5 网格占用状态
        - [18:20] 邻车意图: 左邻车想右换道、右邻车想左换道
        - [20:22] TTC安全: 前车TTC、目标车道最小TTC
        - [22]    冷却期: 换道冷却期剩余时间（归一化）

        Args:
            vehicle_id: 车辆ID

        Returns:
            np.array: 23维观测向量 [0,1]
        """
        # 获取原始状态
        current_lane = self.get_vehicle_current_lane(vehicle_id)
        target_lanes = self.get_vehicle_target_lanes(vehicle_id)
        local_grid = self.get_local_grid_occupancy(vehicle_id)

        # 转换为数值向量
        obs = []

        # 1. 当前车道归一化 (1维)
        obs.append(current_lane / 3.0 if current_lane is not None else 0.0)

        # 2. 最优目标车道归一化 (1维) - 基于前方占用情况动态选择
        optimal_target = self.get_optimal_target_lane(vehicle_id)
        if optimal_target is None:
            optimal_target = target_lanes[0] if target_lanes else current_lane
        obs.append(optimal_target / 3.0 if optimal_target is not None else 0.0)

        # 3. 是否需要换道 (1维)
        need_change = 1.0 if current_lane != optimal_target else 0.0
        obs.append(need_change)

        # 4. 局部网格感知 (15维) - 已经是[0,1]格式
        obs.extend(local_grid)

        # 5. 邻车意图特征（新增 2025-11-29，2维）
        # 解决"交叉冲突对峙"问题：让Agent能看到邻车是否想换到自己车道
        left_neighbor_wants_right, right_neighbor_wants_left = self._get_neighbor_intentions(vehicle_id, current_lane)
        obs.append(left_neighbor_wants_right)  # 左邻车是否想右换道到我的车道
        obs.append(right_neighbor_wants_left)  # 右邻车是否想左换道到我的车道

        # 6. TTC安全特征（新增 2025-12-02，2维）
        # 让Agent能感知碰撞风险，学会安全换道
        ttc_front, ttc_target = self._calculate_ttc_features(vehicle_id, optimal_target)
        obs.append(ttc_front)   # 与当前车道前车的TTC（归一化）
        obs.append(ttc_target)  # 目标车道最小TTC（归一化）

        # 7. 换道冷却期状态（新增 2025-12-03，1维）
        # 让Agent知道自己是否可以换道，避免无效尝试
        cooldown_ratio = self._get_lane_change_cooldown_ratio(vehicle_id)
        obs.append(cooldown_ratio)  # 冷却期剩余时间比例 [0,1]

        return np.array(obs, dtype=np.float32)  # 23维向量

    def _get_neighbor_intentions(self, vehicle_id, current_lane):
        """
        获取相邻车道邻车的换道意图（新增 2025-11-29）

        Args:
            vehicle_id: 当前车辆ID
            current_lane: 当前车道索引

        Returns:
            tuple: (左邻车想右换道, 右邻车想左换道) 值为0.0或1.0
        """
        left_wants_right = 0.0  # 左边车道(lane+1)的车是否想换到我的车道
        right_wants_left = 0.0  # 右边车道(lane-1)的车是否想换到我的车道

        if current_lane is None:
            return left_wants_right, right_wants_left

        # 获取本车的栅格位置
        my_grid_j = None
        for grid in self.grid_generator.grids:
            if vehicle_id in grid['vehicle_ids']:
                my_grid_j = grid['coordinate'][1]
                break

        if my_grid_j is None:
            return left_wants_right, right_wants_left

        # 检查左边车道 (lane + 1) 的邻车
        left_lane = current_lane + 1
        if left_lane <= 3:  # 最大车道索引为3
            for grid in self.grid_generator.grids:
                if grid['lane_index'] == left_lane and abs(grid['coordinate'][1] - my_grid_j) <= 2:
                    for other_id in grid['vehicle_ids']:
                        if other_id != vehicle_id:
                            other_target = self.get_optimal_target_lane(other_id)
                            if other_target == current_lane:  # 想换到我的车道
                                left_wants_right = 1.0
                                break
                if left_wants_right > 0:
                    break

        # 检查右边车道 (lane - 1) 的邻车
        right_lane = current_lane - 1
        if right_lane >= 0:  # 最小车道索引为0
            for grid in self.grid_generator.grids:
                if grid['lane_index'] == right_lane and abs(grid['coordinate'][1] - my_grid_j) <= 2:
                    for other_id in grid['vehicle_ids']:
                        if other_id != vehicle_id:
                            other_target = self.get_optimal_target_lane(other_id)
                            if other_target == current_lane:  # 想换到我的车道
                                right_wants_left = 1.0
                                break
                if right_wants_left > 0:
                    break

        return left_wants_right, right_wants_left

    def _calculate_ttc_features(self, vehicle_id: str, target_lane: int) -> Tuple[float, float]:
        """
        计算TTC（Time To Collision）安全特征（新增 2025-12-02）
        
        学术依据：Hayward (1972), "Near-miss determination through use of a scale of danger"
        
        TTC = distance / relative_speed
        
        归一化方法：ttc_normalized = 1 / (1 + TTC / τ)
        - TTC → 0:  ttc_normalized → 1 (危险)
        - TTC → ∞:  ttc_normalized → 0 (安全)
        - τ = 5.0秒：归一化时间常数
        
        Args:
            vehicle_id: 车辆ID
            target_lane: 目标车道索引
            
        Returns:
            tuple: (ttc_front_normalized, ttc_target_normalized)
        """
        TAU = 5.0  # 归一化时间常数（秒），基于 Euro NCAP 安全阈值
        
        ttc_front = float('inf')
        ttc_target = float('inf')
        
        try:
            # 获取本车信息
            my_speed = traci.vehicle.getSpeed(vehicle_id)
            my_pos = traci.vehicle.getPosition(vehicle_id)
            
            # === 1. 计算与当前车道前车的 TTC ===
            leader_info = traci.vehicle.getLeader(vehicle_id, dist=50.0)
            if leader_info is not None and leader_info[0] != '':
                leader_id, distance = leader_info
                try:
                    leader_speed = traci.vehicle.getSpeed(leader_id)
                    relative_speed = my_speed - leader_speed
                    
                    # 只有当我比前车快时才计算TTC
                    if relative_speed > 0.1:  # 避免除零，阈值0.1 m/s
                        ttc_front = distance / relative_speed
                except:
                    pass
            
            # === 2. 计算目标车道的最小 TTC ===
            current_lane = self.get_vehicle_current_lane(vehicle_id)
            
            # 只有当目标车道与当前车道不同时才计算
            if target_lane is not None and current_lane is not None and target_lane != current_lane:
                # 构建目标车道ID
                edge_id = traci.vehicle.getRoadID(vehicle_id)
                target_lane_id = f"{edge_id}_{target_lane}"
                
                try:
                    vehicles_on_target = traci.lane.getLastStepVehicleIDs(target_lane_id)
                    
                    for other_id in vehicles_on_target:
                        if other_id == vehicle_id:
                            continue
                        
                        other_pos = traci.vehicle.getPosition(other_id)
                        other_speed = traci.vehicle.getSpeed(other_id)
                        
                        # 纵向距离（正=前方，负=后方）
                        longitudinal_dist = other_pos[0] - my_pos[0]
                        
                        if longitudinal_dist > 0:
                            # 前方车辆：计算我追上它的TTC
                            relative_speed = my_speed - other_speed
                            if relative_speed > 0.1:
                                ttc = longitudinal_dist / relative_speed
                                ttc_target = min(ttc_target, ttc)
                        else:
                            # 后方车辆：计算它追上我的TTC
                            relative_speed = other_speed - my_speed
                            if relative_speed > 0.1:
                                ttc = abs(longitudinal_dist) / relative_speed
                                ttc_target = min(ttc_target, ttc)
                except:
                    pass
            
        except Exception as e:
            pass  # 出错时返回安全值（inf → 归一化后为0）
        
        # 归一化：ttc_normalized = 1 / (1 + TTC / τ)
        # TTC小 → 值趋近1（危险），TTC大 → 值趋近0（安全）
        ttc_front_normalized = 1.0 / (1.0 + ttc_front / TAU) if ttc_front != float('inf') else 0.0
        ttc_target_normalized = 1.0 / (1.0 + ttc_target / TAU) if ttc_target != float('inf') else 0.0
        
        return ttc_front_normalized, ttc_target_normalized

    def get_active_vehicle_ids(self):
        """
        3. 智能体管理: 动态数量 → 固定数量
        获取当前活跃的车辆ID列表（固定顺序）

        Returns:
            list: 活跃车辆ID列表
        """
        active_vehicles = []
        for grid in self.grid_generator.grids:
            for vehicle_id in grid['vehicle_ids']:
                if vehicle_id not in active_vehicles:
                    active_vehicles.append(vehicle_id)

        # 按ID排序保证一致性
        return sorted(active_vehicles)

    def get_qmix_batch_data(self, max_agents=10):
        """
        2. 批量数据接口: 单车 → 多车批处理
        3. 智能体管理: 动态数量 → 固定数量 (方案二+方案三)
        获取Qmix需要的批量数据

        Args:
            max_agents: 最大智能体数量

        Returns:
            tuple: (observations, agent_mask)
                observations: [max_agents, 23] 观测数据（更新：22→23维，新增冷却期状态）
                agent_mask: [max_agents] 智能体掩码 (1=真实, 0=填充)
        """
        # 获取当前活跃车辆
        active_vehicles = self.get_active_vehicle_ids()

        observations = []
        agent_mask = []

        # 方案二：填充到固定维度
        for i in range(max_agents):
            if i < len(active_vehicles):
                # 真实车辆
                vehicle_id = active_vehicles[i]
                obs = self.convert_to_qmix_obs(vehicle_id)
                observations.append(obs)
                agent_mask.append(1)  # 标记为真实智能体
            else:
                # 填充位置（更新：22→23维，新增冷却期状态）
                observations.append(np.zeros(23, dtype=np.float32))
                agent_mask.append(0)  # 标记为填充智能体

        return np.array(observations), np.array(agent_mask)

    def get_global_state_for_qmix(self):
        """
        获取Qmix mixing network需要的全局状态

        Returns:
            np.array: 4维全局状态向量
        """
        # 车道占用密度 (4维)
        lane_densities = []
        for i in range(4):
            lane_grids = [g for g in self.grid_generator.grids if g['lane_index'] == i]
            occupied = sum(1 for g in lane_grids if g['vehicle_count'] > 0)
            density = occupied / len(lane_grids) if lane_grids else 0.0
            lane_densities.append(density)

        return np.array(lane_densities, dtype=np.float32)

    # =====================================
    # 6. 车辆状态管理
    # =====================================

    def update_vehicle_states(self):
        """
        更新所有车辆的状态 Si = (Ci, Ti)
        """
        self.vehicle_states = {}

        # 获取所有活跃车辆
        for grid in self.grid_generator.grids:
            for vehicle_id in grid['vehicle_ids']:
                if vehicle_id not in self.vehicle_states:
                    # 获取当前车道索引 Ci
                    current_lane = self.get_vehicle_current_lane(vehicle_id)

                    # 获取目标车道索引 Ti
                    target_lanes = self.get_vehicle_target_lanes(vehicle_id)

                    # 获取转向意图
                    intent = self.get_vehicle_intent(vehicle_id)

                    # 构建车辆状态 Si
                    self.vehicle_states[vehicle_id] = {
                        'Ci': current_lane,  # 当前车道索引
                        'Ti': target_lanes,  # 目标车道索引列表
                        'intent': intent,  # 转向意图
                        'grid_position': self._get_vehicle_grid_position(vehicle_id)
                    }

    def get_vehicle_state(self, vehicle_id):
        """
        获取指定车辆的状态

        Args:
            vehicle_id: 车辆ID

        Returns:
            dict: 车辆状态 Si = (Ci, Ti, intent, position)
        """
        return self.vehicle_states.get(vehicle_id, None)

    def get_all_vehicle_states(self):
        """
        获取所有车辆的状态

        Returns:
            dict: 所有车辆状态字典
        """
        return self.vehicle_states.copy()

    def is_vehicle_in_target_lane(self, vehicle_id):
        """
        判断车辆是否在目标车道（基于所有可能的目标车道）

        Args:
            vehicle_id: 车辆ID

        Returns:
            bool: True如果在目标车道，False否则
        """
        vehicle_state = self.get_vehicle_state(vehicle_id)
        if vehicle_state is None:
            return False

        current_lane = vehicle_state['Ci']
        target_lanes = vehicle_state['Ti']

        return current_lane in target_lanes

    def is_vehicle_in_optimal_target_lane(self, vehicle_id):
        """
        判断车辆是否在最优目标车道（基于前方占用情况动态选择的目标车道）

        Args:
            vehicle_id: 车辆ID

        Returns:
            bool: True如果在最优目标车道，False否则
        """
        current_lane = self.get_vehicle_current_lane(vehicle_id)
        optimal_target = self.get_optimal_target_lane(vehicle_id)

        if current_lane is None or optimal_target is None:
            return False

        return current_lane == optimal_target

    def get_lane_change_requirement(self, vehicle_id):
        """
        获取车辆变道需求

        Args:
            vehicle_id: 车辆ID

        Returns:
            dict: 变道需求信息
        """
        vehicle_state = self.get_vehicle_state(vehicle_id)
        if vehicle_state is None:
            return None

        current_lane = vehicle_state['Ci']
        target_lanes = vehicle_state['Ti']
        intent = vehicle_state['intent']

        if current_lane in target_lanes:
            return {
                'need_change': False,
                'current_lane': current_lane,
                'target_lanes': target_lanes,
                'intent': intent
            }
        else:
            # 选择最优目标车道（这里简单选择第一个）
            optimal_target = target_lanes[0] if target_lanes else current_lane
            direction = 'left' if optimal_target > current_lane else 'right'

            return {
                'need_change': True,
                'current_lane': current_lane,
                'target_lanes': target_lanes,
                'optimal_target': optimal_target,
                'direction': direction,
                'intent': intent
            }

    # =====================================
    # 6. 统计和调试功能
    # =====================================

    def get_statistics(self):
        """
        获取车辆状态统计信息

        Returns:
            dict: 统计信息
        """
        total_vehicles = len(self.vehicle_states)
        vehicles_in_target = sum(1 for vid in self.vehicle_states if self.is_vehicle_in_target_lane(vid))
        vehicles_need_change = total_vehicles - vehicles_in_target

        # 按转向意图分类统计
        intent_stats = {}
        for vehicle_id, state in self.vehicle_states.items():
            intent = state['intent']
            if intent not in intent_stats:
                intent_stats[intent] = {'total': 0, 'in_target': 0}
            intent_stats[intent]['total'] += 1
            if self.is_vehicle_in_target_lane(vehicle_id):
                intent_stats[intent]['in_target'] += 1

        return {
            'total_vehicles': total_vehicles,
            'vehicles_in_target_lane': vehicles_in_target,
            'vehicles_need_lane_change': vehicles_need_change,
            'intent_statistics': intent_stats
        }

    def print_vehicle_states(self):
        """
        打印所有车辆状态信息
        """
        print("\n=== Vehicle States ===")
        for vehicle_id, state in self.vehicle_states.items():
            target_str = str(state['Ti']) if len(state['Ti']) > 1 else str(state['Ti'][0])
            print(
                f"Vehicle {vehicle_id}: Ci={state['Ci']}, Ti={target_str}, Intent={state['intent']}, Grid={state['grid_position']}")

    def print_specific_vehicle_info(self, target_vehicle_id="l2", verbose=False):
        """
        打印指定车辆的详细信息

        Args:
            target_vehicle_id: 目标车辆ID，默认为"l2"
            verbose: 是否显示详细信息，False时只显示关键信息
        """
        try:
            # 获取当前SUMO仿真信息
            if self.grid_generator.traci_connected:
                current_time = traci.simulation.getTime()  # 仿真时间(秒)
                current_step = traci.simulation.getCurrentTime()  # 仿真步数(毫秒)
                step_length = traci.simulation.getDeltaT()  # 步长(秒)
            else:
                current_time = 0
                current_step = 0
                step_length = 0.6

            # 获取车辆基本信息
            current_lane = self.get_vehicle_current_lane(target_vehicle_id)
            target_lanes = self.get_vehicle_target_lanes(target_vehicle_id)
            intent = self.get_vehicle_intent(target_vehicle_id)
            grid_position = self._get_vehicle_grid_position(target_vehicle_id)
            in_target = self.is_vehicle_in_target_lane(target_vehicle_id)

            # 只有在 verbose=True 时才打印
            if verbose:
                if not verbose:  # 简洁输出格式（这个条件永远不会执行，但保留原逻辑）
                    target_str = str(target_lanes[0]) if len(target_lanes) == 1 else str(target_lanes)
                    status = "✓" if in_target else "✗"
                    # 使用毫秒步数除以1000再除以步长得到真实步数
                    real_step = int(current_step / 1000 / step_length) if step_length > 0 else int(current_time / 0.6)
                    print(
                        f"Step {real_step:3d} (t={current_time:.1f}s) | {target_vehicle_id}: Ci={current_lane}, Ti={target_str}, Intent={intent}, Grid={grid_position}, InTarget={status}")
                else:
                    # 详细输出格式
                    print(f"\n=== Step {int(current_time)} - Vehicle {target_vehicle_id} ===")
                    print(f"Current Lane (Ci): {current_lane}")
                    print(f"Target Lanes (Ti): {target_lanes}")
                    print(f"Turning Intent: {intent}")
                    print(f"Grid Position: {grid_position}")
                    print(f"In Target Lane: {in_target}")

                # 获取变道需求
                lane_change_req = self.get_lane_change_requirement(target_vehicle_id)
                if lane_change_req and lane_change_req['need_change']:
                    print(
                        f"Lane Change: {lane_change_req['current_lane']} → {lane_change_req['optimal_target']} ({lane_change_req['direction']})")

                print("=" * 50)

        except Exception as e:
            print(f"Error getting info for {target_vehicle_id}: {e}")

    def _get_lane_change_cooldown_ratio(self, vehicle_id):
        """
        获取车辆换道冷却期状态（新增 2025-12-03）
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            float: 冷却期剩余时间比例 [0,1]
                  - 0.0: 没有冷却期限制，可以换道
                  - >0.0: 冷却期中，值越大剩余时间越长
        """
        try:
            if self.vehicle_controller is None:
                return 0.0  # 没有控制器引用，假设可以换道
            
            # 检查是否在冷却期中
            if hasattr(self.vehicle_controller, '_is_lane_change_allowed'):
                if self.vehicle_controller._is_lane_change_allowed(vehicle_id):
                    return 0.0  # 可以换道
                else:
                    # 计算剩余冷却期比例
                    if hasattr(self.vehicle_controller, 'vehicle_last_lc_step') and \
                       hasattr(self.vehicle_controller, 'lane_change_cooldown') and \
                       hasattr(self.vehicle_controller, 'current_step'):
                        
                        last_lc_step = self.vehicle_controller.vehicle_last_lc_step.get(vehicle_id, 0)
                        current_step = self.vehicle_controller.current_step
                        cooldown_duration = self.vehicle_controller.lane_change_cooldown
                        
                        steps_since_lc = current_step - last_lc_step
                        remaining_steps = cooldown_duration - steps_since_lc
                        
                        if remaining_steps > 0:
                            # 归一化到 [0,1]，1表示刚换道，0表示冷却期结束
                            return min(remaining_steps / cooldown_duration, 1.0)
                    
                    return 1.0  # 在冷却期但无法计算具体剩余时间
            else:
                return 0.0  # 控制器没有冷却期功能，假设可以换道
                
        except Exception as e:
            return 0.0  # 出错时假设可以换道