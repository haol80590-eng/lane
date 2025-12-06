"""
动作空间定义模块
定义车辆在栅格系统中的6种可执行动作
"""

class ActionSpace:
    def __init__(self, max_grid_sequence=87):
        """
        初始化动作空间
        
        Args:
            max_grid_sequence: 最大栅格序号（默认为87，保持兼容性）
        """
        # 动作定义
        self.actions = {
            1: {
                'name': 'Forward',
                'description': '直行',
                'lateral_velocity': 0,
                'longitudinal_velocity': 1,
                'delta_i': 0,
                'delta_j': 1
            },
            2: {
                'name': 'Turn_Left',
                'description': '左转',
                'lateral_velocity': 1,
                'longitudinal_velocity': 1,
                'delta_i': 1,
                'delta_j': 1
            },
            3: {
                'name': 'Turn_Right',
                'description': '右转',
                'lateral_velocity': -1,
                'longitudinal_velocity': 1,
                'delta_i': -1,
                'delta_j': 1
            },
            4: {
                'name': 'Stay',
                'description': '跟随车流',
                'lateral_velocity': 0,
                'longitudinal_velocity': 0,
                'delta_i': 0,
                'delta_j': 0
            }
        }
        
        # 动作索引列表
        self.action_indices = list(self.actions.keys())
        
        # 栅格系统约束
        self.min_lane_index = 0  # 最小车道索引
        self.max_lane_index = 3  # 最大车道索引
        self.min_grid_sequence = 1  # 最小栅格序号
        self.max_grid_sequence = max_grid_sequence  # 最大栅格序号
    
    def get_action_info(self, action_id):
        """
        获取动作信息
        
        Args:
            action_id: 动作ID (1-7)
            
        Returns:
            dict: 动作信息字典
        """
        return self.actions.get(action_id, None)
    
    def get_action_name(self, action_id):
        """
        获取动作名称
        
        Args:
            action_id: 动作ID
            
        Returns:
            str: 动作名称
        """
        action = self.get_action_info(action_id)
        return action['name'] if action else None
    
    def get_action_description(self, action_id):
        """
        获取动作描述
        
        Args:
            action_id: 动作ID
            
        Returns:
            str: 动作描述
        """
        action = self.get_action_info(action_id)
        return action['description'] if action else None
    
    def get_action_delta(self, action_id):
        """
        获取动作的栅格坐标变化量
        
        Args:
            action_id: 动作ID
            
        Returns:
            tuple: (delta_i, delta_j) 栅格坐标变化量
        """
        action = self.get_action_info(action_id)
        if action:
            return (action['delta_i'], action['delta_j'])
        return None
    

    
    def get_action_by_name(self, action_name):
        """
        根据动作名称获取动作ID
        
        Args:
            action_name: 动作名称
            
        Returns:
            int: 动作ID，如果未找到返回None
        """
        for action_id, action_info in self.actions.items():
            if action_info['name'] == action_name:
                return action_id
        return None
    
    def print_action_space(self):
        """
        打印动作空间信息
        """
        print("=== Action Space Definition ===")
        print("SUMO Lane Layout (South to North):")
        print("  Lane 0 (南) ← Y=-12.25")
        print("  Lane 1      ← Y=-8.75")  
        print("  Lane 2      ← Y=-5.25")
        print("  Lane 3 (北) ← Y=-1.75")
        print()
        
        print(f"Total actions: {len(self.actions)}")
        print("\nAction details:")
        for action_id, action_info in self.actions.items():
            print(f"  {action_id}: {action_info['name']} ({action_info['description']})")
            print(f"      Lateral: {action_info['lateral_velocity']}, Longitudinal: {action_info['longitudinal_velocity']}")
            print(f"      Grid delta: (Δi={action_info['delta_i']}, Δj={action_info['delta_j']})")
            
            # 添加方向说明
            if action_info['delta_i'] > 0:
                direction = "向北(车道索引增大)"
            elif action_info['delta_i'] < 0:
                direction = "向南(车道索引减小)"
            else:
                direction = "保持车道"
            print(f"      Direction: {direction}")
        
        print(f"\nGrid constraints:")
        print(f"  Lane index range: [{self.min_lane_index}, {self.max_lane_index}] (南→北)")
        print(f"  Grid sequence range: [{self.min_grid_sequence}, {self.max_grid_sequence}] (起点→终点)")
        print("=" * 50)
    
    def get_action_matrix(self):
        """
        获取动作矩阵表示
        
        Returns:
            dict: 动作矩阵字典
        """
        matrix = {}
        for action_id, action_info in self.actions.items():
            matrix[action_id] = {
                'name': action_info['name'],
                'delta': (action_info['delta_i'], action_info['delta_j']),
                'velocity': (action_info['lateral_velocity'], action_info['longitudinal_velocity'])
            }
        return matrix 