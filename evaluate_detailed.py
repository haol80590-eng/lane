"""
详细评估脚本 - 记录直行车辆的完整轨迹信息
用于分析车辆行为和换道决策
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 获取脚本所在目录的绝对路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_DRL_DIR = _SCRIPT_DIR.parent

# 添加DRL目录到Python路径（用于common包的导入）
if str(_DRL_DIR) not in sys.path:
    sys.path.insert(0, str(_DRL_DIR))

# 切换工作目录
os.chdir(_SCRIPT_DIR)

import torch
import numpy as np
import traci

# 导入本地模块
from ippo_model import IPPOModel
from common.qmix_env import QmixEnvironment
from common.config_manager import ConfigManager


# 动作名称映射
ACTION_NAMES = {
    0: 'Forward (直行)',
    1: 'Turn_Left (左换道)',
    2: 'Turn_Right (右换道)',
    3: 'Stay (保持)'
}


class DetailedEvaluator:
    """详细评估器 - 记录车辆完整轨迹"""
    
    def __init__(self, experiment_dir: str, model_name: str = 'final_model.pt'):
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型和环境
        self._load_model()
        self._setup_environment()
        
        # 车辆轨迹记录
        self.vehicle_trajectories = {}
        
    def _load_model(self):
        """加载模型"""
        # 加载配置
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        args = config.get('args', {})
        network_config = config.get('config', {}).get('network', {})
        
        # 获取模型参数
        n_agents = args.get('n_agents', 10)
        obs_dim = network_config.get('obs_dim', 20)
        action_dim = network_config.get('action_dim', 4)
        hidden_dim = args.get('hidden_dim', 256)
        
        self.model = IPPOModel(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            shared_params=args.get('shared_params', True),
            device=self.device
        )
        
        # 使用IPPOModel的load方法加载权重
        model_path = self.experiment_dir / 'models' / self.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model.load(str(model_path))
        self.model.eval()
        
        # 加载观测标准化参数
        model_stem = Path(self.model_name).stem
        rms_path = self.experiment_dir / 'models' / f'rms_{model_stem}.json'
        if rms_path.exists():
            with open(rms_path, 'r') as f:
                rms_data = json.load(f)
            self.obs_mean = np.array(rms_data['mean'])
            self.obs_var = np.array(rms_data['var'])
        else:
            self.obs_mean = np.zeros(obs_dim)
            self.obs_var = np.ones(obs_dim)
        
        print(f"[OK] 模型已加载: {model_path}")
    
    def _setup_environment(self):
        """设置环境"""
        # 使用固定路径（与evaluate_ippo.py保持一致）
        env_config = {
            'net_file': r'C:\Users\user\Desktop\lxpaper\sumo\net.net.xml',
            'cfg_file': r'C:\Users\user\Desktop\lxpaper\sumo\car.sumocfg',
            'max_agents': 10,
            'max_steps': 500,
            'gui_mode': True  # 使用GUI模式
        }
        
        self.env = QmixEnvironment(config=env_config)
        print("[OK] 环境已初始化 (GUI模式)")
    
    def _normalize_obs(self, obs):
        """标准化观测"""
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
    
    def _get_vehicle_details(self, vehicle_id):
        """获取车辆的详细信息"""
        try:
            # 基础信息
            position = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            lane_position = traci.vehicle.getLanePosition(vehicle_id)
            
            # 路由信息
            route = traci.vehicle.getRoute(vehicle_id)
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            
            # 前车信息
            leader_info = traci.vehicle.getLeader(vehicle_id, 100)
            if leader_info:
                leader_id, leader_dist = leader_info
            else:
                leader_id, leader_dist = None, None
            
            # 获取栅格位置
            grid_pos = self._get_grid_position(vehicle_id)
            
            return {
                'position_x': position[0],
                'position_y': position[1],
                'speed_mps': speed,
                'speed_kmh': speed * 3.6,
                'lane_id': lane_id,
                'lane_index': lane_index,
                'lane_position': lane_position,
                'route': route,
                'route_index': route_index,
                'destination': route[-1] if route else None,
                'leader_id': leader_id,
                'leader_distance': leader_dist,
                'grid_position': grid_pos
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_grid_position(self, vehicle_id):
        """获取车辆在栅格中的位置"""
        try:
            for grid in self.env.qminx.grid_generator.grids:
                if vehicle_id in grid.get('vehicle_ids', []):
                    return {
                        'lane_index': grid['lane_index'],
                        'grid_sequence': grid['grid_sequence'],
                        'coordinate': grid['coordinate']
                    }
        except:
            pass
        return None
    
    def _get_target_lane_info(self, vehicle_id):
        """获取车辆的目标车道信息"""
        try:
            target_lanes = self.env.qminx.get_vehicle_target_lanes(vehicle_id)
            optimal_target = self.env.qminx.get_optimal_target_lane(vehicle_id)
            intent = self.env.qminx._determine_vehicle_intent(vehicle_id)
            
            return {
                'intent': intent,
                'target_lanes': target_lanes,
                'optimal_target_lane': optimal_target
            }
        except:
            return None
    
    def run_detailed_episode(self):
        """运行一个episode并记录详细信息"""
        print("\n" + "=" * 70)
        print("开始详细评估 - 记录直行车辆轨迹")
        print("=" * 70)
        
        # 重置环境和轨迹记录
        self.vehicle_trajectories = {}
        obs_tuple = self.env.reset()
        raw_obs_tensor, global_state, agent_mask = obs_tuple
        
        # 标准化观测
        obs_np = raw_obs_tensor.cpu().numpy()
        norm_obs = self._normalize_obs(obs_np)
        obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=self.device)
        agent_mask = agent_mask.to(self.device)
        
        step = 0
        done = False
        
        # 识别直行车辆 (s1, s2, s3, s4)
        straight_vehicles = ['s1', 's2', 's3', 's4']
        
        print(f"\n监控直行车辆: {straight_vehicles}")
        print("-" * 70)
        
        while not done:
            step += 1
            
            # 获取当前活跃车辆
            try:
                active_vehicles = self.env.qminx.get_active_vehicle_ids()
            except:
                active_vehicles = []
            
            # 选择动作
            action_mask = torch.ones(self.model.n_agents, self.model.action_dim, device=self.device)
            with torch.no_grad():
                actions, _, _ = self.model.select_actions(
                    obs_tensor, agent_mask, action_mask, deterministic=True
                )
            actions_list = actions.cpu().numpy().tolist()
            
            # 记录每个直行车辆的信息
            for i, vid in enumerate(active_vehicles):
                if vid in straight_vehicles:
                    # 获取详细信息
                    details = self._get_vehicle_details(vid)
                    target_info = self._get_target_lane_info(vid)
                    
                    # 获取该车辆的动作
                    action = actions_list[i] if i < len(actions_list) else -1
                    action_name = ACTION_NAMES.get(action, f'Unknown({action})')
                    
                    # 初始化轨迹记录
                    if vid not in self.vehicle_trajectories:
                        self.vehicle_trajectories[vid] = {
                            'vehicle_id': vid,
                            'type': 'straight',
                            'destination': details.get('destination'),
                            'target_info': target_info,
                            'trajectory': []
                        }
                    
                    # 记录当前步信息
                    step_info = {
                        'step': step,
                        'action': action,
                        'action_name': action_name,
                        'lane_index': details.get('lane_index'),
                        'grid_position': details.get('grid_position'),
                        'speed_mps': details.get('speed_mps'),
                        'speed_kmh': details.get('speed_kmh'),
                        'position_x': details.get('position_x'),
                        'lane_position': details.get('lane_position'),
                        'leader_id': details.get('leader_id'),
                        'leader_distance': details.get('leader_distance')
                    }
                    self.vehicle_trajectories[vid]['trajectory'].append(step_info)
                    
                    # 实时打印
                    grid_pos = details.get('grid_position')
                    grid_str = f"({grid_pos['lane_index']}, {grid_pos['grid_sequence']})" if grid_pos else "离开栅格"
                    
                    print(f"Step {step:3d} | {vid} | "
                          f"车道:{details.get('lane_index', '?')} | "
                          f"栅格:{grid_str:12s} | "
                          f"速度:{details.get('speed_kmh', 0):.1f}km/h | "
                          f"动作:{action_name}")
            
            # 执行动作
            step_result = self.env.step(actions_list)
            next_raw_obs, _, next_agent_mask, _, done, info = step_result
            
            if not done:
                next_obs_np = next_raw_obs.cpu().numpy()
                norm_obs = self._normalize_obs(next_obs_np)
                obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=self.device)
                agent_mask = next_agent_mask.to(self.device)
        
        # 延长仿真观察交叉口
        print("\n[EXTEND] 继续仿真观察交叉口通行...")
        for ext_step in range(100):
            try:
                traci.simulationStep()
                remaining = traci.vehicle.getIDList()
                
                # 记录仍在仿真中的直行车辆
                for vid in straight_vehicles:
                    if vid in remaining:
                        details = self._get_vehicle_details(vid)
                        
                        if vid in self.vehicle_trajectories:
                            step_info = {
                                'step': step + ext_step + 1,
                                'action': -1,
                                'action_name': 'SUMO控制',
                                'lane_index': details.get('lane_index'),
                                'grid_position': None,  # 已离开栅格
                                'speed_mps': details.get('speed_mps'),
                                'speed_kmh': details.get('speed_kmh'),
                                'position_x': details.get('position_x'),
                                'lane_id': details.get('lane_id'),
                                'lane_position': details.get('lane_position')
                            }
                            self.vehicle_trajectories[vid]['trajectory'].append(step_info)
                            
                            print(f"Step {step + ext_step + 1:3d} | {vid} | "
                                  f"车道:{details.get('lane_index', '?')} | "
                                  f"道路:{details.get('lane_id', '?'):20s} | "
                                  f"速度:{details.get('speed_kmh', 0):.1f}km/h | "
                                  f"[交叉口/出口]")
                
                if not remaining:
                    print("[EXTEND] 所有车辆已离开仿真")
                    break
            except Exception as e:
                print(f"[EXTEND] 仿真结束: {e}")
                break
        
        return self.vehicle_trajectories
    
    def save_trajectories(self, output_path: str = None):
        """保存轨迹数据到文件"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.experiment_dir / f'straight_vehicle_trajectories_{timestamp}.json'
        
        output_path = Path(output_path)
        
        # 准备保存数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_dir': str(self.experiment_dir),
            'model_name': self.model_name,
            'vehicles': self.vehicle_trajectories,
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] 轨迹数据已保存: {output_path}")
        
        # 同时保存可读的文本报告
        txt_path = output_path.with_suffix('.txt')
        self._save_readable_report(txt_path)
        print(f"[SAVED] 可读报告已保存: {txt_path}")
        
        return output_path
    
    def _generate_summary(self):
        """生成轨迹摘要"""
        summary = {}
        
        for vid, data in self.vehicle_trajectories.items():
            traj = data['trajectory']
            if not traj:
                continue
            
            # 分析换道行为
            lane_changes = []
            prev_lane = None
            for step_info in traj:
                curr_lane = step_info.get('lane_index')
                if prev_lane is not None and curr_lane is not None and prev_lane != curr_lane:
                    lane_changes.append({
                        'step': step_info['step'],
                        'from_lane': prev_lane,
                        'to_lane': curr_lane,
                        'action': step_info['action_name']
                    })
                prev_lane = curr_lane
            
            # 统计动作分布
            action_counts = {}
            for step_info in traj:
                action = step_info.get('action_name', 'Unknown')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            summary[vid] = {
                'total_steps': len(traj),
                'target_lane': data.get('target_info', {}).get('optimal_target_lane'),
                'initial_lane': traj[0].get('lane_index') if traj else None,
                'final_lane': traj[-1].get('lane_index') if traj else None,
                'lane_changes': lane_changes,
                'action_distribution': action_counts,
                'avg_speed_kmh': np.mean([s.get('speed_kmh', 0) for s in traj if s.get('speed_kmh') is not None])
            }
        
        return summary
    
    def _save_readable_report(self, output_path):
        """保存可读的文本报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("直行车辆详细轨迹分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            summary = self._generate_summary()
            
            for vid, data in self.vehicle_trajectories.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"车辆: {vid}\n")
                f.write(f"{'='*60}\n")
                
                target_info = data.get('target_info', {})
                f.write(f"转向意图: {target_info.get('intent', 'Unknown')}\n")
                f.write(f"目标车道: {target_info.get('optimal_target_lane', 'Unknown')}\n")
                f.write(f"可选目标车道: {target_info.get('target_lanes', [])}\n")
                f.write(f"目的地: {data.get('destination', 'Unknown')}\n\n")
                
                if vid in summary:
                    s = summary[vid]
                    f.write(f"--- 摘要 ---\n")
                    f.write(f"初始车道: {s.get('initial_lane')}\n")
                    f.write(f"最终车道: {s.get('final_lane')}\n")
                    f.write(f"总步数: {s.get('total_steps')}\n")
                    f.write(f"平均速度: {s.get('avg_speed_kmh', 0):.1f} km/h\n")
                    f.write(f"换道次数: {len(s.get('lane_changes', []))}\n")
                    
                    if s.get('lane_changes'):
                        f.write(f"\n换道记录:\n")
                        for lc in s['lane_changes']:
                            f.write(f"  Step {lc['step']}: 车道 {lc['from_lane']} → {lc['to_lane']} ({lc['action']})\n")
                    
                    f.write(f"\n动作分布:\n")
                    for action, count in s.get('action_distribution', {}).items():
                        f.write(f"  {action}: {count} 次\n")
                
                f.write(f"\n--- 完整轨迹 ---\n")
                f.write(f"{'Step':>5} | {'车道':>4} | {'栅格位置':>12} | {'速度(km/h)':>10} | {'动作'}\n")
                f.write("-" * 60 + "\n")
                
                for step_info in data['trajectory']:
                    grid_pos = step_info.get('grid_position')
                    if grid_pos:
                        grid_str = f"({grid_pos['lane_index']}, {grid_pos['grid_sequence']})"
                    else:
                        grid_str = "离开栅格"
                    
                    f.write(f"{step_info['step']:5d} | "
                           f"{step_info.get('lane_index', '?'):>4} | "
                           f"{grid_str:>12} | "
                           f"{step_info.get('speed_kmh', 0):>10.1f} | "
                           f"{step_info.get('action_name', 'Unknown')}\n")
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'env'):
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description='详细评估 - 记录直行车辆轨迹')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='实验目录路径')
    parser.add_argument('--model', type=str, default='final_model.pt',
                        help='模型文件名')
    args = parser.parse_args()
    
    evaluator = DetailedEvaluator(args.experiment_dir, args.model)
    
    try:
        trajectories = evaluator.run_detailed_episode()
        evaluator.save_trajectories()
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
