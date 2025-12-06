"""
IPPO 模型可视化脚本 - 修复版
用于在 SUMO GUI 中观察训练好的模型效果

关键修复：
1. 增加 SUMO 连接稳定性
2. 使用与训练一致的奖励计算
3. 添加详细的调试信息
"""

import os
import sys
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import numpy as np
import torch
import traci

from ippo.ippo_model import IPPOModel
from common.qmix_env import QmixEnvironment


class RunningMeanStd:
    """观测标准化器（仅用于推理）"""
    def __init__(self, mean, var, epsilon=1e-8):
        self.mean = np.array(mean)
        self.var = np.array(var)
        self.epsilon = epsilon

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


def load_model_and_rms(experiment_dir: str, model_name: str = "final_model.pt", device: str = "cpu"):
    """加载模型和RMS统计量"""
    exp_path = Path(experiment_dir)
    
    # 加载配置
    with open(exp_path / "config.json", 'r') as f:
        config = json.load(f)
    
    args = config.get('args', {})
    network_config = config.get('config', {}).get('network', {})
    
    # 使用配置中的正确参数
    n_agents = args.get('n_agents', 10)
    obs_dim = network_config.get('obs_dim', 22)  # 关键：使用22维
    action_dim = network_config.get('action_dim', 4)
    hidden_dim = args.get('hidden_dim', 256)  # 关键：使用256
    
    print(f"[CONFIG] n_agents={n_agents}, obs_dim={obs_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
    
    # 创建模型
    model = IPPOModel(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        shared_params=args.get('shared_params', True),
        device=device
    )
    
    # 加载权重
    model_path = exp_path / "models" / model_name
    model.load(str(model_path))
    model.eval()
    print(f"[MODEL] 已加载: {model_path}")
    
    # 加载RMS
    model_stem = Path(model_name).stem
    rms_path = exp_path / "models" / f"rms_{model_stem}.json"
    with open(rms_path, 'r') as f:
        rms_data = json.load(f)
    
    obs_rms = RunningMeanStd(rms_data['mean'], rms_data['var'])
    print(f"[RMS] 已加载: {rms_path}")
    print(f"      Mean shape: {obs_rms.mean.shape}, Var shape: {obs_rms.var.shape}")
    
    return model, obs_rms, config


def find_latest_experiment():
    """自动找到最新的实验目录"""
    script_dir = Path(__file__).resolve().parent
    ippo_experiments_dir = script_dir / 'ippo_experiments'
    
    if not ippo_experiments_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {ippo_experiments_dir}")
    
    # 找到所有实验目录
    experiment_dirs = sorted([d for d in ippo_experiments_dir.iterdir() if d.is_dir()])
    
    if not experiment_dirs:
        raise FileNotFoundError(f"没有找到实验目录: {ippo_experiments_dir}")
    
    latest_dir = experiment_dirs[-1]
    print(f"📂 使用最新实验: {latest_dir.name}")
    
    return str(latest_dir)


def run_visualization(experiment_dir: str, num_episodes: int = 3, step_delay: float = 0.05, model_name: str = "final_model.pt"):
    """
    在 SUMO GUI 中可视化运行模型
    
    Args:
        experiment_dir: 实验目录
        num_episodes: 运行的episode数量
        step_delay: 每步之间的延迟（秒），用于观察
        model_name: 模型文件名
    """
    device = 'cpu'
    
    # 加载模型
    model, obs_rms, config = load_model_and_rms(experiment_dir, model_name, device)
    
    # 获取SUMO配置路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    sumo_dir = project_root / 'sumo'
    cfg_file = str((sumo_dir / 'car.sumocfg').resolve())
    net_file = str((sumo_dir / 'net.net.xml').resolve())
    
    print(f"\n[SUMO] 配置文件: {cfg_file}")
    
    # 创建环境（GUI模式）
    env_config = {
        'net_file': net_file,
        'cfg_file': cfg_file,
        'max_agents': model.n_agents,
        'max_steps': 500,
        'gui_mode': True,
        'verbose': True  # 开启详细输出
    }
    
    env = QmixEnvironment(config=env_config)
    
    print("\n" + "="*60)
    print("开始可视化评估")
    print("="*60)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print("="*60)
        
        try:
            # 重置环境
            obs_tuple = env.reset()
            if obs_tuple[0] is None:
                print("[ERROR] 环境重置失败，跳过此episode")
                continue
                
            raw_obs_tensor, global_state, agent_mask = obs_tuple
            
            # 标准化观测
            obs_np = raw_obs_tensor.cpu().numpy()
            norm_obs_np = obs_rms.normalize(obs_np)
            obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=device)
            agent_mask = agent_mask.to(device)
            
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # 创建动作掩码
                action_mask = torch.ones(model.n_agents, model.action_dim, device=device)
                
                # 选择动作（确定性策略）
                with torch.no_grad():
                    actions, log_probs, values = model.select_actions(
                        obs_tensor, agent_mask, action_mask, deterministic=True
                    )
                
                # 打印动作信息（每10步）
                if episode_length % 10 == 0:
                    active_count = int(agent_mask.sum().item())
                    active_actions = actions[:active_count].cpu().numpy().tolist()
                    print(f"  Step {episode_length}: {active_count} agents, actions={active_actions[:5]}...")
                
                # 执行动作
                actions_list = actions.cpu().numpy().tolist()
                
                try:
                    step_result = env.step(actions_list)
                    next_raw_obs, next_global_state, next_agent_mask, reward, done, info = step_result
                except Exception as e:
                    print(f"  [ERROR] Step执行失败: {e}")
                    done = True
                    break
                
                if next_raw_obs is None:
                    print("  [ERROR] 收到空观测，结束episode")
                    done = True
                    break
                
                total_reward += reward if isinstance(reward, (int, float)) else sum(reward)
                episode_length += 1
                
                if not done:
                    # 标准化下一步观测
                    next_obs_np = next_raw_obs.cpu().numpy()
                    next_norm_obs_np = obs_rms.normalize(next_obs_np)
                    obs_tensor = torch.tensor(next_norm_obs_np, dtype=torch.float32, device=device)
                    agent_mask = next_agent_mask.to(device)
                
                # 添加延迟以便观察
                if step_delay > 0:
                    time.sleep(step_delay)
            
            # 打印episode统计
            total_exited = info.get('total_exited_vehicles', 0)
            vehicles_in_target = info.get('vehicles_in_target_lane', 0)
            target_lane_rate = vehicles_in_target / total_exited if total_exited > 0 else 0.0
            
            success = total_exited >= model.n_agents and vehicles_in_target >= model.n_agents
            
            print(f"\n[RESULT] Episode {episode + 1}:")
            print(f"  长度: {episode_length} 步")
            print(f"  奖励: {total_reward:.2f}")
            print(f"  离开车辆: {total_exited}/{model.n_agents}")
            print(f"  目标车道: {vehicles_in_target}/{total_exited} ({target_lane_rate:.1%})")
            print(f"  成功: {'✅' if success else '❌'}")
            print(f"  换道次数: {info.get('total_lane_changes', 0)}")
            print(f"  危险接近: {info.get('dangerous_approaches', 0)}")
            
            # Episode结束后继续仿真一段时间
            print(f"\n  [EXTEND] 继续仿真观察...")
            for extend_step in range(50):
                try:
                    traci.simulationStep()
                    remaining = traci.vehicle.getIDList()
                    if not remaining:
                        print(f"  [EXTEND] 所有车辆已离开 (step {extend_step})")
                        break
                    if step_delay > 0:
                        time.sleep(step_delay)
                except:
                    break
            
            # 等待用户继续
            input("\n按 Enter 继续下一个Episode...")
            
        except Exception as e:
            print(f"[ERROR] Episode {episode + 1} 出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 关闭环境
    env.close()
    print("\n[DONE] 可视化完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IPPO 模型可视化')
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        default=None,
                        help='实验目录路径（默认为最新实验）')
    parser.add_argument('--episodes', type=int, default=3, help='Episode数量')
    parser.add_argument('--delay', type=float, default=0.03, help='每步延迟(秒)')
    parser.add_argument('--model', type=str, default='final_model.pt', help='模型文件名')
    
    args = parser.parse_args()
    
    # 如果没有指定实验目录，自动找最新的
    if args.experiment_dir is None:
        experiment_dir = find_latest_experiment()
    else:
        experiment_dir = args.experiment_dir
    
    run_visualization(experiment_dir, args.episodes, args.delay, args.model)
