"""
批量评估多个模型检查点

使用方法:
    python batch_evaluate.py --experiment_dir ippo_experiments/ippo_lane_change_20251201_014649 --episodes 50
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ippo.ippo_model import IPPOModel
from common.qmix_env import QmixEnvironment


class RunningMeanStd:
    """观测标准化器"""

    def __init__(self, mean, var, epsilon=1e-8):
        self.mean = np.array(mean)
        self.var = np.array(var)
        self.epsilon = epsilon

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


def evaluate_model(env, model, obs_rms, device, num_episodes=50, deterministic=True):
    """评估单个模型"""
    stats = {
        'success': [],
        'total_reward': [],
        'episode_length': [],
        'target_lane_rate': [],
    }

    for _ in range(num_episodes):
        obs_tuple = env.reset()
        raw_obs_tensor, global_state, agent_mask = obs_tuple

        obs_np = raw_obs_tensor.cpu().numpy()
        norm_obs_np = obs_rms.normalize(obs_np)
        obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=device)
        agent_mask = agent_mask.to(device)

        total_reward = 0
        episode_length = 0
        done = False

        while not done:
            action_mask = torch.ones(model.n_agents, model.action_dim, device=device)

            with torch.no_grad():
                actions, _, _ = model.select_actions(
                    obs_tensor, agent_mask, action_mask, deterministic=deterministic
                )

            step_result = env.step(actions.cpu().numpy().tolist())
            next_raw_obs, _, next_agent_mask, reward, done, info = step_result

            total_reward += reward if isinstance(reward, (int, float)) else sum(reward)
            episode_length += 1

            if not done:
                next_obs_np = next_raw_obs.cpu().numpy()
                next_norm_obs_np = obs_rms.normalize(next_obs_np)
                obs_tensor = torch.tensor(next_norm_obs_np, dtype=torch.float32, device=device)
                agent_mask = next_agent_mask.to(device)

        total_exited = info.get('total_exited_vehicles', 0)
        vehicles_in_target = info.get('vehicles_in_target_lane', 0)
        required_agents = info.get('required_agents', model.n_agents)

        success = 1 if (total_exited >= required_agents and
                        vehicles_in_target >= required_agents) else 0
        target_lane_rate = vehicles_in_target / total_exited if total_exited > 0 else 0.0

        stats['success'].append(success)
        stats['total_reward'].append(total_reward)
        stats['episode_length'].append(episode_length)
        stats['target_lane_rate'].append(target_lane_rate)

    return {
        'success_rate': np.mean(stats['success']),
        'avg_reward': np.mean(stats['total_reward']),
        'std_reward': np.std(stats['total_reward']),
        'avg_length': np.mean(stats['episode_length']),
        'avg_target_rate': np.mean(stats['target_lane_rate']),
    }


def main():
    parser = argparse.ArgumentParser(description='批量评估IPPO模型检查点')
    parser.add_argument('--experiment_dir', type=str,
                        default='ippo_experiments/ippo_lane_change_20251201_014649',
                        help='实验目录路径')
    parser.add_argument('--episodes', type=int, default=50, help='每个检查点评估次数')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'

    # 加载配置
    with open(experiment_dir / "config.json", 'r') as f:
        config = json.load(f)

    args_config = config.get('args', {})
    network_config = config.get('config', {}).get('network', {})

    # 创建环境
    env_config = {
        'net_file': r'C:\Users\user\Desktop\lxpaper\sumo\net.net.xml',
        'cfg_file': r'C:\Users\user\Desktop\lxpaper\sumo\car.sumocfg',
        #'net_file': r'D:\lxpaper\sumo\net.net.xml',
        #'cfg_file': r'D:\lxpaper\sumo\car.sumocfg',
        'max_agents': args_config.get('n_agents', 10),
        'max_steps': 500,
        'gui_mode': False
    }
    env = QmixEnvironment(config=env_config)

    # 获取所有模型检查点
    models_dir = experiment_dir / "models"
    checkpoints = sorted([f for f in os.listdir(models_dir)
                          if f.endswith('.pt') and not f.startswith('rms_')])

    print("=" * 70)
    print("IPPO 批量模型评估")
    print("=" * 70)
    print(f"实验目录: {experiment_dir}")
    print(f"检查点数量: {len(checkpoints)}")
    print(f"每个检查点评估: {args.episodes} episodes")
    print(f"设备: {device}")
    print("=" * 70)

    results = []

    for checkpoint in checkpoints:
        model_stem = Path(checkpoint).stem
        rms_file = f"rms_{model_stem}.json"
        rms_path = models_dir / rms_file

        if not rms_path.exists():
            print(f"[SKIP] {checkpoint} - 缺少RMS文件")
            continue

        # 创建模型
        model = IPPOModel(
            n_agents=args_config.get('n_agents', 10),
            obs_dim=network_config.get('obs_dim', 20),
            action_dim=network_config.get('action_dim', 4),
            hidden_dim=args_config.get('hidden_dim', 128),
            shared_params=args_config.get('shared_params', True),
            device=device
        )

        # 加载权重
        model.load(str(models_dir / checkpoint))
        model.eval()

        # 加载RMS
        with open(rms_path, 'r') as f:
            rms_data = json.load(f)
        obs_rms = RunningMeanStd(mean=rms_data['mean'], var=rms_data['var'])

        # 评估
        print(f"\n[EVAL] 评估 {checkpoint}...")
        metrics = evaluate_model(env, model, obs_rms, device, args.episodes)

        results.append({
            'checkpoint': checkpoint,
            **metrics
        })

        print(f"  成功率: {metrics['success_rate']:.1%} | "
              f"平均奖励: {metrics['avg_reward']:.1f} ± {metrics['std_reward']:.1f} | "
              f"目标车道率: {metrics['avg_target_rate']:.1%}")

    env.close()

    # 打印汇总
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"{'检查点':<25} {'成功率':>10} {'平均奖励':>12} {'目标车道率':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['checkpoint']:<25} {r['success_rate']:>10.1%} "
              f"{r['avg_reward']:>12.1f} {r['avg_target_rate']:>12.1%}")

    # 找最佳模型
    if results:
        best = max(results, key=lambda x: x['success_rate'])
        print("\n" + "=" * 70)
        print(f"[BEST] 最佳模型: {best['checkpoint']}")
        print(f"       成功率: {best['success_rate']:.1%}")
        print(f"       平均奖励: {best['avg_reward']:.1f}")
        print("=" * 70)

    # 保存结果
    eval_dir = experiment_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = eval_dir / f"batch_eval_{timestamp}.json"

    with open(result_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'episodes_per_checkpoint': args.episodes,
            'results': results
        }, f, indent=2)

    print(f"\n[SAVE] 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
