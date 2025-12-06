"""
IPPO 模型评估脚本

功能：
1. 加载训练好的模型和RMS统计量
2. 在环境中运行多个episode进行评估
3. 记录成功率、奖励、碰撞等指标
4. 生成评估报告

使用方法:
    python evaluate_ippo.py --experiment_dir ippo_experiments/ippo_lane_change_20251201_014649
    python evaluate_ippo.py --experiment_dir ippo_experiments/ippo_lane_change_20251201_014649 --episodes 200 --gui
"""

import os
import sys

# 修复 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from ippo.ippo_model import IPPOModel
from common.qmix_env import QmixEnvironment
from common.config_manager import ConfigManager


class RunningMeanStd:
    """观测标准化器（仅用于推理，不更新）"""

    def __init__(self, mean, var, epsilon=1e-8):
        self.mean = np.array(mean)
        self.var = np.array(var)
        self.epsilon = epsilon

    def normalize(self, x):
        """标准化"""
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


class IPPOEvaluator:
    """IPPO 模型评估器"""

    def __init__(self, experiment_dir: str, model_name: str = "final_model.pt",
                 device: str = "auto", gui: bool = False):
        """
        初始化评估器

        Args:
            experiment_dir: 实验目录路径
            model_name: 模型文件名
            device: 计算设备
            gui: 是否使用SUMO GUI
        """
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name
        self.gui = gui

        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 加载配置
        self.config = self._load_config()

        # 创建环境
        self.env = self._create_environment()

        # 创建并加载模型
        self.model = self._load_model()

        # 加载RMS统计量
        self.obs_rms = self._load_rms()

        # 统计数据
        self.stats = defaultdict(list)

    def _load_config(self) -> dict:
        """加载实验配置"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[CONFIG] 配置已加载: {config_path}")
        return config

    def _create_environment(self) -> QmixEnvironment:
        """创建环境"""
        args = self.config.get('args', {})
        env_config = self.config.get('config', {}).get('environment', {})

        # 自动检测 SUMO 配置路径（使用绝对路径）
        script_dir = Path(__file__).resolve().parent  # ippo 目录
        project_root = script_dir.parent.parent  # lxpaper 目录
        sumo_dir = project_root / 'sumo'
        
        # 确保路径是绝对路径
        cfg_file = (sumo_dir / 'car.sumocfg').resolve()
        net_file = (sumo_dir / 'net.net.xml').resolve()
        
        print(f"[PATH] SUMO配置: {cfg_file}")
        
        if not cfg_file.exists():
            raise FileNotFoundError(f"SUMO配置文件不存在: {cfg_file}")
        
        config = {
            'net_file': str(net_file),
            'cfg_file': str(cfg_file),
            'max_agents': args.get('n_agents', 10),
            'max_steps': env_config.get('max_episode_length', 500),
            'gui_mode': self.gui
        }

        env = QmixEnvironment(config=config)
        print(f"[ENV] 环境已创建 (GUI={self.gui})")
        return env

    def _load_model(self) -> IPPOModel:
        """加载模型"""
        args = self.config.get('args', {})
        network_config = self.config.get('config', {}).get('network', {})

        model = IPPOModel(
            n_agents=args.get('n_agents', 10),
            obs_dim=network_config.get('obs_dim', 20),
            action_dim=network_config.get('action_dim', 4),
            hidden_dim=args.get('hidden_dim', 128),
            shared_params=args.get('shared_params', True),
            device=self.device
        )

        # 加载权重
        model_path = self.experiment_dir / "models" / self.model_name
        model.load(str(model_path))
        model.eval()  # 设置为评估模式

        return model

    def _load_rms(self) -> RunningMeanStd:
        """加载RMS统计量"""
        # 根据模型名称找到对应的RMS文件
        model_stem = Path(self.model_name).stem
        rms_path = self.experiment_dir / "models" / f"rms_{model_stem}.json"

        with open(rms_path, 'r') as f:
            rms_data = json.load(f)

        rms = RunningMeanStd(
            mean=rms_data['mean'],
            var=rms_data['var']
        )
        print(f"[RMS] RMS统计量已加载: {rms_path}")
        return rms

    def evaluate(self, num_episodes: int = 100, deterministic: bool = True, extend_steps: int = 0) -> dict:
        """
        运行评估

        Args:
            num_episodes: 评估的episode数量
            deterministic: 是否使用确定性策略
            extend_steps: episode结束后继续仿真的步数（GUI模式下观察交叉口通行）

        Returns:
            评估结果字典
        """
        print("\n" + "=" * 60)
        print("[START] 开始评估")
        print(f"   Episodes: {num_episodes}")
        print(f"   确定性策略: {deterministic}")
        print(f"   设备: {self.device}")
        if extend_steps > 0:
            print(f"   延长仿真: {extend_steps} 步")
        print("=" * 60 + "\n")

        for episode in range(num_episodes):
            episode_stats = self._run_episode(deterministic, extend_steps)

            # 记录统计
            for key, value in episode_stats.items():
                self.stats[key].append(value)

            # 打印进度
            if (episode + 1) % 10 == 0 or episode == 0:
                success_rate = np.mean(self.stats['success'][-min(10, episode + 1):])
                avg_reward = np.mean(self.stats['total_reward'][-min(10, episode + 1):])
                print(f"  Episode {episode + 1}/{num_episodes} | "
                      f"成功率: {success_rate:.1%} | "
                      f"平均奖励: {avg_reward:.1f}")

        # 计算最终统计
        results = self._compute_final_stats()

        # 打印报告
        self._print_report(results)

        # 保存结果
        self._save_results(results, num_episodes, deterministic)

        return results

    def _run_episode(self, deterministic: bool = True, extend_steps: int = 0) -> dict:
        """运行单个episode

        Args:
            deterministic: 是否使用确定性策略
            extend_steps: episode结束后继续仿真的步数（用于观察车辆通过交叉口）
        """
        # 重置环境
        obs_tuple = self.env.reset()
        raw_obs_tensor, global_state, agent_mask = obs_tuple

        # 标准化观测
        obs_np = raw_obs_tensor.cpu().numpy()
        norm_obs_np = self.obs_rms.normalize(obs_np)
        obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=self.device)
        agent_mask = agent_mask.to(self.device)

        total_reward = 0
        episode_length = 0
        done = False

        while not done:
            # 创建动作掩码
            action_mask = torch.ones(self.model.n_agents, self.model.action_dim,
                                     device=self.device)

            # 选择动作
            with torch.no_grad():
                actions, log_probs, values = self.model.select_actions(
                    obs_tensor, agent_mask, action_mask, deterministic=deterministic
                )

            # 执行动作
            actions_list = actions.cpu().numpy().tolist()
            step_result = self.env.step(actions_list)
            next_raw_obs, next_global_state, next_agent_mask, reward, done, info = step_result

            total_reward += reward if isinstance(reward, (int, float)) else sum(reward)
            episode_length += 1

            if not done:
                # 标准化下一步观测
                next_obs_np = next_raw_obs.cpu().numpy()
                next_norm_obs_np = self.obs_rms.normalize(next_obs_np)
                obs_tensor = torch.tensor(next_norm_obs_np, dtype=torch.float32,
                                          device=self.device)
                agent_mask = next_agent_mask.to(self.device)

        # 【新增】Episode结束后继续仿真，观察车辆通过交叉口
        if extend_steps > 0 and self.gui:
            import traci
            print(f"  [EXTEND] 继续仿真 {extend_steps} 步，观察交叉口通行...")
            for _ in range(extend_steps):
                try:
                    traci.simulationStep()
                    # 检查是否还有车辆
                    remaining = traci.vehicle.getIDList()
                    if not remaining:
                        print(f"  [EXTEND] 所有车辆已离开仿真")
                        break
                except:
                    break

        # 提取episode统计
        total_exited = info.get('total_exited_vehicles', 0)
        vehicles_in_target = info.get('vehicles_in_target_lane', 0)
        required_agents = info.get('required_agents', self.model.n_agents)

        success = 1 if (total_exited >= required_agents and
                        vehicles_in_target >= required_agents) else 0

        target_lane_rate = vehicles_in_target / total_exited if total_exited > 0 else 0.0

        return {
            'success': success,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'vehicles_exited': total_exited,
            'vehicles_in_target': vehicles_in_target,
            'target_lane_rate': target_lane_rate,
            'avg_travel_time': info.get('avg_travel_time', 0.0),
            'dangerous_approaches': info.get('dangerous_approaches', 0),
            'total_lane_changes': info.get('total_lane_changes', 0),
        }

    def _compute_final_stats(self) -> dict:
        """计算最终统计结果"""
        results = {}

        for key, values in self.stats.items():
            arr = np.array(values)
            results[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }

        # 添加成功率
        results['success_rate'] = float(np.mean(self.stats['success']))

        return results

    def _print_report(self, results: dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("[REPORT] 评估报告")
        print("=" * 60)

        print(f"\n{'指标':<25} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
        print("-" * 65)

        # 核心指标
        key_metrics = [
            ('success', '成功率'),
            ('total_reward', '总奖励'),
            ('episode_length', 'Episode长度'),
            ('target_lane_rate', '目标车道率'),
            ('avg_travel_time', '平均通过时间'),
            ('dangerous_approaches', '危险接近次数'),
            ('total_lane_changes', '换道次数'),
        ]

        for key, name in key_metrics:
            if key in results and isinstance(results[key], dict):
                stats = results[key]
                print(f"{name:<25} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
                      f"{stats['min']:>10.2f} {stats['max']:>10.2f}")

        print("\n" + "=" * 60)
        print(f"[RESULT] 最终成功率: {results['success_rate']:.1%}")
        print("=" * 60)

    def _save_results(self, results: dict, num_episodes: int, deterministic: bool):
        """保存评估结果"""
        # 创建评估结果目录
        eval_dir = self.experiment_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存结果
        eval_results = {
            'timestamp': timestamp,
            'model': self.model_name,
            'num_episodes': num_episodes,
            'deterministic': deterministic,
            'device': self.device,
            'results': results,
            'raw_stats': {k: v for k, v in self.stats.items()}
        }

        result_path = eval_dir / f"eval_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=lambda x: x.tolist()
            if isinstance(x, np.ndarray) else x)

        print(f"\n[SAVE] 评估结果已保存: {result_path}")

    def close(self):
        """关闭环境"""
        self.env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='IPPO 模型评估脚本',
        usage='python evaluate_ippo.py <experiment_dir> [options]\n'
              '示例: python evaluate_ippo.py ippo_experiments/ippo_lane_change_20251201_214543 --gui --extend 100 --episodes 5'
    )

    # 位置参数（实验目录）
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='实验目录路径（如 ippo_experiments/xxx）')
    # 兼容旧格式
    parser.add_argument('--experiment_dir', dest='experiment_dir_opt', type=str,
                        help='实验目录路径（旧格式，建议使用位置参数）')
    parser.add_argument('--model', type=str, default='final_model.pt',
                        help='模型文件名 (默认: final_model.pt)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='评估episode数量 (默认: 5)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='使用确定性策略 (默认: True)')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略')
    parser.add_argument('--gui', action='store_true',
                        help='启用SUMO GUI可视化')
    parser.add_argument('--extend', type=int, default=0,
                        help='Episode结束后继续仿真的步数 (默认: 0，GUI模式建议: 100)')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')

    args = parser.parse_args()
    
    # 处理实验目录（优先使用位置参数，其次使用 --experiment_dir）
    if args.experiment_dir is None and args.experiment_dir_opt is not None:
        args.experiment_dir = args.experiment_dir_opt
    
    if args.experiment_dir is None:
        parser.error('请指定实验目录，如: python evaluate_ippo.py ippo_experiments/xxx --gui')
    
    return args


def main():
    args = parse_args()

    # 确定是否使用确定性策略
    deterministic = not args.stochastic

    print("=" * 60)
    print("[IPPO] 模型评估")
    print("=" * 60)
    print(f"实验目录: {args.experiment_dir}")
    print(f"模型: {args.model}")
    print(f"评估Episodes: {args.episodes}")
    print(f"确定性策略: {deterministic}")
    print(f"GUI模式: {args.gui}")
    if args.extend > 0:
        print(f"延长仿真: {args.extend} 步")
    print("=" * 60)

    # 创建评估器
    evaluator = IPPOEvaluator(
        experiment_dir=args.experiment_dir,
        model_name=args.model,
        device=args.device,
        gui=args.gui
    )

    try:
        # 运行评估
        results = evaluator.evaluate(
            num_episodes=args.episodes,
            deterministic=deterministic,
            extend_steps=args.extend
        )
    finally:
        evaluator.close()

    print("\n[DONE] 评估完成！")


if __name__ == "__main__":
    main()
