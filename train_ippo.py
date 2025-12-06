"""
IPPO 训练入口脚本

使用方法:
    python train_ippo.py
    python train_ippo.py --episodes 10000 --rollout_steps 512
"""

import os
import sys

# 修复 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保能找到模块 - 必须在项目导入之前
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

from ippo.ippo_model import IPPOModel  # noqa: E402
from ippo.ippo_trainer import IPPOTrainer  # noqa: E402
from common.qmix_env import QmixEnvironment  # noqa: E402
from common.config_manager import ConfigManager  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='IPPO 训练脚本')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=10000,
                        help='总训练 episode 数（RTX 3090 推荐: 10000）')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                        help='每次 rollout 的步数（RTX 3090 推荐: 2048）')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='评估间隔（episode）')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='保存间隔（episode）')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志间隔（episode）')
    
    # PPO 超参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clip 参数')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='熵正则化系数')
    parser.add_argument('--value_coef', type=float, default=1.0,
                        help='价值损失系数')
    parser.add_argument('--ppo_epochs', type=int, default=15,
                        help='PPO 更新轮数（RTX 3090 推荐: 15）')
    parser.add_argument('--mini_batch_size', type=int, default=512,
                        help='Mini-batch 大小（RTX 3090 推荐: 512）')
    
    # 网络参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度（RTX 3090 推荐: 256）')
    parser.add_argument('--shared_params', action='store_true', default=True,
                        help='是否共享网络参数')
    
    # 环境参数
    parser.add_argument('--n_agents', type=int, default=10,
                        help='智能体数量')
    parser.add_argument('--sumo_gui', action='store_true', default=False,
                        help='是否使用 SUMO GUI')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("🚗 IPPO 车辆换道训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"智能体数量: {args.n_agents}")
    print(f"总 Episodes: {args.episodes}")
    print(f"学习率: {args.lr}")
    print(f"共享参数: {args.shared_params}")
    print("=" * 60)
    
    # 加载配置
    config = ConfigManager.generate_default_config()
    config['environment']['sumo_gui'] = args.sumo_gui
    config['network']['n_agents'] = args.n_agents
    
    # 创建环境配置（自动检测项目路径，使用绝对路径）
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent  # ippo 目录
    project_root = script_dir.parent.parent  # lxpaper 目录
    sumo_dir = project_root / 'sumo'
    
    # 确保使用绝对路径
    cfg_file = (sumo_dir / 'car.sumocfg').resolve()
    net_file = (sumo_dir / 'net.net.xml').resolve()
    
    print(f"📂 SUMO 配置: {cfg_file}")
    
    if not cfg_file.exists():
        raise FileNotFoundError(f"SUMO 配置文件不存在: {cfg_file}")
    
    env_config = {
        'net_file': str(net_file),
        'cfg_file': str(cfg_file),
        'max_agents': args.n_agents,
        'max_steps': config['environment'].get('max_episode_length', 500),
        'gui_mode': args.sumo_gui
    }
    
    # 创建环境
    print("\n📦 初始化环境...")
    env = QmixEnvironment(config=env_config)
    
    # 创建模型
    print("🧠 创建 IPPO 模型...")
    model = IPPOModel(
        n_agents=args.n_agents,
        obs_dim=config['network']['obs_dim'],
        action_dim=config['network']['action_dim'],
        hidden_dim=args.hidden_dim,
        shared_params=args.shared_params,
        device=device
    )
    
    # 创建训练器
    print("🏋️ 创建训练器...")
    trainer = IPPOTrainer(
        env=env,
        model=model,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=0.5,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        device=device
    )
    
    # 设置实验目录
    trainer.setup_experiment_dir()
    
    # 保存配置
    config_path = trainer.experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'config': config
        }, f, indent=2, default=str)
    print(f"📝 配置已保存: {config_path}")
    
    # 开始训练
    try:
        trainer.train(
            total_episodes=args.episodes,
            rollout_steps=args.rollout_steps,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_interval=args.log_interval
        )
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断，保存当前状态...")
        trainer.save_checkpoint()
        trainer.save_training_stats()
    finally:
        env.close()
    
    print("\n✅ 训练完成！")
    print(f"📂 结果保存在: {trainer.experiment_dir}")


if __name__ == "__main__":
    main()
