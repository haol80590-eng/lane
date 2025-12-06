"""
IPPO 模型可视化演示脚本
在 SUMO GUI 中展示训练好的模型效果
"""

import os
import sys

# 修复 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保能找到模块 - 必须在项目导入之前
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import time  # noqa: E402
import torch  # noqa: E402

from ippo.ippo_model import IPPOModel  # noqa: E402
from common.qmix_env import QmixEnvironment  # noqa: E402


def run_demo(model_path: str, num_episodes: int = 5, delay: float = 0.1):
    """
    运行可视化演示
    
    Args:
        model_path: 模型文件路径
        num_episodes: 演示的 episode 数
        delay: 每步延迟（秒），方便观察
    """
    device = torch.device('cpu')
    
    # 环境配置 - 开启 GUI
    env_config = {
        'net_file': r'D:\lxpaper\sumo\net.net.xml',
        'cfg_file': r'D:\lxpaper\sumo\car.sumocfg',
        'max_agents': 10,
        'max_steps': 500,
        'gui_mode': True  # 开启 GUI
    }
    
    print("=" * 60)
    print("🚗 IPPO 模型可视化演示")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"演示 Episodes: {num_episodes}")
    print(f"每步延迟: {delay}s")
    print("=" * 60)
    
    # 创建环境
    print("\n📦 初始化环境（SUMO GUI）...")
    env = QmixEnvironment(config=env_config)
    
    # 创建模型
    print("🧠 加载 IPPO 模型...")
    model = IPPOModel(
        n_agents=10,
        obs_dim=22,  # 更新：20→22（新增TTC安全特征）
        action_dim=4,
        hidden_dim=128,
        shared_params=True,
        device=device
    )
    
    # 加载权重
    model.load(model_path)
    model.eval()
    print("✅ 模型加载成功！")
    
    # 动作名称映射
    action_names = ['前进', '左换道', '右换道', '跟车']
    
    try:
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"🎬 Episode {episode + 1}/{num_episodes}")
            print("=" * 60)
            
            # 重置环境
            obs_tuple = env.reset()
            obs_tensor, global_state, agent_mask = obs_tuple
            obs_tensor = obs_tensor.to(device)
            agent_mask = agent_mask.to(device)
            
            episode_reward = 0
            step = 0
            done = False
            
            while not done:
                # 创建动作掩码
                action_mask = torch.ones(model.n_agents, model.action_dim, device=device)
                
                # 选择动作（确定性）
                with torch.no_grad():
                    actions, _, _ = model.select_actions(
                        obs_tensor, agent_mask, action_mask, deterministic=True
                    )
                
                # 执行动作
                actions_list = actions.cpu().numpy().tolist()
                step_result = env.step(actions_list)
                next_obs, next_global_state, next_agent_mask, reward, done, info = step_result
                
                # 统计
                active_count = int(agent_mask.sum().item())
                step += 1
                sim_time = step * 0.3  # 仿真时间（秒）
                
                # 每 20 步打印一次状态
                if step % 20 == 0:
                    target_rate = info.get('target_lane_rate', 0)
                    print(f"  Step {step:3d} | 时间 {sim_time:5.1f}s | 活跃车辆: {active_count} | "
                          f"目标车道率: {target_rate:.1%}")
                
                # 检查是否快结束
                if active_count == 0:
                    print(f"  ⚠️ Step {step}: 栅格区域没有车辆了，episode 即将结束")
                
                # 更新状态
                obs_tensor = next_obs.to(device)
                agent_mask = next_agent_mask.to(device)
                
                # 延迟以便观察
                if delay > 0:
                    time.sleep(delay)
            
            # Episode 结束
            target_rate = info.get('target_lane_rate', 0)
            success = "✅ 成功" if target_rate >= 0.5 else "❌ 失败"
            print(f"\n📊 Episode {episode + 1} 结果:")
            print(f"   总步数: {step}")
            print(f"   目标车道完成率: {target_rate:.1%}")
            print(f"   结果: {success}")
            
            # Episode 之间暂停
            if episode < num_episodes - 1:
                input("\n按 Enter 继续下一个 Episode...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被中断")
    
    finally:
        env.close()
        print("\n✅ 演示结束！")


def main():
    parser = argparse.ArgumentParser(description='IPPO 模型可视化演示')
    parser.add_argument('--model', type=str, 
                        default=r'D:\lxpaper\DRL\ippo_experiments\ippo_lane_change_20251128_163749\models\final_model.pt',
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=3,
                        help='演示的 episode 数')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='每步延迟（秒）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    run_demo(args.model, args.episodes, args.delay)


if __name__ == "__main__":
    main()
