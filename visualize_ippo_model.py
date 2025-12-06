"""
可视化训练好的IPPO模型
在SUMO-GUI中运行，观察智能体的实际行为
"""
import sys
import io
import os
import json
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from common.qmix_env import QmixEnvironment
from ippo.ippo_model import IPPOModel

# 修复Windows控制台编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_trained_model(model_path, config):
    """加载训练好的IPPO模型"""
    print(f"正在加载模型: {model_path}")
    
    # 创建模型
    model = IPPOModel(
        n_agents=config.get('n_agents', 10),
        obs_dim=config.get('obs_dim', 20),
        action_dim=config.get('action_dim', 4),
        hidden_dim=config.get('hidden_dim', 128),
        shared_params=config.get('shared_params', True),
        device='cpu'
    )
    
    # 加载权重
    model.load(model_path)
    model.eval()  # 设置为评估模式
    
    print("✓ 模型加载成功")
    return model


def to_numpy(data):
    """将Tensor转换为Numpy数组"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return data


def visualize_episode(env, model, max_steps=1000, deterministic=True):
    """运行一个可视化Episode"""
    print("\n" + "="*60)
    print("开始可视化Episode（SUMO-GUI窗口已打开）")
    print("="*60)
    
    obs, state, agent_mask = env.reset()
    # 转换为numpy
    obs = to_numpy(obs)
    state = to_numpy(state)
    agent_mask = to_numpy(agent_mask)
    
    episode_reward = 0
    step = 0
    info = {}  # 初始化 info，避免未赋值错误
    
    print(f"\n初始车辆数: {int(np.sum(agent_mask))}")
    print(f"使用策略: {'确定性（greedy）' if deterministic else '随机采样'}")
    
    while step < max_steps:
        step += 1
        
        if np.sum(agent_mask) == 0:
            print(f"\n[Step {step}] 所有车辆已退出")
            break
        
        # 使用训练好的模型选择动作
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)  # [n_agents, obs_dim]
            agent_mask_tensor = torch.FloatTensor(agent_mask)  # [n_agents]
            
            # 获取动作掩码（如果环境提供）
            action_masks = None
            if hasattr(env, 'get_action_masks'):
                action_masks = torch.FloatTensor(env.get_action_masks())
            
            # 使用IPPO模型选择动作
            actions_tensor, log_probs, values = model.select_actions(
                obs_tensor,
                agent_mask_tensor,
                action_masks,
                deterministic=deterministic
            )
            
            actions = actions_tensor.cpu().numpy()
            
            # 每50步打印动作分布
            if step % 50 == 0:
                print(f"\n[Debug Step {step}] 动作选择:")
                action_names = ['Stay', 'Left', 'Right', 'Acc']
                for idx in range(len(agent_mask)):
                    if agent_mask[idx]:
                        print(f"  Agent {idx}: {action_names[actions[idx]]} (value: {values[idx].item():.2f})")
        
        # 执行动作
        next_obs, next_state, next_agent_mask, reward, done, info = env.step(actions)
        
        # 转换为numpy
        next_obs = to_numpy(next_obs)
        next_state = to_numpy(next_state)
        next_agent_mask = to_numpy(next_agent_mask)
        
        episode_reward += reward
        
        # 每50步打印一次信息
        if step % 50 == 0:
            active_count = int(np.sum(next_agent_mask))
            print(f"[Step {step:4d}] 车辆数: {active_count:2d}, 累计奖励: {episode_reward:8.2f}")
        
        obs = next_obs
        state = next_state
        agent_mask = next_agent_mask
        
        if done:
            print(f"\n[Step {step}] Episode结束（done=True）")
            break
    
    print("\n" + "="*60)
    print("Episode结束")
    print("="*60)
    print(f"总步数: {step}")
    print(f"总奖励: {episode_reward:.2f}")
    print(f"成功退出车辆: {info.get('exited_vehicles', 0)}")
    print(f"目标车道成功车辆: {info.get('target_lane_vehicles', 0)}")
    
    success_rate = 0
    if info.get('exited_vehicles', 0) > 0:
        success_rate = info.get('target_lane_vehicles', 0) / info.get('exited_vehicles', 1) * 100
    print(f"成功率: {success_rate:.1f}%")
    print("="*60 + "\n")
    
    return episode_reward, success_rate


def main():
    parser = argparse.ArgumentParser(description='IPPO模型可视化')
    parser.add_argument('experiment', type=str, nargs='?',
                        default='ippo_experiments/ippo_lane_change_20251201_214543',
                        help='实验目录路径（如 ippo_experiments/xxx 或直接 xxx）')
    parser.add_argument('--model', type=str, default='final_model.pt',
                        help='模型文件名')
    parser.add_argument('--episodes', type=int, default=1,
                        help='运行Episode数')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='启用SUMO-GUI（默认启用）')
    parser.add_argument('--extend', type=int, default=0,
                        help='额外延长步数')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略（默认使用确定性策略）')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每个Episode最大步数')
    args = parser.parse_args()
    
    # 处理 extend 参数
    args.max_steps += args.extend
    
    # 构建路径（支持多种输入格式）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理实验路径：支持 ippo_experiments/xxx 或直接 xxx
    exp_path = args.experiment.replace('\\', '/').strip('/')
    if exp_path.startswith('ippo_experiments/'):
        exp_name = exp_path.replace('ippo_experiments/', '')
    else:
        exp_name = exp_path
    
    experiment_dir = os.path.join(script_dir, 'ippo_experiments', exp_name)
    model_path = os.path.join(experiment_dir, 'models', args.model)
    config_path = os.path.join(experiment_dir, 'config.json')
    
    print("="*60)
    print("IPPO模型可视化评估")
    print("="*60)
    print(f"实验目录: {experiment_dir}")
    print(f"模型路径: {model_path}")
    print(f"配置路径: {config_path}")
    print(f"Episode数: {args.episodes}")
    print(f"策略模式: {'随机采样' if args.stochastic else '确定性（greedy）'}")
    print("="*60 + "\n")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在: {model_path}")
        print("可用的模型文件:")
        models_dir = os.path.join(experiment_dir, 'models')
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return
    
    if not os.path.exists(config_path):
        print(f"❌ 错误：配置文件不存在: {config_path}")
        return
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("✓ 配置加载成功")
    
    # 从 args 字段读取训练参数（训练时保存的格式）
    args_config = config.get('args', {})
    network_config = config.get('config', {}).get('network', {})
    
    # 构造模型配置（优先使用 args，其次使用 network config）
    model_config = {
        'n_agents': args_config.get('n_agents', network_config.get('n_agents', 10)),
        'obs_dim': network_config.get('obs_dim', 20),
        'action_dim': network_config.get('action_dim', 4),
        'hidden_dim': args_config.get('hidden_dim', network_config.get('agent_hidden_dim', 256)),
        'shared_params': args_config.get('shared_params', True)
    }
    
    print(f"  模型参数: n_agents={model_config['n_agents']}, hidden_dim={model_config['hidden_dim']}")
    
    # 加载模型
    model = load_trained_model(model_path, model_config)
    
    # SUMO配置路径（自动查找）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # lxpaper 目录
    sumo_cfg_path = os.path.join(project_root, 'sumo', 'car.sumocfg')
    sumo_net_path = os.path.join(project_root, 'sumo', 'net.net.xml')
    
    # 构造环境配置
    env_config = {
        'cfg_file': sumo_cfg_path,
        'net_file': sumo_net_path,
        'max_agents': model_config['n_agents'],
        'max_steps': args.max_steps,
        'gui_mode': True,  # 启用GUI
        'render_mode': 'human'
    }
    
    print(f"✓ SUMO配置路径: {sumo_cfg_path}")
    
    # 创建环境
    print("\n正在创建环境...")
    env = QmixEnvironment(config=env_config)
    print("✓ 环境创建成功（SUMO-GUI应已启动）\n")
    
    # 运行可视化Episode
    rewards = []
    success_rates = []
    
    for i in range(args.episodes):
        print(f"\n{'='*20} Episode {i+1}/{args.episodes} {'='*20}")
        reward, success_rate = visualize_episode(
            env, model, 
            max_steps=args.max_steps,
            deterministic=not args.stochastic
        )
        rewards.append(reward)
        success_rates.append(success_rate)
        
        if i < args.episodes - 1:
            input("\n按Enter键继续下一个Episode...")
    
    # 总结
    print("\n" + "="*60)
    print("评估完成")
    print("="*60)
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"平均成功率: {np.mean(success_rates):.1f}%")
    print(f"奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print("="*60)
    
    # 清理
    env.close()
    print("\n环境已关闭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
