"""
IPPO (Independent PPO) 训练器
使用个体奖励训练多智能体策略

核心特点：
1. 每个智能体使用独立的 Actor-Critic（可共享参数）
2. 每个智能体根据自己的个体奖励更新
3. 使用 GAE (Generalized Advantage Estimation) 计算优势函数
4. PPO-Clip 目标函数
"""

import os
import sys

# 确保能找到模块 - 必须在项目导入之前
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from .ippo_model import IPPOModel  # noqa: E402
from common.qmix_env import QmixEnvironment  # noqa: E402


class RunningMeanStd:
    """
    动态标准化层 (Running Mean and Std)
    用于将观测值标准化为均值0，方差1
    """

    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4
        self.epsilon = epsilon

    def update(self, x):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """根据batch的统计量更新全局统计量"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """标准化"""
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


class RolloutBuffer:
    """
    经验缓冲区
    存储一个 rollout 的所有数据
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []  # 个体奖励 [n_agents]
        self.values = []
        self.dones = []
        self.agent_masks = []
        self.action_masks = []

    def add(self, obs, action, log_prob, reward, value, done, agent_mask, action_mask):
        """添加一步数据"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.agent_masks.append(agent_mask)
        self.action_masks.append(action_mask)

    def clear(self):
        """清空缓冲区"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.agent_masks = []
        self.action_masks = []

    def get_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """获取批次数据"""
        return {
            'observations': torch.stack(self.observations).to(device),
            'actions': torch.stack(self.actions).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.stack(self.rewards).to(device),
            'values': torch.stack(self.values).to(device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'agent_masks': torch.stack(self.agent_masks).to(device),
            'action_masks': torch.stack(self.action_masks).to(device) if self.action_masks[0] is not None else None
        }

    def __len__(self):
        return len(self.observations)


class IPPOTrainer:
    """
    IPPO 训练器

    使用 PPO 算法独立训练每个智能体
    """

    def __init__(self,
                 env: QmixEnvironment,
                 model: IPPOModel,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 device: str = 'cpu',
                 verbose: bool = False):
                # verbose: bool = True):
        """
        初始化 IPPO 训练器

        Args:
            env: 环境
            model: IPPO 模型
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            clip_epsilon: PPO clip 参数
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            ppo_epochs: PPO 更新轮数
            mini_batch_size: mini-batch 大小
            device: 计算设备
            verbose: 是否打印详细日志（False 可提速 20-30%）
        """
        self.env = env
        self.model = model
        self.device = torch.device(device)
        self.verbose = verbose  # 控制打印输出

        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
        self.scheduler = None  # 将在 train 中初始化

        # 观测标准化
        self.obs_rms = RunningMeanStd(shape=(model.obs_dim,))

        # 经验缓冲区
        self.buffer = RolloutBuffer()

        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.training_history = defaultdict(list)

        # GAE bootstrap value 存储
        self.last_values = None

        # 实验目录
        self.experiment_dir = None

    def setup_experiment_dir(self, base_dir: str = "ippo_experiments"):
        """设置实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(base_dir) / f"ippo_lane_change_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)

        print(f"📂 实验目录: {self.experiment_dir}")
        return self.experiment_dir

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor,
                    agent_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)

        Args:
            rewards: [T, n_agents] 个体奖励
            values: [T, n_agents] 状态价值
            dones: [T] episode 结束标志
            next_value: [n_agents] 最后状态的价值
            agent_masks: [T, n_agents] 智能体掩码

        Returns:
            advantages: [T, n_agents]
            returns: [T, n_agents]
        """
        T, n_agents = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # 逐个智能体计算 GAE
        for agent_idx in range(n_agents):
            gae = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = next_value[agent_idx]
                    next_non_terminal = 1.0 - dones[t]
                else:
                    next_val = values[t + 1, agent_idx]
                    next_non_terminal = 1.0 - dones[t]

                # 只对活跃的智能体计算
                if agent_masks[t, agent_idx] == 0:
                    advantages[t, agent_idx] = 0
                    returns[t, agent_idx] = 0
                    continue

                delta = rewards[t, agent_idx] + self.gamma * next_val * next_non_terminal - values[t, agent_idx]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t, agent_idx] = gae
                returns[t, agent_idx] = gae + values[t, agent_idx]

        return advantages, returns

    def collect_rollout(self, rollout_steps: int = 128) -> Dict[str, Any]:
        """
        收集一个 rollout 的经验

        Args:
            rollout_steps: rollout 步数

        Returns:
            rollout_info: 收集信息
        """
        self.buffer.clear()
        self.model.eval()

        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        # ====== 学术评价指标（新增 2025-11-30，性能优化版）======
        episode_avg_travel_times = []  # 平均通过时间
        episode_dangerous_approaches = []  # 危险接近次数
        episode_lane_changes = []  # 换道次数
        episode_target_lane_rates = []  # 目标车道达成率

        current_episode_reward = 0
        current_episode_length = 0

        # 重置环境 - 返回 (observations, global_state, agent_mask)
        obs_tuple = self.env.reset()
        raw_obs_tensor, global_state, agent_mask = obs_tuple

        # 更新并应用 Obs Normalization
        obs_np = raw_obs_tensor.cpu().numpy()
        self.obs_rms.update(obs_np)
        norm_obs_np = self.obs_rms.normalize(obs_np)
        obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=self.device)

        # 原始观测用于奖励计算，标准化观测用于网络输入
        raw_obs_tensor = raw_obs_tensor.to(self.device)
        agent_mask = agent_mask.to(self.device)

        for step in range(rollout_steps):
            # 创建动作掩码（所有动作都有效）
            action_mask = torch.ones(self.model.n_agents, self.model.action_dim, device=self.device)

            # 🔧 关键：保存执行动作前的车辆ID列表
            # 用于正确分配离开奖励（离开的车辆在step后不再活跃）
            prev_vehicle_ids = self.env.qminx.get_active_vehicle_ids()

            # 选择动作 (使用标准化观测)
            with torch.no_grad():
                actions, log_probs, values = self.model.select_actions(
                    obs_tensor, agent_mask, action_mask, deterministic=False
                )

            # 执行动作 - 转换为列表
            actions_list = actions.cpu().numpy().tolist()

            # step 返回 (next_obs, next_global_state, next_agent_mask, reward, done, info)
            step_result = self.env.step(actions_list)
            next_raw_obs, next_global_state, next_agent_mask, reward, done, info = step_result

            # 更新并应用 Obs Normalization 对下一步观测
            next_obs_np = next_raw_obs.cpu().numpy()
            self.obs_rms.update(next_obs_np)
            next_norm_obs_np = self.obs_rms.normalize(next_obs_np)
            next_obs = torch.tensor(next_norm_obs_np, dtype=torch.float32, device=self.device)
            next_raw_obs = next_raw_obs.to(self.device)

            # 计算个体奖励（使用执行动作前的车辆列表，确保离开奖励正确分配）
            # 修复（2025-11-29）：传入actions以计算动作意图奖励
            individual_rewards = self._compute_individual_rewards(
                info, raw_obs_tensor, agent_mask, prev_vehicle_ids, actions_list
            )
            reward_tensor = torch.tensor(individual_rewards, dtype=torch.float32, device=self.device)

            # 存储经验 (存储标准化后的观测)
            self.buffer.add(
                obs_tensor, actions, log_probs, reward_tensor, values,
                done, agent_mask, action_mask
            )

            # 更新统计
            current_episode_reward += sum(individual_rewards)
            current_episode_length += 1
            self.total_steps += 1

            # 检查 episode 结束
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)

                # 成功判定：要求 10 辆车(或配置的智能体数量)全部在目标车道驶离栅格区域
                total_exited = info.get('total_exited_vehicles', 0)
                vehicles_in_target = info.get('vehicles_in_target_lane', 0)
                required_agents = info.get('required_agents', self.model.n_agents)

                success_all = 1 if (total_exited >= required_agents and
                                    vehicles_in_target >= required_agents) else 0
                episode_successes.append(success_all)

                # ====== 收集学术评价指标 ======
                episode_avg_travel_times.append(info.get('avg_travel_time', 0.0))
                episode_dangerous_approaches.append(info.get('dangerous_approaches', 0))
                episode_lane_changes.append(info.get('total_lane_changes', 0))
                target_lane_rate = vehicles_in_target / total_exited if total_exited > 0 else 0.0
                episode_target_lane_rates.append(target_lane_rate)

                if self.verbose:
                    print(
                        f"   📊 Episode结束统计: 在目标车道离开 {vehicles_in_target}/{total_exited} "
                        f"({target_lane_rate:.1%}), 通过时间 {info.get('avg_travel_time', 0):.1f}步, "
                        f"危险接近 {info.get('dangerous_approaches', 0)} 次"
                    )

                # 重置
                current_episode_reward = 0
                current_episode_length = 0
                self.episode_count += 1

                obs_tuple = self.env.reset()
                raw_obs_tensor, global_state, agent_mask = obs_tuple

                # 重置后的标准化处理
                obs_np = raw_obs_tensor.cpu().numpy()
                # 注意：重置时不一定需要 update，但为了更准可以 update
                # self.obs_rms.update(obs_np)
                norm_obs_np = self.obs_rms.normalize(obs_np)
                obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=self.device)

                raw_obs_tensor = raw_obs_tensor.to(self.device)
                agent_mask = agent_mask.to(self.device)
            else:
                obs_tensor = next_obs
                raw_obs_tensor = next_raw_obs
                agent_mask = next_agent_mask.to(self.device)

        # 计算最后状态的价值（用于 GAE）
        action_mask = torch.ones(self.model.n_agents, self.model.action_dim, device=self.device)
        with torch.no_grad():
            _, _, last_values = self.model.select_actions(
                obs_tensor, agent_mask, action_mask, deterministic=True
            )

        # 存储 last_values 用于 GAE bootstrap
        self.last_values = last_values

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'last_values': last_values,
            'steps': len(self.buffer),
            # ====== 学术评价指标（性能优化版）======
            'avg_travel_times': episode_avg_travel_times,
            'dangerous_approaches': episode_dangerous_approaches,
            'lane_changes': episode_lane_changes,
            'target_lane_rates': episode_target_lane_rates,
        }

    def _process_observations(self, obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """处理观测字典，转换为张量"""
        n_agents = self.model.n_agents
        obs_dim = self.model.obs_dim
        action_dim = self.model.action_dim

        obs_tensor = torch.zeros(n_agents, obs_dim, device=self.device)
        agent_mask = torch.zeros(n_agents, device=self.device)
        action_mask = torch.ones(n_agents, action_dim, device=self.device)

        observations = obs_dict.get('observations', {})
        action_masks = obs_dict.get('action_masks', {})

        for i, (vid, obs) in enumerate(observations.items()):
            if i >= n_agents:
                break
            obs_tensor[i] = torch.tensor(obs, dtype=torch.float32, device=self.device)
            agent_mask[i] = 1.0

            if vid in action_masks:
                action_mask[i] = torch.tensor(action_masks[vid], dtype=torch.float32, device=self.device)

        return obs_tensor, agent_mask, action_mask

    def _tensor_to_action_dict(self, actions: torch.Tensor, obs_dict: Dict) -> Dict[str, int]:
        """将动作张量转换为动作字典"""
        action_dict = {}
        observations = obs_dict.get('observations', {})

        for i, vid in enumerate(observations.keys()):
            if i >= self.model.n_agents:
                break
            action_dict[vid] = actions[i].item()

        return action_dict

    def _compute_individual_rewards(self, info: Dict, obs_tensor: torch.Tensor,
                                    agent_mask: torch.Tensor,
                                    prev_vehicle_ids: List[str] = None,
                                    actions: List[int] = None) -> List[float]:
        """
        计算个体奖励 - 使用 reward.py 中的 IPPO 奖励函数

        【2025-12-01 最终版】两组件极简设计

        奖励函数：r = r_lane + r_collision

        1. r_lane: 车道势能（紧迫度调制）
           - 早期(p<0.4): u≈0.01，几乎无换道动机
           - 后期(p>0.6): u≈5.0，强烈换道动机

        2. r_collision: 减速度惩罚
           - 紧急刹车时给予惩罚
           - Agent 从真实经验学习安全驾驶

        关键修复：使用 prev_vehicle_ids 处理离开奖励的时序问题

        Args:
            info: 环境返回的信息字典
            obs_tensor: 观测张量 [n_agents, obs_dim]
            agent_mask: 智能体掩码 [n_agents]（执行动作前的掩码）
            prev_vehicle_ids: 执行动作前的车辆ID列表

        Returns:
            List[float]: 每个智能体的个体奖励
        """
        n_agents = self.model.n_agents
        rewards = [0.0] * n_agents

        try:
            # 使用执行动作前的车辆列表（如果提供）
            # 这确保了离开奖励能正确分配给对应的智能体索引
            if prev_vehicle_ids is None:
                prev_vehicle_ids = self.env.qminx.get_active_vehicle_ids()

            if not prev_vehicle_ids:
                return rewards

            # 调用 reward.py 中的 IPPO 奖励函数
            # 极简设计：r_lane + r_collision 两个组件
            reward_dict = self.env.reward_calculator.calculate_individual_rewards_for_ippo(
                prev_vehicle_ids,
                include_exited=True,
                actions=actions  # 保留参数但不再使用（向后兼容）
            )

            # 将车辆奖励映射到智能体索引
            for i, vid in enumerate(prev_vehicle_ids):
                if i < n_agents and agent_mask[i].item() > 0:
                    rewards[i] = reward_dict.get(vid, 0.0)  # 默认0，不人为打分

        except Exception as e:
            print(f"⚠️ 计算个体奖励时出错: {e}")
            # 出错时返回0，不人为打分
            for i in range(n_agents):
                if agent_mask[i].item() > 0:
                    rewards[i] = 0.0

        return rewards

    def update(self) -> Dict[str, float]:
        """
        执行 PPO 更新

        Returns:
            metrics: 更新指标
        """
        self.model.train()

        # 获取数据
        batch = self.buffer.get_batch(self.device)

        # 计算 GAE - 使用正确的 bootstrap value
        # 如果 rollout 最后一步不是 episode 结束，使用 last_values；否则使用 0
        if self.last_values is not None:
            bootstrap_value = self.last_values
        else:
            # 回退到原来的方法（不应该发生，但保险起见）
            bootstrap_value = batch['values'][-1]

        advantages, returns = self.compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            bootstrap_value,  # 使用正确的 bootstrap value
            batch['agent_masks']
        )

        # 标准化优势
        valid_mask = batch['agent_masks'].bool()
        if valid_mask.sum() > 1:
            adv_mean = advantages[valid_mask].mean()
            adv_std = advantages[valid_mask].std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # PPO 更新
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_losses = []

        T, n_agents = batch['observations'].shape[:2]
        total_samples = T * n_agents

        for epoch in range(self.ppo_epochs):
            # 随机打乱
            indices = torch.randperm(T)

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                mb_indices = indices[start:end]

                # 获取 mini-batch
                mb_obs = batch['observations'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_agent_masks = batch['agent_masks'][mb_indices]
                mb_action_masks = batch['action_masks'][mb_indices] if batch['action_masks'] is not None else None

                # 评估动作
                new_log_probs, values, entropy = self.model.evaluate_actions(
                    mb_obs, mb_actions, mb_agent_masks, mb_action_masks
                )

                # 计算 ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages

                # 只对活跃智能体计算损失
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * mb_agent_masks).sum() / (mb_agent_masks.sum() + 1e-8)

                value_loss = F.mse_loss(values * mb_agent_masks, mb_returns * mb_agent_masks, reduction='sum')
                value_loss = value_loss / (mb_agent_masks.sum() + 1e-8)

                entropy_loss = -(entropy * mb_agent_masks).sum() / (mb_agent_masks.sum() + 1e-8)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 更新学习率 (如果 scheduler 存在)
                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            'total_loss': total_loss / (self.ppo_epochs * (T // self.mini_batch_size + 1)),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }

    def train(self, total_episodes: int = 10000, rollout_steps: int = 2048,
              eval_interval: int = 200, save_interval: int = 500,
              log_interval: int = 50):
        """
        训练主循环

        Args:
            total_episodes: 总训练 episode 数
            rollout_steps: 每次 rollout 的步数
            eval_interval: 评估间隔
            save_interval: 保存间隔
            log_interval: 日志间隔
        """
        if self.experiment_dir is None:
            self.setup_experiment_dir()

        print("=" * 60)
        print("🚀 开始 IPPO 训练")
        print(f"   总 Episodes: {total_episodes}")
        print(f"   Rollout 步数: {rollout_steps}")
        print("=" * 60)

        start_time = time.time()

        # 初始化学习率调度器
        if self.scheduler is None:
            # 计算总步数：episodes * (rollout_steps / mini_batch_size) * ppo_epochs ?
            # 不，Standard practice for LinearLR is total_iters = number of scheduler steps.
            # 我们在 update 的内部循环中每次 step()，即每个 mini-batch 更新一次？
            # 或者每个 update() 更新一次？
            # 用户建议: total_iters=ppo_epochs*total_episodes
            # 这意味着建议我们在每个 epoch 甚至每个 update 调用 step。
            # 上面的代码是在 mini-batch 循环内部调用的 step ? No, I put it inside the mini-batch loop?
            # Wait, let's check where I put scheduler.step()
            # I put it inside the mini-batch loop (inner-most loop).
            # The total updates = total_episodes * (rollout_steps // mini_batch_size) * ppo_epochs
            # This is huge.
            # User suggestion: total_iters = ppo_epochs * total_episodes
            # This implies the user thinks we step once per epoch?
            # If I put step() inside the mini-batch loop, it decays too fast.
            # Let's put step() inside the epoch loop, or adjust total_iters.
            # Let's assume we step inside the innermost loop for finest granularity.
            # steps_per_episode = (rollout_steps // self.mini_batch_size + 1) * self.ppo_epochs
            # total_iters = total_episodes * steps_per_episode

            # 修正：为了简单且稳健，我们按照用户的建议 `total_iters=ppo_epochs*total_episodes`
            # 并在每个 epoch 结束时 step，或者在每次 update 时 step ppo_epochs 次。
            # 既然我把 step() 放在了最内层循环，那我应该把 total_iters 设为最内层循环的总次数。
            # updates_per_rollout = self.ppo_epochs * (rollout_steps // self.mini_batch_size)
            # total_iters = total_episodes * updates_per_rollout

            updates_per_rollout = self.ppo_epochs * (rollout_steps // self.mini_batch_size)
            total_iters = total_episodes * updates_per_rollout
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01,
                                                         total_iters=total_iters)

        while self.episode_count < total_episodes:
            # 收集经验
            rollout_info = self.collect_rollout(rollout_steps)

            # 更新策略
            if len(self.buffer) > 0:
                update_metrics = self.update()
            else:
                update_metrics = {}

            # 记录统计
            mean_reward = 0.0
            mean_length = 0.0
            success_rate = 0.0

            if rollout_info['episode_rewards']:
                mean_reward = np.mean(rollout_info['episode_rewards'])
                mean_length = np.mean(rollout_info['episode_lengths'])
                if rollout_info['episode_successes']:
                    success_rate = np.mean(rollout_info['episode_successes'])

                self.training_history['episode_rewards'].extend(rollout_info['episode_rewards'])
                self.training_history['episode_lengths'].extend(rollout_info['episode_lengths'])
                self.training_history['episode_success'].extend(rollout_info['episode_successes'])
                self.training_history['total_loss'].append(update_metrics.get('total_loss', 0))

                # ====== 学术评价指标（性能优化版）======
                if rollout_info.get('avg_travel_times'):
                    self.training_history['avg_travel_times'].extend(rollout_info['avg_travel_times'])
                if rollout_info.get('dangerous_approaches'):
                    self.training_history['dangerous_approaches'].extend(rollout_info['dangerous_approaches'])
                if rollout_info.get('lane_changes'):
                    self.training_history['lane_changes'].extend(rollout_info['lane_changes'])
                if rollout_info.get('target_lane_rates'):
                    self.training_history['target_lane_rates'].extend(rollout_info['target_lane_rates'])

            # 日志
            if self.episode_count % log_interval < len(rollout_info['episode_rewards']):
                elapsed = time.time() - start_time
                # 换行确保不被 SUMO 状态栏覆盖
                print(f"\n[Episode {self.episode_count:5d}] "
                      f"Reward: {mean_reward:7.2f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Success: {success_rate:.2%} | "
                      f"Loss: {update_metrics.get('total_loss', 0):.4f} | "
                      f"Time: {elapsed:.0f}s", flush=True)

            # 评估
            if self.episode_count % eval_interval < len(rollout_info['episode_rewards']):
                eval_metrics = self.evaluate(num_episodes=5)
                print(f"   📊 评估: Reward={eval_metrics['mean_reward']:.2f}, "
                      f"Success={eval_metrics['success_rate']:.2%}", flush=True)

            # 保存
            if self.episode_count % save_interval < len(rollout_info['episode_rewards']):
                self.save_checkpoint()

        # 最终保存
        self.save_checkpoint(final=True)
        self.save_training_stats()

        total_time = time.time() - start_time
        print("=" * 60)
        print(f"✅ 训练完成！总时间: {total_time / 60:.1f} 分钟")
        print(f"   总 Episodes: {self.episode_count}")
        print(f"   总步数: {self.total_steps}")
        print("=" * 60)

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估当前策略"""
        self.model.eval()

        episode_rewards = []
        episode_lengths = []
        successes = []

        for _ in range(num_episodes):
            # 重置环境
            obs_tuple = self.env.reset()
            raw_obs_tensor, global_state, agent_mask = obs_tuple

            # 应用 Normalization (不更新)
            obs_np = raw_obs_tensor.cpu().numpy()
            norm_obs_np = self.obs_rms.normalize(obs_np)
            obs_tensor = torch.tensor(norm_obs_np, dtype=torch.float32, device=self.device)

            raw_obs_tensor = raw_obs_tensor.to(self.device)
            agent_mask = agent_mask.to(self.device)

            episode_reward = 0
            episode_length = 0
            done = False
            info = {}

            while not done:
                action_mask = torch.ones(self.model.n_agents, self.model.action_dim, device=self.device)

                # 保存执行动作前的车辆ID
                prev_vehicle_ids = self.env.qminx.get_active_vehicle_ids()

                with torch.no_grad():
                    actions, _, _ = self.model.select_actions(
                        obs_tensor, agent_mask, action_mask, deterministic=True
                    )

                actions_list = actions.cpu().numpy().tolist()
                step_result = self.env.step(actions_list)
                next_raw_obs, next_global_state, next_agent_mask, reward, done, info = step_result

                # Normalization
                next_obs_np = next_raw_obs.cpu().numpy()
                next_norm_obs_np = self.obs_rms.normalize(next_obs_np)
                next_obs = torch.tensor(next_norm_obs_np, dtype=torch.float32, device=self.device)

                # 计算奖励 (使用原始观测，传入actions以计算意图奖励)
                individual_rewards = self._compute_individual_rewards(
                    info, raw_obs_tensor, agent_mask, prev_vehicle_ids, actions_list
                )
                episode_reward += sum(individual_rewards)
                episode_length += 1

                obs_tensor = next_obs
                raw_obs_tensor = next_raw_obs.to(self.device)
                agent_mask = next_agent_mask.to(self.device)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 与训练时保持一致：只有当 10 辆车全部在目标车道驶离栅格区域才记为成功
            total_exited = info.get('total_exited_vehicles', 0)
            vehicles_in_target = info.get('vehicles_in_target_lane', 0)
            required_agents = info.get('required_agents', self.model.n_agents)
            success_all = 1 if (total_exited >= required_agents and
                                vehicles_in_target >= required_agents) else 0
            successes.append(success_all)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean(successes)
        }

    def save_checkpoint(self, final: bool = False):
        """保存检查点"""
        if self.experiment_dir is None:
            return

        if final:
            path = self.experiment_dir / "models" / "final_model.pt"
        else:
            path = self.experiment_dir / "models" / f"checkpoint_{self.episode_count}.pt"

        self.model.save(str(path))

        # 保存 Normalization 统计量
        if hasattr(self, 'obs_rms'):
            rms_path = path.parent / f"rms_{path.stem}.json"
            with open(rms_path, 'w') as f:
                json.dump({
                    'mean': self.obs_rms.mean.tolist(),
                    'var': self.obs_rms.var.tolist(),
                    'count': self.obs_rms.count
                }, f)

    def save_training_stats(self):
        """保存训练统计"""
        if self.experiment_dir is None:
            return

        stats_path = self.experiment_dir / "training_stats.json"

        stats = {
            'episode_rewards': self.training_history['episode_rewards'],
            'episode_lengths': self.training_history['episode_lengths'],
            'episode_success': self.training_history['episode_success'],
            'total_loss': self.training_history['total_loss'],
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            # ====== 学术评价指标 ======
            'avg_travel_times': self.training_history.get('avg_travel_times', []),
            'dangerous_approaches': self.training_history.get('dangerous_approaches', []),
            'lane_changes': self.training_history.get('lane_changes', []),
            'target_lane_rates': self.training_history.get('target_lane_rates', []),
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 训练统计已保存: {stats_path}")

        # 自动生成可视化图片
        self.plot_training_results()

    def plot_training_results(self):
        """生成训练结果可视化图片（论文标准格式）"""
        if self.experiment_dir is None:
            return

        try:
            import matplotlib.pyplot as plt
            from scipy.ndimage import uniform_filter1d

            rewards = np.array(self.training_history['episode_rewards'])
            successes = np.array(self.training_history['episode_success'])

            if len(rewards) < 10:
                print("⚠️ 数据太少，跳过绘图")
                return

            # 论文标准字体设置
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.unicode_minus'] = False

            # ========== 图1: 论文标准奖励曲线（带阴影） ==========
            fig1, ax1 = plt.subplots(figsize=(8, 5))

            # 计算滑动窗口统计量
            window = min(20, len(rewards) // 10) if len(rewards) > 100 else 10

            # 计算均值和标准误差
            mean_rewards = uniform_filter1d(rewards, size=window, mode='nearest')

            # 计算滑动窗口标准差和标准误差
            std_rewards = np.zeros_like(rewards, dtype=float)
            for i in range(len(rewards)):
                start = max(0, i - window // 2)
                end = min(len(rewards), i + window // 2 + 1)
                std_rewards[i] = np.std(rewards[start:end])
            se_rewards = std_rewards / np.sqrt(window)  # 标准误差

            episodes = np.arange(1, len(rewards) + 1)

            # 绘制均值曲线和阴影区域
            ax1.plot(episodes, mean_rewards, color='#1f77b4', linewidth=2, label='IPPO')
            ax1.fill_between(episodes,
                             mean_rewards - se_rewards,
                             mean_rewards + se_rewards,
                             color='#1f77b4', alpha=0.3)

            ax1.set_xlabel('Episodes', fontsize=14)
            ax1.set_ylabel('Average Reward', fontsize=14)
            ax1.set_title('Training Performance', fontsize=16)
            ax1.legend(loc='lower right', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            plt.tight_layout()
            output_path1 = self.experiment_dir / 'reward_curve.png'
            plt.savefig(output_path1, dpi=300, bbox_inches='tight')
            plt.close()

            # ========== 图2: 成功率曲线（带阴影） ==========
            fig2, ax2 = plt.subplots(figsize=(8, 5))

            # 计算滑动成功率和置信区间
            mean_success = uniform_filter1d(successes.astype(float), size=window, mode='nearest')
            std_success = np.zeros_like(successes, dtype=float)
            for i in range(len(successes)):
                start = max(0, i - window // 2)
                end = min(len(successes), i + window // 2 + 1)
                std_success[i] = np.std(successes[start:end])
            se_success = std_success / np.sqrt(window)

            ax2.plot(episodes, mean_success, color='#2ca02c', linewidth=2, label='IPPO')
            ax2.fill_between(episodes,
                             np.clip(mean_success - se_success, 0, 1),
                             np.clip(mean_success + se_success, 0, 1),
                             color='#2ca02c', alpha=0.3)

            ax2.set_xlabel('Episodes', fontsize=14)
            ax2.set_ylabel('Success Rate', fontsize=14)
            ax2.set_title('Lane Change Success Rate', fontsize=16)
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc='lower right', fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            plt.tight_layout()
            output_path2 = self.experiment_dir / 'success_rate.png'
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            plt.close()

            # ========== 图3: 综合分析图（保留原版本用于调试） ==========
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            fig3, axes = plt.subplots(2, 2, figsize=(12, 9))

            # 奖励曲线
            ax = axes[0, 0]
            ax.plot(episodes, rewards, 'b-', alpha=0.2, linewidth=0.5)
            ax.plot(episodes, mean_rewards, 'r-', linewidth=2)
            ax.fill_between(episodes, mean_rewards - se_rewards, mean_rewards + se_rewards, alpha=0.3, color='red')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Rewards')
            ax.grid(True, alpha=0.3)

            # 成功率
            ax = axes[0, 1]
            ax.plot(episodes, mean_success, 'g-', linewidth=2)
            ax.fill_between(episodes, np.clip(mean_success - se_success, 0, 1),
                            np.clip(mean_success + se_success, 0, 1), alpha=0.3, color='green')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Success Rate')
            ax.set_title(f'Success Rate (Final: {np.mean(successes[-50:]):.1%})')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # 损失曲线
            losses = self.training_history['total_loss']
            ax = axes[1, 0]
            if losses:
                ax.plot(range(1, len(losses) + 1), losses, 'purple', alpha=0.5)
                ax.set_xlabel('Update')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                ax.grid(True, alpha=0.3)

            # 奖励分布
            ax = axes[1, 1]
            n = len(rewards)
            early, late = rewards[:n // 3], rewards[2 * n // 3:]
            ax.hist(early, bins=20, alpha=0.6, label=f'Early (μ={np.mean(early):.1f})', color='red')
            ax.hist(late, bins=20, alpha=0.6, label=f'Late (μ={np.mean(late):.1f})', color='green')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Count')
            ax.set_title('Reward Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path3 = self.experiment_dir / 'training_analysis.png'
            plt.savefig(output_path3, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"📈 论文标准图: {output_path1}")
            print(f"📈 成功率曲线: {output_path2}")
            print(f"📈 综合分析图: {output_path3}")

            # ========== 图4: 学术评价指标（新增） ==========
            avg_travel_times = self.training_history.get('avg_travel_times', [])
            dangerous_approaches = self.training_history.get('dangerous_approaches', [])
            lane_changes = self.training_history.get('lane_changes', [])
            target_lane_rates = self.training_history.get('target_lane_rates', [])

            # 只有当有数据时才绘制
            if len(avg_travel_times) >= 10:
                plt.rcParams['font.family'] = 'Times New Roman'
                fig4, axes = plt.subplots(2, 2, figsize=(12, 9))

                eval_episodes = np.arange(1, len(avg_travel_times) + 1)
                eval_window = min(20, len(avg_travel_times) // 5) if len(avg_travel_times) > 50 else max(5,
                                                                                                         len(avg_travel_times) // 5)

                # 4.1 平均通行时间
                ax = axes[0, 0]
                travel_arr = np.array(avg_travel_times)
                travel_smooth = uniform_filter1d(travel_arr, size=eval_window, mode='nearest')
                ax.plot(eval_episodes, travel_arr, 'b-', alpha=0.3, linewidth=0.8)
                ax.plot(eval_episodes, travel_smooth, 'b-', linewidth=2, label='Avg Travel Time')
                ax.set_xlabel('Episode', fontsize=12)
                ax.set_ylabel('Travel Time (steps)', fontsize=12)
                ax.set_title('Average Travel Time', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # 4.2 危险接近次数
                ax = axes[0, 1]
                danger_arr = np.array(dangerous_approaches)
                danger_smooth = uniform_filter1d(danger_arr.astype(float), size=eval_window, mode='nearest')
                ax.plot(eval_episodes, danger_arr, 'r-', alpha=0.3, linewidth=0.8)
                ax.plot(eval_episodes, danger_smooth, 'r-', linewidth=2, label='Dangerous Approaches')
                ax.set_xlabel('Episode', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Dangerous Approaches per Episode', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # 4.3 换道次数
                ax = axes[1, 0]
                lane_arr = np.array(lane_changes)
                lane_smooth = uniform_filter1d(lane_arr.astype(float), size=eval_window, mode='nearest')
                ax.plot(eval_episodes, lane_arr, 'orange', alpha=0.3, linewidth=0.8)
                ax.plot(eval_episodes, lane_smooth, 'orange', linewidth=2, label='Lane Changes')
                ax.set_xlabel('Episode', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Total Lane Changes per Episode', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # 4.4 目标车道达成率
                ax = axes[1, 1]
                target_arr = np.array(target_lane_rates)
                target_smooth = uniform_filter1d(target_arr, size=eval_window, mode='nearest')
                ax.plot(eval_episodes, target_arr, 'g-', alpha=0.3, linewidth=0.8)
                ax.plot(eval_episodes, target_smooth, 'g-', linewidth=2, label='Target Lane Rate')
                ax.set_xlabel('Episode', fontsize=12)
                ax.set_ylabel('Rate', fontsize=12)
                ax.set_title(f'Target Lane Achievement Rate (Final: {np.mean(target_arr[-20:]):.1%})', fontsize=14)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                ax.legend()

                plt.tight_layout()
                output_path4 = self.experiment_dir / 'academic_metrics.png'
                plt.savefig(output_path4, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"📈 学术评价指标图: {output_path4}")
            else:
                print(f"⚠️ 学术指标数据不足 ({len(avg_travel_times)} < 10)，跳过学术指标绘图")

        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
