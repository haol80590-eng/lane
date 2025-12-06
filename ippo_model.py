"""
IPPO (Independent PPO) 模型
每个智能体使用独立的 Actor-Critic 网络，但可以共享参数

适用场景：
- 多智能体独立决策
- 个体奖励
- 弱协作/避让型任务
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    
    Actor: 输出动作概率分布
    Critic: 输出状态价值估计
    
    输入维度：22维局部观测（更新：20→22，新增TTC安全特征）
    - 自身状态 (3维): 当前车道 + 最优目标车道 + 是否需换道
    - 局部感知 (15维): 3×5局部网格占用状态
    - 邻车意图 (2维): 左邻车想右换道 + 右邻车想左换道
    - TTC安全 (2维): 前车TTC + 目标车道最小TTC（新增 2025-12-02）
    
    Actor 输出：4维动作概率
    Critic 输出：1维状态价值
    """
    
    def __init__(self, obs_dim: int = 22, action_dim: int = 4, hidden_dim: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # === 共享特征提取层 ===
        # 观测结构：[自身状态3] + [局部栅格15] + [邻车意图2] + [TTC安全2] = 22维
        
        # 1. 自身状态特征提取 (3维 → 32维)
        #    包含：当前车道、目标车道、是否需换道
        self.self_state_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # 2. 局部网格特征提取 (15维 → 64维)
        #    包含：3×5 局部栅格占用状态
        self.local_grid_encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. 邻车意图特征提取 (2维 → 16维)
        #    包含：左邻车想右换道、右邻车想左换道
        #    这是关键的协作信号，需要单独编码以增强其影响力
        self.neighbor_intent_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # 4. TTC安全特征提取 (2维 → 16维) 【新增 2025-12-02】
        #    包含：前车TTC、目标车道最小TTC
        #    让Agent能感知碰撞风险，学会安全换道
        self.ttc_safety_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # 5. 特征融合层 (32+64+16+16 → 128)
        #    将四种特征整合为统一的决策表示
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 + 64 + 16 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # === Actor 头 (策略网络) ===
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # === Critic 头 (价值网络) ===
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Actor 输出层使用更小的初始化
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        # Critic 输出层使用标准初始化
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
    
    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        提取共享特征
        
        观测结构 (22维):
        - [0:3]   自身状态: 当前车道、目标车道、是否需换道
        - [3:18]  局部栅格: 3×5 网格占用状态
        - [18:20] 邻车意图: 左邻车想右换道、右邻车想左换道
        - [20:22] TTC安全: 前车TTC、目标车道最小TTC（新增）
        """
        # 确保输入是批次格式
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 分离四种特征
        self_state = obs[:, :3]           # 自身状态 (3维)
        local_grid = obs[:, 3:18]         # 局部栅格 (15维)
        neighbor_intent = obs[:, 18:20]   # 邻车意图 (2维)
        ttc_safety = obs[:, 20:22]        # TTC安全 (2维) 【新增】
        
        # 分别编码四种特征
        self_features = self.self_state_encoder(self_state)         # → 32维
        grid_features = self.local_grid_encoder(local_grid)         # → 64维
        intent_features = self.neighbor_intent_encoder(neighbor_intent)  # → 16维
        ttc_features = self.ttc_safety_encoder(ttc_safety)          # → 16维 【新增】
        
        # 融合特征
        combined = torch.cat([self_features, grid_features, intent_features, ttc_features], dim=1)
        features = self.feature_fusion(combined)
        
        return features
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测 [batch_size, obs_dim]
            action_mask: 动作掩码 [batch_size, action_dim]，1=有效，0=无效
            
        Returns:
            action_probs: 动作概率 [batch_size, action_dim]
            state_value: 状态价值 [batch_size, 1]
        """
        single_input = obs.dim() == 1
        if single_input:
            obs = obs.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        
        # 提取特征
        features = self._extract_features(obs)
        
        # Actor: 计算动作 logits
        action_logits = self.actor_head(features)
        
        # 应用动作掩码
        if action_mask is not None:
            # 将无效动作的 logits 设为很小的负数
            action_logits = action_logits.masked_fill(action_mask == 0, -1e8)
        
        # Softmax 得到概率
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: 计算状态价值
        state_value = self.critic_head(features)
        
        if single_input:
            return action_probs.squeeze(0), state_value.squeeze(0)
        return action_probs, state_value
    
    def get_action(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        根据观测选择动作
        
        Args:
            obs: 观测
            action_mask: 动作掩码
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(obs, action_mask)
            
            if deterministic:
                action = action_probs.argmax().item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
            
            log_prob = torch.log(action_probs[action] + 1e-8).item()
            value = state_value.item() if state_value.dim() == 0 else state_value.squeeze().item()
            
            return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                        action_mask: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于 PPO 更新）
        
        Args:
            obs: 观测 [batch_size, obs_dim]
            actions: 动作 [batch_size]
            action_mask: 动作掩码 [batch_size, action_dim]
            
        Returns:
            log_probs: 动作对数概率 [batch_size]
            values: 状态价值 [batch_size]
            entropy: 策略熵 [batch_size]
        """
        action_probs, state_values = self.forward(obs, action_mask)
        
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy


class IPPOModel:
    """
    IPPO 模型管理器
    
    管理多个智能体的 Actor-Critic 网络
    支持参数共享模式
    """
    
    def __init__(self,
                 n_agents: int = 10,
                 obs_dim: int = 22,  # 更新：20→22，新增TTC安全特征
                 action_dim: int = 4,
                 hidden_dim: int = 128,
                 shared_params: bool = True,
                 device: str = 'cpu'):
        """
        初始化 IPPO 模型
        
        Args:
            n_agents: 智能体数量
            obs_dim: 观测维度 (22维：3自身+15栅格+2意图+2TTC)
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            shared_params: 是否共享参数（推荐 True）
            device: 计算设备
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.device = torch.device(device)
        
        # 创建网络
        if shared_params:
            # 所有智能体共享同一个网络
            self.network = ActorCriticNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
            self.networks = None
        else:
            # 每个智能体有独立的网络
            self.network = None
            self.networks = nn.ModuleList([
                ActorCriticNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
                for _ in range(n_agents)
            ])
    
    def get_network(self, agent_idx: int = 0) -> ActorCriticNetwork:
        """获取指定智能体的网络"""
        if self.shared_params:
            return self.network
        else:
            return self.networks[agent_idx]
    
    def parameters(self):
        """获取所有可训练参数"""
        if self.shared_params:
            return self.network.parameters()
        else:
            return self.networks.parameters()
    
    def select_actions(self,
                      observations: torch.Tensor,
                      agent_mask: torch.Tensor,
                      action_masks: Optional[torch.Tensor] = None,
                      deterministic: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为所有智能体选择动作
        
        Args:
            observations: 所有智能体的观测 [n_agents, obs_dim]
            agent_mask: 智能体掩码 [n_agents]，1=活跃，0=非活跃
            action_masks: 动作掩码 [n_agents, action_dim]
            deterministic: 是否确定性选择
            
        Returns:
            actions: 选择的动作 [n_agents]
            log_probs: 动作对数概率 [n_agents]
            values: 状态价值 [n_agents]
        """
        n_agents = observations.size(0)
        actions = torch.zeros(n_agents, dtype=torch.long, device=self.device)
        log_probs = torch.zeros(n_agents, device=self.device)
        values = torch.zeros(n_agents, device=self.device)
        
        for i in range(n_agents):
            if agent_mask[i] == 0:
                continue
            
            obs = observations[i]
            action_mask = action_masks[i] if action_masks is not None else None
            network = self.get_network(i)
            
            action, log_prob, value = network.get_action(obs, action_mask, deterministic)
            actions[i] = action
            log_probs[i] = log_prob
            values[i] = value
        
        return actions, log_probs, values
    
    def evaluate_actions(self,
                        observations: torch.Tensor,
                        actions: torch.Tensor,
                        agent_mask: torch.Tensor,
                        action_masks: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估所有智能体的动作
        
        Args:
            observations: [batch_size, n_agents, obs_dim]
            actions: [batch_size, n_agents]
            agent_mask: [batch_size, n_agents]
            action_masks: [batch_size, n_agents, action_dim]
            
        Returns:
            log_probs: [batch_size, n_agents]
            values: [batch_size, n_agents]
            entropy: [batch_size, n_agents]
        """
        batch_size = observations.size(0)
        n_agents = observations.size(1)
        
        log_probs = torch.zeros(batch_size, n_agents, device=self.device)
        values = torch.zeros(batch_size, n_agents, device=self.device)
        entropy = torch.zeros(batch_size, n_agents, device=self.device)
        
        for i in range(n_agents):
            obs_i = observations[:, i, :]  # [batch_size, obs_dim]
            actions_i = actions[:, i]  # [batch_size]
            mask_i = agent_mask[:, i]  # [batch_size]
            action_mask_i = action_masks[:, i, :] if action_masks is not None else None
            
            network = self.get_network(i)
            
            # 只对活跃的智能体计算
            active_indices = mask_i.nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                continue
            
            active_obs = obs_i[active_indices]
            active_actions = actions_i[active_indices]
            active_action_mask = action_mask_i[active_indices] if action_mask_i is not None else None
            
            lp, v, ent = network.evaluate_actions(active_obs, active_actions, active_action_mask)
            
            log_probs[active_indices, i] = lp
            values[active_indices, i] = v
            entropy[active_indices, i] = ent
        
        return log_probs, values, entropy
    
    def save(self, path: str):
        """保存模型"""
        if self.shared_params:
            torch.save(self.network.state_dict(), path)
        else:
            torch.save(self.networks.state_dict(), path)
        print(f"💾 IPPO 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if self.shared_params:
            self.network.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
        else:
            self.networks.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
        print(f"📂 IPPO 模型已加载: {path}")
    
    def train(self):
        """设置为训练模式"""
        if self.shared_params:
            self.network.train()
        else:
            self.networks.train()
    
    def eval(self):
        """设置为评估模式"""
        if self.shared_params:
            self.network.eval()
        else:
            self.networks.eval()
