"""
超参数配置管理模块
提供统一的配置文件管理、参数验证和配置模板功能

核心功能:
1. YAML配置文件加载和保存
2. 参数类型验证和范围检查
3. 默认配置模板生成
4. 配置继承和覆盖
5. 运行时配置更新
6. 配置版本管理

使用方法:
```python
from config_manager import ConfigManager

# 加载配置
config = ConfigManager.load_config('configs/qmix_default.yaml')

# 验证配置
ConfigManager.validate_config(config)

# 保存配置
ConfigManager.save_config(config, 'experiments/exp1/config.yaml')
```
"""

import os
import yaml
import json
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

class ConfigError(Exception):
    """配置相关错误"""
    pass

class ConfigManager:
    """
    配置管理器 - 统一管理所有超参数配置
    
    支持功能:
    - YAML配置文件读写
    - 参数类型和范围验证
    - 配置模板生成
    - 配置继承和合并
    - 实验配置版本控制
    """

    # === 全局路径配置 ===
    # 自动检测项目路径，无需手动修改
    # common/config_manager.py -> common -> DRL -> lxpaper
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # common/../.. = lxpaper
    DEFAULT_SUMO_CONFIG_PATH = str((_PROJECT_ROOT / 'sumo' / 'car.sumocfg').resolve())

    # === 默认配置架构定义 ===
    DEFAULT_CONFIG_SCHEMA = {
        # 训练超参数
        'training': {
            'total_episodes': {'type': int, 'min': 1, 'max': 10000, 'default': 5000},
            'batch_size': {'type': int, 'min': 1, 'max': 512, 'default': 32},
            'learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0, 'default': 0.0005},
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'epsilon_start': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 1.0},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.05},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.999},
            'grad_clip_norm': {'type': float, 'min': 0.1, 'max': 100.0, 'default': 10.0},
            'lr_decay_rate': {'type': float, 'min': 0.1, 'max': 1.0, 'default': 0.99},
            'lr_decay_interval': {'type': int, 'min': 100, 'max': 10000, 'default': 1000},
            # QMix多智能体Done训练策略参数
            'use_individual_rewards': {'type': bool, 'default': False},
            'agent_done_td_lambda': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.8},
            'bootstrap_terminated_agents': {'type': bool, 'default': True}
        },
        
        # 网络架构参数
        'network': {
            'n_agents': {'type': int, 'min': 1, 'max': 50, 'default': 10},
            'obs_dim': {'type': int, 'min': 1, 'max': 200, 'default': 23},  # 更新：22→23（新增冷却期状态）
            'action_dim': {'type': int, 'min': 2, 'max': 20, 'default': 4},
            'state_dim': {'type': int, 'min': 1, 'max': 100, 'default': 4},
            'agent_hidden_dim': {'type': int, 'min': 16, 'max': 1024, 'default': 128},
            'mixing_embed_dim': {'type': int, 'min': 8, 'max': 256, 'default': 32},
            'hypernet_embed': {'type': int, 'min': 16, 'max': 512, 'default': 64},
            # QMix多智能体网络特殊参数
            'enable_agent_masking': {'type': bool, 'default': True},
            'mixing_network_type': {'type': str, 'choices': ['qmix', 'vdn', 'qtran'], 'default': 'qmix'},
            'agent_communication': {'type': bool, 'default': False},
            'shared_agent_network': {'type': bool, 'default': True}
        },
        
        # 训练调度参数
        'schedule': {
            'target_update_interval': {'type': int, 'min': 50, 'max': 5000, 'default': 200},
            'collection_interval': {'type': int, 'min': 100, 'max': 10000, 'default': 1000},
            'training_interval': {'type': int, 'min': 1, 'max': 100, 'default': 4},
            'evaluation_interval': {'type': int, 'min': 1000, 'max': 50000, 'default': 1000},
            'save_interval': {'type': int, 'min': 1000, 'max': 100000, 'default': 10000},
            'log_interval': {'type': int, 'min': 100, 'max': 10000, 'default': 1000}
        },
        
        # 经验回放缓冲区参数
        'replay_buffer': {
            'capacity': {'type': int, 'min': 1000, 'max': 10000000, 'default': 100000},
            'warmup_size': {'type': int, 'min': 100, 'max': 10000, 'default': 1000},
            'priority_replay': {'type': bool, 'default': False},
            'priority_alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.6},
            'priority_beta': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.4},
            # QMix多智能体Done处理参数
            'enable_agent_done_mask': {'type': bool, 'default': True},
            'agent_done_filter_threshold': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'individual_termination_handling': {'type': str, 'choices': ['mask_based', 'reward_based', 'hybrid'], 'default': 'mask_based'},
            'agent_done_penalty_scale': {'type': float, 'min': 0.0, 'max': 10.0, 'default': 1.0},
            'filter_incomplete_episodes': {'type': bool, 'default': False}
        },
        
        # 环境配置
        'environment': {
            'max_episode_length': {'type': int, 'min': 100, 'max': 10000, 'default': 500},
            'sumo_cfg_path': {'type': str, 'default': DEFAULT_SUMO_CONFIG_PATH},
            'sumo_gui': {'type': bool, 'default': False},
            'grid_width': {'type': float, 'min': 1.0, 'max': 10.0, 'default': 3.5},
            'grid_length': {'type': float, 'min': 1.0, 'max': 20.0, 'default': 5.0},
            'grid_length': {'type': float, 'min': 1.0, 'max': 20.0, 'default': 5.0},
            'lane_change_reward_scale': {'type': float, 'min': 0.1, 'max': 10.0, 'default': 1.0},
            'collision_penalty': {'type': float, 'min': -100.0, 'max': -1.0, 'default': -10.0}
        },
        
        # 系统配置
        'system': {
            'device': {'type': str, 'choices': ['auto', 'cpu', 'cuda'], 'default': 'auto'},
            'num_workers': {'type': int, 'min': 1, 'max': 16, 'default': 4},
            'seed': {'type': int, 'min': 0, 'max': 999999, 'default': 42},
            'log_level': {'type': str, 'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR'], 'default': 'INFO'},
            'tensorboard_log': {'type': bool, 'default': True},
            'save_replay_buffer': {'type': bool, 'default': False}
        },
        
        # 实验配置
        'experiment': {
            'name': {'type': str, 'default': ''},
            'description': {'type': str, 'default': ''},
            'tags': {'type': list, 'default': []},
            'save_dir': {'type': str, 'default': 'qmix_experiments'},
            'checkpoint_every': {'type': int, 'min': 1000, 'max': 100000, 'default': 10000}
        }
    }
    
    @classmethod
    def generate_default_config(cls) -> Dict[str, Any]:
        """
        生成默认配置字典
        
        Returns:
            config: 默认配置字典
        """
        config = {}
        
        for section_name, section_schema in cls.DEFAULT_CONFIG_SCHEMA.items():
            config[section_name] = {}
            for param_name, param_info in section_schema.items():
                config[section_name][param_name] = param_info['default']
        
        # 添加元信息
        config['_meta'] = {
            'config_version': '1.0',
            'created_at': datetime.now().isoformat(),
            'created_by': 'ConfigManager.generate_default_config()'
        }
        
        return config
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            config: 配置字典
            
        Raises:
            ConfigError: 配置文件加载失败
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ConfigError("配置文件为空")
            
            # 验证配置
            cls.validate_config(config)
            
            # 补充缺失的默认值
            config = cls.merge_with_defaults(config)
            
            print(f"✅ 配置已加载: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML解析错误: {e}")
        except Exception as e:
            raise ConfigError(f"配置加载失败: {e}")
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config: 配置字典
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加保存时间戳
        config_copy = copy.deepcopy(config)
        if '_meta' not in config_copy:
            config_copy['_meta'] = {}
        config_copy['_meta']['saved_at'] = datetime.now().isoformat()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_copy, f, 
                         default_flow_style=False,
                         allow_unicode=True,
                         indent=2)
            
            print(f"💾 配置已保存: {save_path}")
            
        except Exception as e:
            raise ConfigError(f"配置保存失败: {e}")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """
        验证配置参数的类型和范围
        
        Args:
            config: 待验证的配置字典
            
        Raises:
            ConfigError: 配置验证失败
        """
        for section_name, section_schema in cls.DEFAULT_CONFIG_SCHEMA.items():
            if section_name not in config:
                continue  # 允许部分配置缺失，后续用默认值填充
            
            section_config = config[section_name]
            
            for param_name, param_info in section_schema.items():
                if param_name not in section_config:
                    continue  # 允许参数缺失
                
                value = section_config[param_name]
                cls._validate_parameter(section_name, param_name, value, param_info)
        
        # === QMix特殊参数逻辑校验 ===
        cls._validate_qmix_specific_logic(config)
    
    @classmethod
    def _validate_parameter(cls, section_name: str, param_name: str, 
                          value: Any, param_info: Dict[str, Any]) -> None:
        """
        验证单个参数
        
        Args:
            section_name: 配置段名称
            param_name: 参数名称
            value: 参数值
            param_info: 参数信息字典
            
        Raises:
            ConfigError: 参数验证失败
        """
        param_type = param_info['type']
        param_path = f"{section_name}.{param_name}"
        
        # 类型检查 - 简化版本，避免复杂的类型检查
        if param_type == int and not isinstance(value, int):
            raise ConfigError(f"参数 {param_path} 类型错误: 期望 int, 得到 {type(value).__name__}")
        elif param_type == float and not isinstance(value, (int, float)):
            raise ConfigError(f"参数 {param_path} 类型错误: 期望 float, 得到 {type(value).__name__}")
        elif param_type == str and not isinstance(value, str):
            raise ConfigError(f"参数 {param_path} 类型错误: 期望 str, 得到 {type(value).__name__}")
        elif param_type == bool and not isinstance(value, bool):
            raise ConfigError(f"参数 {param_path} 类型错误: 期望 bool, 得到 {type(value).__name__}")
        elif param_type == list and not isinstance(value, list):
            raise ConfigError(f"参数 {param_path} 类型错误: 期望 list, 得到 {type(value).__name__}")
        
        # 数值范围检查
        if param_type in [int, float]:
            if 'min' in param_info and value < param_info['min']:
                raise ConfigError(f"参数 {param_path} 小于最小值: {value} < {param_info['min']}")
            if 'max' in param_info and value > param_info['max']:
                raise ConfigError(f"参数 {param_path} 大于最大值: {value} > {param_info['max']}")
        
        # 选择值检查
        if 'choices' in param_info:
            if value not in param_info['choices']:
                raise ConfigError(f"参数 {param_path} 不在允许的选择中: {value} not in {param_info['choices']}")
        
        # 字符串路径检查
        if param_type == str and param_name.endswith('_path') and value:
            try:
                expanded_path = os.path.expanduser(str(value))
                if not os.path.exists(expanded_path):
                    print(f"⚠️ 警告: 路径 {param_path} = {value} 不存在")
            except Exception:
                # 如果路径检查失败，忽略
                pass
    
    @classmethod
    def _validate_qmix_specific_logic(cls, config: Dict[str, Any]) -> None:
        """
        验证QMix特殊参数的逻辑关系
        
        Args:
            config: 配置字典
            
        Raises:
            ConfigError: QMix特殊逻辑验证失败
        """
        try:
            # 1. 检查batch_size和buffer_capacity的关系
            if 'training' in config and 'replay_buffer' in config:
                batch_size = config['training'].get('batch_size', 32)
                buffer_capacity = config['replay_buffer'].get('capacity', 100000)
                warmup_size = config['replay_buffer'].get('warmup_size', 1000)
                
                if batch_size > buffer_capacity:
                    raise ConfigError(f"batch_size ({batch_size}) 不能大于 buffer_capacity ({buffer_capacity})")
                
                if warmup_size > buffer_capacity:
                    raise ConfigError(f"warmup_size ({warmup_size}) 不能大于 buffer_capacity ({buffer_capacity})")
                
                if batch_size > warmup_size:
                    print(f"⚠️ 警告: batch_size ({batch_size}) 大于 warmup_size ({warmup_size})，可能影响训练稳定性")
            
            # 2. 检查agent_dones相关参数的逻辑一致性
            if 'replay_buffer' in config:
                rb_config = config['replay_buffer']
                enable_mask = rb_config.get('enable_agent_done_mask', True)
                handling_type = rb_config.get('individual_termination_handling', 'mask_based')
                filter_threshold = rb_config.get('agent_done_filter_threshold', 0.1)
                
                # 如果启用agent_done_mask，但处理方式不是mask_based，给出警告
                if enable_mask and handling_type != 'mask_based':
                    print(f"⚠️ 警告: enable_agent_done_mask=True 但 individual_termination_handling='{handling_type}'，可能存在冲突")
                
                # 检查过滤阈值的合理性
                if not enable_mask and filter_threshold > 0:
                    print(f"⚠️ 警告: agent_done掩码未启用但设置了过滤阈值 ({filter_threshold})，该参数将被忽略")
            
            # 3. 检查网络架构参数的合理性
            if 'network' in config:
                net_config = config['network']
                n_agents = net_config.get('n_agents', 10)
                mixing_type = net_config.get('mixing_network_type', 'qmix')
                agent_comm = net_config.get('agent_communication', False)
                shared_net = net_config.get('shared_agent_network', True)
                
                # 检查智能体数量与网络类型的兼容性
                if mixing_type == 'vdn' and n_agents > 20:
                    print(f"⚠️ 警告: VDN网络在智能体数量过多 ({n_agents}) 时可能性能较差")
                
                # 检查通信机制和共享网络的兼容性
                if agent_comm and not shared_net:
                    print(f"⚠️ 警告: 启用智能体通信但禁用共享网络，可能影响通信效果")
            
            # 4. 检查训练参数与QMix特性的兼容性
            if 'training' in config:
                train_config = config['training']
                use_individual = train_config.get('use_individual_rewards', False)
                bootstrap = train_config.get('bootstrap_terminated_agents', True)
                
                # 如果使用个体奖励但不引导已终止智能体，给出建议
                if use_individual and not bootstrap:
                    print(f"💡 建议: 使用个体奖励时建议启用 bootstrap_terminated_agents 以提高训练稳定性")
            
            # 5. 检查目标网络更新频率的合理性
            if 'schedule' in config and 'training' in config:
                target_interval = config['schedule'].get('target_update_interval', 200)
                total_episodes = config['training'].get('total_episodes', 500)
                
                if target_interval > total_episodes * 50:  # 假设每个Episode平均50步
                    print(f"⚠️ 警告: 目标网络更新间隔 ({target_interval}) 相对训练Episodes ({total_episodes}) 过大")
                
                if target_interval < 50:
                    print(f"⚠️ 警告: 目标网络更新间隔 ({target_interval}) 过小，可能影响训练稳定性")
        
        except Exception as e:
            raise ConfigError(f"QMix特殊逻辑验证失败: {e}")
    
    @classmethod
    def merge_with_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将配置与默认值合并，补充缺失的参数
        
        Args:
            config: 部分配置字典
            
        Returns:
            merged_config: 合并后的完整配置
        """
        default_config = cls.generate_default_config()
        merged_config = copy.deepcopy(default_config)
        
        # 递归合并配置
        cls._deep_merge(merged_config, config)
        
        return merged_config
    
    @classmethod
    def _deep_merge(cls, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        深度合并两个字典
        
        Args:
            target: 目标字典（会被修改）
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                cls._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    @classmethod
    def create_config_template(cls, template_name: str, save_dir: str = 'configs') -> str:
        """
        创建配置模板文件
        
        Args:
            template_name: 模板名称 ('default', 'fast_train', 'high_quality', 'debug')
            save_dir: 保存目录
            
        Returns:
            template_path: 生成的模板文件路径
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if template_name == 'default':
            config = cls.generate_default_config()
            config['experiment']['name'] = 'qmix_default'
            config['experiment']['description'] = 'Default QMIX configuration for lane changing'
            
        elif template_name == 'fast_train':
            config = cls.generate_default_config()
            # 快速训练配置
            config['training']['total_episodes'] = 100
            config['training']['batch_size'] = 64
            config['training']['learning_rate'] = 0.001
            config['schedule']['evaluation_interval'] = 2000
            config['schedule']['save_interval'] = 5000
            config['replay_buffer']['capacity'] = 50000
            config['experiment']['name'] = 'qmix_fast_train'
            config['experiment']['description'] = 'Fast training configuration for quick experiments'
            config['experiment']['tags'] = ['fast', 'development']
            
        elif template_name == 'high_quality':
            config = cls.generate_default_config()
            # 高质量训练配置
            config['training']['total_episodes'] = 2000
            config['training']['batch_size'] = 32
            config['training']['learning_rate'] = 0.0003
            config['training']['epsilon_decay'] = 0.999
            config['network']['agent_hidden_dim'] = 256
            config['network']['mixing_embed_dim'] = 64
            config['schedule']['target_update_interval'] = 500
            config['replay_buffer']['capacity'] = 500000
            config['experiment']['name'] = 'qmix_high_quality'
            config['experiment']['description'] = 'High quality training configuration for best performance'
            config['experiment']['tags'] = ['high-quality', 'production']
            
        elif template_name == 'debug':
            config = cls.generate_default_config()
            # 调试配置
            config['training']['total_episodes'] = 20
            config['training']['batch_size'] = 8
            config['schedule']['log_interval'] = 50
            config['schedule']['evaluation_interval'] = 200
            config['schedule']['save_interval'] = 500
            config['replay_buffer']['capacity'] = 5000
            config['system']['log_level'] = 'DEBUG'
            config['environment']['sumo_gui'] = True
            config['experiment']['name'] = 'qmix_debug'
            config['experiment']['description'] = 'Debug configuration for development and testing'
            config['experiment']['tags'] = ['debug', 'development']
            
        elif template_name == 'agent_done_focused':
            config = cls.generate_default_config()
            # 专门测试agent_done处理的配置
            config['training']['total_episodes'] = 200
            config['training']['use_individual_rewards'] = True
            config['training']['bootstrap_terminated_agents'] = True
            config['training']['agent_done_td_lambda'] = 0.9
            config['replay_buffer']['enable_agent_done_mask'] = True
            config['replay_buffer']['individual_termination_handling'] = 'hybrid'
            config['replay_buffer']['agent_done_filter_threshold'] = 0.2
            config['replay_buffer']['agent_done_penalty_scale'] = 2.0
            config['replay_buffer']['filter_incomplete_episodes'] = True
            config['network']['enable_agent_masking'] = True
            config['schedule']['target_update_interval'] = 100  # 更频繁的更新
            config['experiment']['name'] = 'qmix_agent_done_focused'
            config['experiment']['description'] = 'Configuration focused on testing agent_done handling mechanisms'
            config['experiment']['tags'] = ['agent_done', 'multi_agent', 'termination']
            
        else:
            raise ConfigError(f"未知的模板名称: {template_name}")
        
        # 保存模板
        template_path = os.path.join(save_dir, f'{template_name}.yaml')
        cls.save_config(config, template_path)
        
        print(f"📋 配置模板已创建: {template_path}")
        return template_path
    
    @classmethod
    def update_config_from_args(cls, config: Dict[str, Any], args) -> Dict[str, Any]:
        """
        从命令行参数更新配置
        
        Args:
            config: 基础配置字典
            args: argparse.Namespace 对象
            
        Returns:
            updated_config: 更新后的配置
        """
        updated_config = copy.deepcopy(config)
        
        # 映射命令行参数到配置路径
        arg_mapping = {
            # 训练参数
            'total_episodes': ('training', 'total_episodes'),
            'batch_size': ('training', 'batch_size'),
            'learning_rate': ('training', 'learning_rate'),
            'gamma': ('training', 'gamma'),
            'epsilon_start': ('training', 'epsilon_start'),
            'epsilon_min': ('training', 'epsilon_min'),
            'epsilon_decay': ('training', 'epsilon_decay'),
            
            # 网络参数
            'n_agents': ('network', 'n_agents'),
            'obs_dim': ('network', 'obs_dim'),
            'action_dim': ('network', 'action_dim'),
            'state_dim': ('network', 'state_dim'),
            'agent_hidden_dim': ('network', 'agent_hidden_dim'),
            'mixing_embed_dim': ('network', 'mixing_embed_dim'),
            
            # 调度参数
            'target_update_interval': ('schedule', 'target_update_interval'),
            'collection_interval': ('schedule', 'collection_interval'),
            'training_interval': ('schedule', 'training_interval'),
            'evaluation_interval': ('schedule', 'evaluation_interval'),
            'save_interval': ('schedule', 'save_interval'),
            'log_interval': ('schedule', 'log_interval'),
            
            # 缓冲区参数
            'buffer_capacity': ('replay_buffer', 'capacity'),
            
            # 系统参数
            'device': ('system', 'device'),
            'seed': ('system', 'seed'),
            
            # 实验参数
            'exp_name': ('experiment', 'name'),
            'save_dir': ('experiment', 'save_dir'),
            
            # 环境参数
            'sumo_cfg_path': ('environment', 'sumo_cfg_path')
        }
        
        # 应用命令行参数覆盖
        for arg_name, (section, param) in arg_mapping.items():
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value is not None:
                    updated_config[section][param] = arg_value
        
        return updated_config
    
    @classmethod
    def get_config_summary(cls, config: Dict[str, Any]) -> str:
        """
        获取配置摘要字符串
        
        Args:
            config: 配置字典
            
        Returns:
            summary: 配置摘要字符串
        """
        lines = []
        lines.append("=== QMIX Configuration Summary ===")
        
        # 实验信息
        exp_name = config.get('experiment', {}).get('name', 'Unnamed')
        exp_desc = config.get('experiment', {}).get('description', '')
        lines.append(f"Experiment: {exp_name}")
        if exp_desc:
            lines.append(f"Description: {exp_desc}")
        
        # 关键训练参数
        training = config.get('training', {})
        lines.append(f"\nTraining:")
        lines.append(f"  Episodes: {training.get('total_episodes', 'N/A'):,}")
        lines.append(f"  Batch Size: {training.get('batch_size', 'N/A')}")
        lines.append(f"  Learning Rate: {training.get('learning_rate', 'N/A')}")
        lines.append(f"  Gamma: {training.get('gamma', 'N/A')}")
        
        # 网络架构
        network = config.get('network', {})
        lines.append(f"\nNetwork:")
        lines.append(f"  Agents: {network.get('n_agents', 'N/A')}")
        lines.append(f"  Obs Dim: {network.get('obs_dim', 'N/A')}")
        lines.append(f"  Action Dim: {network.get('action_dim', 'N/A')}")
        lines.append(f"  Hidden Dim: {network.get('agent_hidden_dim', 'N/A')}")
        
        # 系统配置
        system = config.get('system', {})
        lines.append(f"\nSystem:")
        lines.append(f"  Device: {system.get('device', 'N/A')}")
        lines.append(f"  Seed: {system.get('seed', 'N/A')}")
        
        # QMix多智能体Done处理配置
        replay_buffer = config.get('replay_buffer', {})
        lines.append(f"\nQMix Agent Done Handling:")
        lines.append(f"  Enable Mask: {replay_buffer.get('enable_agent_done_mask', 'N/A')}")
        lines.append(f"  Handling Type: {replay_buffer.get('individual_termination_handling', 'N/A')}")
        lines.append(f"  Filter Threshold: {replay_buffer.get('agent_done_filter_threshold', 'N/A')}")
        
        return '\n'.join(lines)
    
    @classmethod
    def create_experiment_config(cls, base_config_path: str, 
                                experiment_name: str,
                                modifications: Dict[str, Any],
                                save_dir: str = 'experiments') -> str:
        """
        基于基础配置创建实验特定的配置
        
        Args:
            base_config_path: 基础配置文件路径
            experiment_name: 实验名称
            modifications: 要修改的配置项字典
            save_dir: 保存目录
            
        Returns:
            exp_config_path: 实验配置文件路径
        """
        # 加载基础配置
        base_config = cls.load_config(base_config_path)
        
        # 应用修改
        exp_config = copy.deepcopy(base_config)
        cls._deep_merge(exp_config, modifications)
        
        # 更新实验信息
        exp_config['experiment']['name'] = experiment_name
        exp_config['experiment']['description'] = f"Based on {base_config_path} with custom modifications"
        
        # 保存实验配置
        exp_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        exp_config_path = os.path.join(exp_dir, 'config.yaml')
        
        cls.save_config(exp_config, exp_config_path)
        
        print(f"🧪 实验配置已创建: {exp_config_path}")
        return exp_config_path 