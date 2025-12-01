"""
Configuration Management System for PointNet/PointNet++ experiments

This module provides a centralized configuration management system for tracking
experimental parameters, model hyperparameters, and training configurations.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import datetime


class ExperimentConfig:
    """
    Centralized configuration management for experiments
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment configuration
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict if config_dict is not None else {}
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter"""
        self.config[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter"""
        return self.config.get(key, default)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with a dictionary"""
        self.config.update(config_dict)
    
    def save_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def save_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.config, indent=4)


class ModelConfig:
    """Configuration for model architectures"""
    
    @staticmethod
    def get_pointnet_config() -> Dict[str, Any]:
        """Default PointNet configuration"""
        return {
            'model_name': 'pointnet_cls',
            'global_feat': True,
            'feature_transform': True,
            'channel': 3,
            'dropout': 0.4,
            'fc_dims': [1024, 512, 256],
            'batch_norm': True
        }
    
    @staticmethod
    def get_pointnet2_ssg_config() -> Dict[str, Any]:
        """Default PointNet++ SSG configuration"""
        return {
            'model_name': 'pointnet2_cls_ssg',
            'sa_configs': [
                {'npoint': 512, 'radius': 0.2, 'nsample': 32, 'mlp': [64, 64, 128]},
                {'npoint': 128, 'radius': 0.4, 'nsample': 64, 'mlp': [128, 128, 256]},
                {'npoint': None, 'radius': None, 'nsample': None, 'mlp': [256, 512, 1024]}
            ],
            'dropout': 0.4,
            'fc_dims': [512, 256],
            'normal_channel': True
        }
    
    @staticmethod
    def get_pointnet2_msg_config() -> Dict[str, Any]:
        """Default PointNet++ MSG configuration"""
        return {
            'model_name': 'pointnet2_cls_msg',
            'sa_configs': [
                {
                    'npoint': 512,
                    'radius_list': [0.1, 0.2, 0.4],
                    'nsample_list': [16, 32, 128],
                    'mlp_list': [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
                },
                {
                    'npoint': 128,
                    'radius_list': [0.2, 0.4, 0.8],
                    'nsample_list': [32, 64, 128],
                    'mlp_list': [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
                }
            ],
            'dropout': 0.4,
            'fc_dims': [512, 256]
        }


class TrainingConfig:
    """Configuration for training parameters"""
    
    @staticmethod
    def get_classification_config() -> Dict[str, Any]:
        """Default classification training configuration"""
        return {
            'task': 'classification',
            'batch_size': 24,
            'num_point': 1024,
            'num_category': 40,
            'epoch': 200,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'decay_rate': 1e-4,
            'lr_scheduler': 'StepLR',
            'lr_step_size': 20,
            'lr_gamma': 0.7,
            'use_normals': False,
            'use_uniform_sample': False,
            'data_augmentation': {
                'random_point_dropout': True,
                'random_scale': True,
                'random_shift': True
            }
        }
    
    @staticmethod
    def get_partseg_config() -> Dict[str, Any]:
        """Default part segmentation training configuration"""
        return {
            'task': 'part_segmentation',
            'batch_size': 16,
            'num_point': 2048,
            'epoch': 250,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'decay_rate': 1e-4,
            'lr_scheduler': 'StepLR',
            'lr_step_size': 20,
            'lr_gamma': 0.5,
            'normal': True
        }
    
    @staticmethod
    def get_semseg_config() -> Dict[str, Any]:
        """Default semantic segmentation training configuration"""
        return {
            'task': 'semantic_segmentation',
            'batch_size': 8,
            'num_point': 4096,
            'test_area': 5,
            'epoch': 100,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'decay_rate': 1e-4,
            'lr_scheduler': 'StepLR',
            'lr_step_size': 10,
            'lr_gamma': 0.7,
            'block_size': 1.0,
            'sample_rate': 1.0
        }


def create_experiment_config(args, task='classification') -> ExperimentConfig:
    """
    Create experiment configuration from command line arguments
    
    Args:
        args: Parsed command line arguments
        task: Task type (classification, part_segmentation, semantic_segmentation)
    
    Returns:
        ExperimentConfig object
    """
    config = ExperimentConfig()
    
    # General settings
    config.set('task', task)
    config.set('timestamp', config.timestamp)
    
    # Model settings
    config.set('model', vars(args).get('model', 'pointnet_cls'))
    
    # Training settings
    for key, value in vars(args).items():
        config.set(key, value)
    
    return config


def save_experiment_config(config: ExperimentConfig, experiment_dir: str) -> None:
    """
    Save experiment configuration to both YAML and JSON formats
    
    Args:
        config: ExperimentConfig object
        experiment_dir: Directory to save configuration files
    """
    yaml_path = os.path.join(experiment_dir, 'config.yaml')
    json_path = os.path.join(experiment_dir, 'config.json')
    
    config.save_yaml(yaml_path)
    config.save_json(json_path)
    
    print(f"Configuration saved to:\n  - {yaml_path}\n  - {json_path}")
