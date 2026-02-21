"""
SinMDM Wrapper for Dunhuang Dance Motion Generation System
敦煌舞蹈动作生成系统 - SinMDM 模型封装

封装 SinMDM 的训练和推理接口
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# SinMDM 路径配置
SINMDM_ROOT = Path(__file__).parent.parent.parent / "sinmdm"


class SinMDMWrapper:
    """SinMDM 模型封装器"""
    
    def __init__(self, sinmdm_path: Optional[str] = None):
        """
        初始化 SinMDM 封装器
        
        Args:
            sinmdm_path: SinMDM 代码根目录路径
        """
        self.sinmdm_path = Path(sinmdm_path) if sinmdm_path else SINMDM_ROOT
        
        if not self.sinmdm_path.exists():
            raise FileNotFoundError(f"SinMDM not found at: {self.sinmdm_path}")
        
        # 添加到 Python 路径
        if str(self.sinmdm_path) not in sys.path:
            sys.path.insert(0, str(self.sinmdm_path))
        
        self.model = None
        self.model_path = None
        self.device = 'cuda'
        
        # 默认训练参数
        self.default_train_args = {
            'arch': 'unet',
            'dataset': 'bvh_general',
            'lr_method': 'ExponentialLR',
            'lr_gamma': 0.99998,
            'use_scale_shift_norm': True,
            'use_checkpoint': True,
        }
        
        # 默认生成参数
        self.default_gen_args = {
            'num_samples': 1,
            'motion_length': 10.0,  # 秒
            'seed': None,
        }
    
    def train(
        self,
        bvh_path: str,
        save_dir: str,
        epochs: int = 50000,
        **kwargs
    ) -> str:
        """
        训练 SinMDM 模型
        
        Args:
            bvh_path: 输入 BVH 文件路径
            save_dir: 模型保存目录
            epochs: 训练轮数
            **kwargs: 额外的训练参数
            
        Returns:
            str: 训练完成的模型路径
        """
        import subprocess
        
        bvh_path = Path(bvh_path).absolute()
        save_dir = Path(save_dir).absolute()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建训练命令
        cmd = [
            sys.executable, '-m', 'train.train_sinmdm',
            '--arch', kwargs.get('arch', self.default_train_args['arch']),
            '--dataset', 'bvh_general',
            '--save_dir', str(save_dir),
            '--sin_path', str(bvh_path),
            '--lr_method', self.default_train_args['lr_method'],
            '--lr_gamma', str(self.default_train_args['lr_gamma']),
        ]
        
        if kwargs.get('use_scale_shift_norm', True):
            cmd.append('--use_scale_shift_norm')
        if kwargs.get('use_checkpoint', True):
            cmd.append('--use_checkpoint')
        if kwargs.get('device'):
            cmd.extend(['--device', str(kwargs['device'])])
        if kwargs.get('seed'):
            cmd.extend(['--seed', str(kwargs['seed'])])
        
        print(f"[SinMDM] Starting training...")
        print(f"[SinMDM] Input: {bvh_path}")
        print(f"[SinMDM] Output: {save_dir}")
        
        # 执行训练
        result = subprocess.run(
            cmd,
            cwd=str(self.sinmdm_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[SinMDM] Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        print(f"[SinMDM] Training completed")
        
        # 查找最新的模型文件
        model_files = list(save_dir.glob("model*.pt"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.model_path = str(latest_model)
            return self.model_path
        
        return str(save_dir)
    
    def generate(
        self,
        model_path: str,
        num_samples: int = 1,
        motion_length: float = 10.0,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用训练好的模型生成动作
        
        Args:
            model_path: 模型路径
            num_samples: 生成样本数
            motion_length: 生成动作时长（秒）
            output_dir: 输出目录
            seed: 随机种子
            **kwargs: 额外参数
            
        Returns:
            Dict: 包含生成结果的字典
                - 'bvh_files': 生成的 BVH 文件列表
                - 'npy_file': numpy 数据文件
                - 'video_file': 可视化视频文件（如果有）
        """
        import subprocess
        
        model_path = Path(model_path).absolute()
        
        if output_dir is None:
            output_dir = model_path.parent / 'generated'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建生成命令
        cmd = [
            sys.executable, '-m', 'sample.generate',
            '--model_path', str(model_path),
            '--num_samples', str(num_samples),
            '--motion_length', str(motion_length),
        ]
        
        if seed is not None:
            cmd.extend(['--seed', str(seed)])
        
        print(f"[SinMDM] Generating {num_samples} sample(s)...")
        print(f"[SinMDM] Motion length: {motion_length}s")
        
        # 执行生成
        result = subprocess.run(
            cmd,
            cwd=str(self.sinmdm_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[SinMDM] Generation failed: {result.stderr}")
            raise RuntimeError(f"Generation failed: {result.stderr}")
        
        print(f"[SinMDM] Generation completed")
        
        # 收集生成结果
        results = {
            'bvh_files': [],
            'npy_file': None,
            'video_file': None,
            'output_dir': str(output_dir)
        }
        
        # 查找生成的文件（SinMDM 默认输出位置）
        save_dir = model_path.parent if model_path.is_file() else model_path
        
        for bvh_file in save_dir.glob("sample*.bvh"):
            results['bvh_files'].append(str(bvh_file))
        
        npy_files = list(save_dir.glob("results*.npy"))
        if npy_files:
            results['npy_file'] = str(npy_files[0])
        
        mp4_files = list(save_dir.glob("*.mp4"))
        if mp4_files:
            results['video_file'] = str(mp4_files[0])
        
        return results
    
    def load_model(self, model_path: str):
        """
        加载预训练模型（可选：用于更细粒度的控制）
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        print(f"[SinMDM] Model path set: {model_path}")
        # 实际加载需要根据 SinMDM 的具体实现
    
    def get_training_command(
        self,
        bvh_path: str,
        save_dir: str,
        **kwargs
    ) -> str:
        """
        获取训练命令（不执行）
        
        用于用户手动执行或调试
        """
        cmd_parts = [
            'python', '-m', 'train.train_sinmdm',
            '--arch', kwargs.get('arch', 'unet'),
            '--dataset', 'bvh_general',
            '--save_dir', f'"{save_dir}"',
            '--sin_path', f'"{bvh_path}"',
            '--lr_method', 'ExponentialLR',
            '--lr_gamma', '0.99998',
            '--use_scale_shift_norm',
            '--use_checkpoint',
        ]
        
        return ' '.join(cmd_parts)
    
    def get_generation_command(
        self,
        model_path: str,
        num_samples: int = 1,
        motion_length: float = 10.0,
        seed: Optional[int] = None
    ) -> str:
        """
        获取生成命令（不执行）
        """
        cmd_parts = [
            'python', '-m', 'sample.generate',
            '--model_path', f'"{model_path}"',
            '--num_samples', str(num_samples),
            '--motion_length', str(motion_length),
        ]
        
        if seed is not None:
            cmd_parts.extend(['--seed', str(seed)])
        
        return ' '.join(cmd_parts)
