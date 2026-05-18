"""
改进的数据加载模块 - 支持自动数据集划分（修正版）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import random
import glob
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalSRDataset(Dataset):
    """热成像超分辨率数据集类"""
    
    def __init__(self, file_list: List[Dict[str, str]], mode: str = 'patch', 
                 degradation: str = 'BI', scale_ratio: int = 4, 
                 patch_size: int = 48, augmentation: bool = False):
        """
        Args:
            file_list: 文件列表
            mode: 'patch' 或 'full' - 训练模式
            degradation: 'BI' or 'BD'
            scale_ratio: 4 or 8
            patch_size: LR patch大小 (仅在patch模式下使用)
            augmentation: 是否进行数据增强
        """
        self.file_list = file_list
        self.mode = mode
        self.degradation = degradation
        self.scale_ratio = scale_ratio
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale_ratio
        self.augmentation = augmentation
        
        logger.info(f"Dataset created with {len(self.file_list)} samples in {mode} mode")
        logger.info(f"LR patch size: {self.patch_size}, HR patch size: {self.hr_patch_size}")
        
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        file_info = self.file_list[idx]
        
        gt_thermal = self._load_image(file_info['gt_thermal'], mode='RGB')  
        hr_rgb = self._load_image(file_info['hr_rgb'], mode='RGB')
        lr_thermal = self._load_image(file_info['lr_thermal'], mode='RGB')  
        
        # 根据模式处理数据
        if self.mode == 'patch':
            lr_thermal, gt_thermal, hr_rgb = self._extract_patch(lr_thermal, gt_thermal, hr_rgb)
        
        # 数据增强
        if self.augmentation:
            lr_thermal, gt_thermal, hr_rgb = self._augment(lr_thermal, gt_thermal, hr_rgb)
        
        # 转换为tensor - 修正：3通道图像不需要expand_dims
        lr_thermal_tensor = self._to_tensor(lr_thermal, expand_dims=False)
        gt_thermal_tensor = self._to_tensor(gt_thermal, expand_dims=False)
        hr_rgb_tensor = self._to_tensor(hr_rgb, expand_dims=False)
        
        return {
            'lr_thermal': lr_thermal_tensor,
            'hr_rgb': hr_rgb_tensor,  # 使用hr_rgb保持与您的输出一致
            'gt_thermal': gt_thermal_tensor,
            'filename': os.path.basename(file_info['gt_thermal'])
        }
    
    def _load_image(self, path: str, mode: str = 'RGB') -> np.ndarray:
        """加载并归一化图像"""
        img = Image.open(path).convert(mode)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    
    def _to_tensor(self, img: np.ndarray, expand_dims: bool = True) -> torch.Tensor:
        """转换为PyTorch tensor"""
        tensor = torch.from_numpy(img).float()
        
        # 处理维度
        if len(tensor.shape) == 2 and expand_dims:
            # 2D灰度图像，添加通道维度
            tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 3:
            # 3D彩色图像，转换为 CHW 格式
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return tensor
    
    def _extract_patch(self, lr_thermal: np.ndarray, gt_thermal: np.ndarray, 
                      hr_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取patch - 修正尺寸关系"""
        # LR图像的高度和宽度
        lr_h, lr_w = lr_thermal.shape[:2]  # 修正：处理3通道图像
        
        # 如果图像太小，使用填充
        if lr_h < self.patch_size or lr_w < self.patch_size:
            lr_thermal, gt_thermal, hr_rgb = self._pad_images(
                lr_thermal, gt_thermal, hr_rgb, lr_h, lr_w
            )
            lr_h, lr_w = lr_thermal.shape[:2]
        
        # 随机选择裁剪位置（基于LR图像）
        lr_top = random.randint(0, lr_h - self.patch_size)
        lr_left = random.randint(0, lr_w - self.patch_size)
        
        # 对应的高分辨率位置
        hr_top = lr_top * self.scale_ratio
        hr_left = lr_left * self.scale_ratio
        
        # 裁剪patches
        lr_patch = lr_thermal[lr_top:lr_top + self.patch_size, 
                             lr_left:lr_left + self.patch_size]
        gt_patch = gt_thermal[hr_top:hr_top + self.hr_patch_size,
                             hr_left:hr_left + self.hr_patch_size]
        rgb_patch = hr_rgb[hr_top:hr_top + self.hr_patch_size,
                          hr_left:hr_left + self.hr_patch_size]
        
        return lr_patch, gt_patch, rgb_patch
    
    def _pad_images(self, lr_thermal: np.ndarray, gt_thermal: np.ndarray, 
                   hr_rgb: np.ndarray, lr_h: int, lr_w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """填充图像 - 修正：处理3通道图像"""
        pad_h = max(0, self.patch_size - lr_h)
        pad_w = max(0, self.patch_size - lr_w)
        
        # 对于3通道图像，需要正确设置padding
        lr_thermal = np.pad(lr_thermal, 
                           ((0, pad_h), (0, pad_w), (0, 0)), 
                           mode='reflect')
        gt_thermal = np.pad(gt_thermal, 
                           ((0, pad_h * self.scale_ratio), (0, pad_w * self.scale_ratio), (0, 0)), 
                           mode='reflect')
        hr_rgb = np.pad(hr_rgb, 
                       ((0, pad_h * self.scale_ratio), (0, pad_w * self.scale_ratio), (0, 0)), 
                       mode='reflect')
        
        return lr_thermal, gt_thermal, hr_rgb
    
    def _augment(self, lr_thermal: np.ndarray, gt_thermal: np.ndarray, 
                hr_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            lr_thermal = np.fliplr(lr_thermal).copy()
            gt_thermal = np.fliplr(gt_thermal).copy()
            hr_rgb = np.fliplr(hr_rgb).copy()
        
        # 随机垂直翻转
        if random.random() > 0.5:
            lr_thermal = np.flipud(lr_thermal).copy()
            gt_thermal = np.flipud(gt_thermal).copy()
            hr_rgb = np.flipud(hr_rgb).copy()
        
        # 随机旋转90度
        if random.random() > 0.5:
            k = random.randint(1, 3)
            lr_thermal = np.rot90(lr_thermal, k).copy()
            gt_thermal = np.rot90(gt_thermal, k).copy()
            hr_rgb = np.rot90(hr_rgb, k).copy()
        
        return lr_thermal, gt_thermal, hr_rgb


def collate_padding(batch: List[Dict]):
    """
    自定义的collate_fn，用于将不同尺寸的图像填充到Batch中的最大尺寸
    适用于 mode='full' 且 batch_size > 1 的情况
    """
    # 1. 收集文件名
    filenames = [item['filename'] for item in batch]
    
    # 2. 计算当前Batch中 LR 的最大 H 和 W
    lr_shapes = [item['lr_thermal'].shape for item in batch]
    max_lr_h = max([s[1] for s in lr_shapes])
    max_lr_w = max([s[2] for s in lr_shapes])
    
    # 3. 计算当前Batch中 HR/GT 的最大 H 和 W
    # 注意：这里分别计算是为了防止因计算误差导致的维度对不齐，
    # 但通常 max_hr_h 应该是 max_lr_h * scale_ratio
    gt_shapes = [item['gt_thermal'].shape for item in batch]
    max_gt_h = max([s[1] for s in gt_shapes])
    max_gt_w = max([s[2] for s in gt_shapes])
    
    # 初始化列表
    lr_batch = []
    hr_rgb_batch = []
    gt_batch = []
    
    for item in batch:
        lr = item['lr_thermal']
        hr_rgb = item['hr_rgb']
        gt = item['gt_thermal']
        
        # --- 填充 LR ---
        # F.pad 参数顺序: (左, 右, 上, 下)
        pad_lr_h = max_lr_h - lr.shape[1]
        pad_lr_w = max_lr_w - lr.shape[2]
        # 使用 reflect 填充可以减少边界伪影，对SR任务更友好，也可以改用 'constant', value=0
        lr_padded = F.pad(lr, (0, pad_lr_w, 0, pad_lr_h), mode='reflect')
        lr_batch.append(lr_padded)
        
        # --- 填充 HR RGB ---
        pad_hr_h = max_gt_h - hr_rgb.shape[1]
        pad_hr_w = max_gt_w - hr_rgb.shape[2]
        hr_rgb_padded = F.pad(hr_rgb, (0, pad_hr_w, 0, pad_hr_h), mode='reflect')
        hr_rgb_batch.append(hr_rgb_padded)
        
        # --- 填充 GT Thermal ---
        # 确保 GT 和 HR RGB 填充逻辑一致
        gt_padded = F.pad(gt, (0, pad_hr_w, 0, pad_hr_h), mode='reflect')
        gt_batch.append(gt_padded)
        
    # 4. 堆叠成 Tensor
    return {
        'lr_thermal': torch.stack(lr_batch),
        'hr_rgb': torch.stack(hr_rgb_batch),
        'gt_thermal': torch.stack(gt_batch),
        'filename': filenames
    }


def load_file_list(root_dir: str, degradation: str = 'BI', 
                  scale_ratio: int = 4) -> List[Dict[str, str]]:
    """加载文件列表"""
    # 构建数据路径
    gt_thermal_dir = os.path.join(root_dir, 'GT_thermal')
    hr_rgb_dir = os.path.join(root_dir, 'HR_RGB')
    lr_thermal_dir = os.path.join(root_dir, 'LR_thermal', degradation, f'X{scale_ratio}')
    
    # 检查目录是否存在
    for dir_path, dir_name in [(gt_thermal_dir, 'GT_thermal'), 
                               (hr_rgb_dir, 'HR_RGB'), 
                               (lr_thermal_dir, 'LR_thermal')]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")
    
    # 获取所有GT thermal文件
    gt_files = sorted(glob.glob(os.path.join(gt_thermal_dir, '*.[pj][np][ge]*')))
    
    if len(gt_files) == 0:
        raise ValueError(f"No files found in {gt_thermal_dir}")
    
    # 构建文件列表
    file_list = []
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        # prefix = basename.split('.')[0]
        prefix = os.path.splitext(basename)[0]
        
        # 构建对应的文件路径
        hr_rgb_file = os.path.join(hr_rgb_dir, basename)
        # lr_thermal_file = os.path.join(lr_thermal_dir, f'{prefix}x{scale_ratio}.png')
        ext = os.path.splitext(basename)[1]
        lr_thermal_file = os.path.join(lr_thermal_dir, f'{prefix}x{scale_ratio}{ext}')
        
        # 检查文件是否都存在
        if os.path.exists(hr_rgb_file) and os.path.exists(lr_thermal_file):
            file_list.append({
                'gt_thermal': gt_file,
                'hr_rgb': hr_rgb_file,
                'lr_thermal': lr_thermal_file
            })
        else:
            logger.warning(f"Missing files for {basename}")
            if not os.path.exists(hr_rgb_file):
                logger.warning(f"  Missing HR RGB: {hr_rgb_file}")
            if not os.path.exists(lr_thermal_file):
                logger.warning(f"  Missing LR thermal: {lr_thermal_file}")
    
    logger.info(f"Found {len(file_list)} valid image triplets")
    return file_list


def split_dataset(file_list: List[Dict[str, str]], train_ratio: float, 
                 val_ratio: float, test_ratio: float, 
                 seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """划分数据集"""
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        logger.warning(f"Data split ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # 计算各部分大小
    total_size = len(file_list)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(file_list)
    
    # 划分数据集
    train_files = file_list[:train_size]
    val_files = file_list[train_size:train_size + val_size]
    test_files = file_list[train_size + val_size:]
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    return train_files, val_files, test_files


def create_dataloaders(config: Dict, mode: str = 'patch') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        mode: 'patch' 或 'full' - 训练模式
    """
    # 加载文件列表
    file_list = load_file_list(
        config['data']['root'],
        config['data']['degradation'],
        config['data']['scale_ratio']
    )
    
    # 划分数据集
    train_files, val_files, test_files = split_dataset(
        file_list,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio'],
        config['experiment']['seed']
    )
    
    # 创建数据集
    common_args = {
        'degradation': config['data']['degradation'],
        'scale_ratio': config['data']['scale_ratio'],
        'patch_size': config['data']['patch_size']
    }
    
    train_dataset = ThermalSRDataset(
        train_files,
        mode=mode,
        augmentation=config['data']['augmentation'],
        **common_args
    )
    
    val_dataset = ThermalSRDataset(
        val_files,
        mode='full',  # 验证时总是使用完整图像
        augmentation=False,
        **common_args
    )
    
    test_dataset = ThermalSRDataset(
        test_files,
        mode='full',  # 测试时总是使用完整图像
        augmentation=False,
        **common_args
    )
    
    # 批量大小
    if mode == 'patch':
        train_batch_size = config['training']['batch_size']
    else:
        train_batch_size = config['training'].get('full_batch_size', 1)

    # 决定是否使用自定义 collate_fn
    # 如果是 patch 模式，Dataset 会保证输出固定尺寸，使用默认 default_collate 即可
    # 如果是 full 模式且 batch_size > 1，必须使用填充
    use_padding_collate = (mode == 'full' and train_batch_size > 1)
    train_collate_fn = collate_padding if use_padding_collate else None
    
    # 数据加载器参数
    loader_kwargs = {
        'num_workers': config['dataloader']['num_workers'],
        'pin_memory': config['dataloader']['pin_memory'],
    }
    
    if config['dataloader']['num_workers'] > 0:
        loader_kwargs.update({
            'prefetch_factor': config['dataloader']['prefetch_factor'],
            'persistent_workers': config['dataloader']['persistent_workers']
        })
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Data loaders created in {mode} mode:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # 验证数据维度
    if len(train_loader) > 0:
        sample_batch = next(iter(train_loader))
        logger.info("Sample batch shapes:")
        logger.info(f"  lr_thermal: {sample_batch['lr_thermal'].shape}")
        logger.info(f"  hr_rgb: {sample_batch['hr_rgb'].shape}")  # 改为hr_rgb
        logger.info(f"  gt_thermal: {sample_batch['gt_thermal'].shape}")
    
    return train_loader, val_loader, test_loader