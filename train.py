# train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import argparse
import sys
from datetime import datetime 

# 导入你的模块
from data_loader import create_dataloaders
from metrics import tensor2img, calc_psnr, calc_ssim
from models.Ablation_PGM2Net_v5_PID import PGM2Net
from models.PMG2NetLoss import MultiScaleLoss



def setup_logger(save_dir):
    """
    配置全局日志：同时输出到控制台和文件
    关键点：配置 root logger 以捕获 data_loader 等所有模块的日志
    """
    log_file = os.path.join(save_dir, 'train.log')

    # 1. 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 2. 定义格式 (匹配你要求的格式: INFO:name:message)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # 3. 创建文件 Handler
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' 覆盖模式，'a' 追加模式
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 4. 添加到根 logger（防止重复添加）
    handlers = root_logger.handlers
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file
        for h in handlers
    )
    if not has_file_handler:
        root_logger.addHandler(file_handler)

    # 5. 确保控制台也有输出
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in handlers)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    return logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=True)

    for batch in pbar:
        # 1. 获取数据 (匹配 dataloader 的 keys)
        lr = batch['lr_thermal'].to(device)
        rgb = batch['hr_rgb'].to(device)
        gt = batch['gt_thermal'].to(device)

        # 2. 前向传播
        optimizer.zero_grad()
        output = model(lr, rgb)

        # 3. 计算损失
        loss = criterion(output, gt)

        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()

        # 5. 记录
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0

    # 验证阶段不计算梯度，节省显存
    with torch.no_grad():
        for batch in tqdm(loader, desc='[Val]'):
            lr = batch['lr_thermal'].to(device)
            rgb = batch['hr_rgb'].to(device)
            gt = batch['gt_thermal'].to(device)

            # 推理
            output = model(lr, rgb)

            # 后处理：Clamp并转为numpy uint8
            output = output.clamp(0, 1)

            # 计算 Batch 中每张图的指标 (验证集 batch_size 通常为 1)
            for i in range(len(output)):
                sr_img = tensor2img(output[i])
                gt_img = tensor2img(gt[i])

                psnr_sum += calc_psnr(gt_img, sr_img)
                ssim_sum += calc_ssim(gt_img, sr_img)

    avg_psnr = psnr_sum / len(loader.dataset)
    avg_ssim = ssim_sum / len(loader.dataset)

    return avg_psnr, avg_ssim


def main(config_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_name = config['experiment']['name']
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')  # Windows/Linux 都安全
    exp_name = f"{base_name}_{time_str}"

    # 可选：写回 config，方便记录复现实验
    config['experiment']['full_name'] = exp_name

    # 3. 设置环境（用追加时间的 exp_name 创建目录）
    save_dir = os.path.join(config['experiment']['save_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    logger = setup_logger(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb_logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment name (base): {base_name}")
    logger.info(f"Experiment name (final): {exp_name}")
    logger.info(f"Save dir: {save_dir}")

    resolved_cfg_path = os.path.join(save_dir, 'config_resolved.yaml')
    with open(resolved_cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Resolved config saved to: {resolved_cfg_path}")

    # 4. 数据加载 (使用你的 dataloader.py)
    train_loader, val_loader, _ = create_dataloaders(config, mode='patch')

    # 5. 模型初始化
    logger.info("Initializing model...")
    # model = PGM2Net(
    #     dim=config['model']['base_dim'],
    #     scale_factor=config['data']['scale_ratio']
    # ).to(device)
    model = PGM2Net(
        dim=config['model']['base_dim'],
        scale_factor=config['data']['scale_ratio'],
        pid_use_edge=True,
        pid_use_flux_mod=True,
        pid_use_dt_mod=False,
    ).to(device)

    # 6. 优化器和损失函数
    # criterion = nn.L1Loss() # 或者 nn.MSELoss()
    criterion = MultiScaleLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=1e-4
    )

    # 学习率调整策略
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr']  # 需要在 yaml 中添加这个参数
    )

    # 7. 训练循环
    best_psnr = 0.0
    start_epoch = 0
    total_epochs = config['training']['epochs']

    for epoch in range(start_epoch, total_epochs):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch, total_epochs)
        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # 记录 Training Log
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/Loss', train_loss, epoch)
        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | LR: {current_lr:.2e}"
        )

        # 更新学习率
        scheduler.step()

        # --- Validation (每隔N个epoch测一次) ---
        if (epoch + 1) % config['training']['val_freq'] == 0:
            val_psnr, val_ssim = validate(model, val_loader, device)

            writer.add_scalar('Val/PSNR', val_psnr, epoch)
            writer.add_scalar('Val/SSIM', val_ssim, epoch)
            logger.info(f"Epoch {epoch} | Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")

            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoints', 'best.pth'))
                logger.info("New best model saved!")

        # 定期保存 checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(save_dir, 'checkpoints', 'latest.pth'))

    writer.close()
    logger.info("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    main(args.config)
