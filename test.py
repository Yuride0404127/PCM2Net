# test.py
import os
import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse

import lpips  # LPIPS dependency

from data_loader import create_dataloaders
from metrics import tensor2img, calc_psnr, calc_ssim, calc_lpips
from models.Ablation_PCM2Net_v5_PID import PCM2Net


def save_result_image(save_path, sr_img, gt_img=None):
    """保存对比图或单张SR图"""
    if gt_img is not None:
        # 拼接图片进行对比: [SR, GT]
        combined = np.hstack((sr_img, gt_img))
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(save_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))


def test(config_path, model_path):
    # 1. Config & Setup
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 结果保存路径
    result_dir = os.path.join(
        config['experiment']['save_dir'],
        config['experiment']['full_name'],
        'results_v2'
    )
    os.makedirs(result_dir, exist_ok=True)

    # 2. Data Loader (Test mode)
    # 注意：这里我们只需要 test_loader
    _, _, test_loader = create_dataloaders(config, mode='full')

    # 3. Model Load
    print(f"Loading model from {model_path}...")
    # model = PCM2Net(
    #     dim=config['model']['base_dim'],
    #     scale_factor=config['data']['scale_ratio']
    # ).to(device)
    model = PCM2Net(
        dim=config['model']['base_dim'],
        scale_factor=config['data']['scale_ratio'],
        pid_use_edge=True,
        pid_use_flux_mod=True,
        pid_use_dt_mod=False,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # 兼容保存整个checkpoint或只保存权重的情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3.5 LPIPS Model Init (init once)
    # net 可选: 'alex'(更快), 'vgg'(更强但更慢), 'squeeze'
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_fn.eval()

    # 4. Inference Loop
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            lr = batch['lr_thermal'].to(device)
            rgb = batch['hr_rgb'].to(device)
            gt = batch['gt_thermal'].to(device)
            filename = batch['filename'][0]  # batch_size is 1

            # Forward
            sr = model(lr, rgb)
            sr = sr.clamp(0, 1)

            # Metrics (PSNR/SSIM use uint8 images)
            sr_img = tensor2img(sr)
            gt_img = tensor2img(gt)

            psnr = calc_psnr(gt_img, sr_img)
            ssim = calc_ssim(gt_img, sr_img)

            # LPIPS (use tensors in [0,1])
            # calc_lpips 会自动处理 1 通道复制到 3 通道，并转换到 [-1,1]
            lpips_val = calc_lpips(gt, sr, lpips_fn)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val
            count += 1

            # Save Image
            save_name = os.path.join(result_dir, filename)
            save_result_image(save_name, sr_img, None)

    print(f"Test Finished.")
    print(f"Average PSNR: {total_psnr / count:.4f}")
    print(f"Average SSIM: {total_ssim / count:.4f}")
    print(f"Average LPIPS: {total_lpips / count:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    args = parser.parse_args()

    test(args.config, args.model)
