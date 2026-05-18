# models/PMG2NetLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Charbonnier
# -----------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))


# -----------------------------
# Edge loss (stable)
# -----------------------------
class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=torch.float32)

        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3)
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        tgt_gray  = target.mean(dim=1, keepdim=True)

        pred_gx = self.sobel_x(pred_gray)
        pred_gy = self.sobel_y(pred_gray)
        tgt_gx  = self.sobel_x(tgt_gray)
        tgt_gy  = self.sobel_y(tgt_gray)

        pred_edge = torch.sqrt(pred_gx**2 + pred_gy**2 + self.eps)
        tgt_edge  = torch.sqrt(tgt_gx**2 + tgt_gy**2 + self.eps)

        return F.l1_loss(pred_edge, tgt_edge)


# -----------------------------
# MS-SSIM loss (preferred)
# -----------------------------
class MSSSIMLoss(nn.Module):
    """
    pred/target should be in [0,1]. (at least mostly)
    """
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range
        self.use_ms = False
        try:
            from pytorch_msssim import ms_ssim
            self.ms_ssim = ms_ssim
            self.use_ms = True
        except Exception:
            self.use_ms = False

    def forward(self, pred, target):
        if self.use_ms:
            return 1.0 - self.ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        # fallback: simple SSIM
        return 1.0 - self._ssim_simple(pred, target)

    def _ssim_simple(self, x, y, C1=0.01**2, C2=0.03**2):
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
        sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()


# -----------------------------
# Best combined SR loss + stage schedule
# -----------------------------
class BestSRLoss(nn.Module):
    """
    Practical best combo for PSNR + SSIM:
    - Full-res Charbonnier (PSNR driver)
    - Weighted multi-scale Charbonnier (weak constraint)
    - MS-SSIM (SSIM driver), ramp up after warmup
    - Edge loss (tiny), ramp up late
    """
    def __init__(
        self,
        w_pix=1.0,
        w_ms=0.2,
        ms_weights=(1.0, 0.5, 0.25),   # scales 1,2,4 (decreasing)
        final_w_ssim=0.12,             # late-stage target
        final_w_edge=0.005,            # late-stage target (keep small)
        warmup_ratio=0.6,              # first 60% epochs: focus PSNR
        data_range=1.0
    ):
        super().__init__()
        self.l_pix = CharbonnierLoss()
        self.l_edge = EdgeLoss()
        self.l_ssim = MSSSIMLoss(data_range=data_range)

        self.w_pix = w_pix
        self.w_ms = w_ms
        self.ms_weights = ms_weights

        # scheduled weights (updated via set_epoch)
        self.final_w_ssim = final_w_ssim
        self.final_w_edge = final_w_edge
        self.warmup_ratio = warmup_ratio
        self.w_ssim = 0.0
        self.w_edge = 0.0

    @torch.no_grad()
    def set_epoch(self, epoch: int, total_epochs: int):
        t = epoch / max(total_epochs - 1, 1)
        if t <= self.warmup_ratio:
            self.w_ssim = 0.0  # 也可以设为 0.03 让结构更早介入
            self.w_edge = 0.0
        else:
            ramp = (t - self.warmup_ratio) / max(1e-6, (1.0 - self.warmup_ratio))
            self.w_ssim = float(self.final_w_ssim * ramp)
            self.w_edge = float(self.final_w_edge * ramp)

    def forward(self, pred, target):
        # 让loss更“可控”：训练时也 clamp，避免网络输出越界导致 SSIM 项不稳定
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        # 1) full-res charbonnier
        loss_pix = self.l_pix(pred, target)

        # 2) weighted multi-scale charbonnier
        scales = [1, 2, 4]
        ms_loss = 0.0
        wsum = 0.0
        for w, s in zip(self.ms_weights, scales):
            pred_s = F.interpolate(pred, scale_factor=1/s, mode='bilinear', align_corners=False)
            tgt_s  = F.interpolate(target, scale_factor=1/s, mode='bilinear', align_corners=False)
            ms_loss += w * self.l_pix(pred_s, tgt_s)
            wsum += w
        loss_ms = ms_loss / max(wsum, 1e-6)

        # 3) ssim loss (optional)
        loss_ssim = self.l_ssim(pred, target) if self.w_ssim > 0 else pred.new_tensor(0.0)

        # 4) edge loss (optional)
        loss_edge = self.l_edge(pred, target) if self.w_edge > 0 else pred.new_tensor(0.0)

        total = (self.w_pix * loss_pix +
                 self.w_ms * loss_ms +
                 self.w_ssim * loss_ssim +
                 self.w_edge * loss_edge)

        # 返回细项方便你在 tqdm/tensorboard 看
        logs = {
            "loss_pix": loss_pix.detach(),
            "loss_ms": loss_ms.detach(),
            "loss_ssim": loss_ssim.detach(),
            "loss_edge": loss_edge.detach(),
            "w_ssim": torch.tensor(self.w_ssim),
            "w_edge": torch.tensor(self.w_edge),
        }
        return total, logs
