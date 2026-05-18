import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("Warning: mamba_ssm not found. PhysicsMambaBlock will fail if not using a fallback.")
    selective_scan_fn = None
    Mamba = None

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self, x):
        res = self.conv2(self.act1(self.conv1(x)))
        return x + res

class EdgeGradientExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dim=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(dim, dim * 2, 3, padding=1), 
            nn.ReLU(True)
        )
        
        self.sobel_x = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, bias=False, groups=dim * 2)
        self.sobel_y = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, bias=False, groups=dim * 2)
        
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x.weight.data = sobel_kernel_x.view(1, 1, 3, 3).repeat(dim * 2, 1, 1, 1) / 8.0
        self.sobel_y.weight.data = sobel_kernel_y.view(1, 1, 3, 3).repeat(dim * 2, 1, 1, 1) / 8.0
        
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.conv_block(x)
        g_x = self.sobel_x(feat)
        g_y = self.sobel_y(feat)
        grad_magnitude = torch.cat([torch.abs(g_x), torch.abs(g_y)], dim=1)
        edge_map = self.fusion(grad_magnitude)
        diff_coeff = 1.0 - edge_map 
        return diff_coeff, edge_map

class PhysicsMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_model / 16)
            
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        
        self.physics_bias_proj = nn.Linear(1, self.d_inner, bias=False)
        
        self.physics_scale = nn.Parameter(torch.tensor(0.01))
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.norm = nn.LayerNorm(d_model) 
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x, diff_coeff_flat):
        B, L, _ = x.shape
        
        residual = x
        x = self.norm(x) 
        
        x_and_z = self.in_proj(x).transpose(1, 2)
        x, z = x_and_z.chunk(2, dim=1)
        
        x = self.conv1d(x)[:, :, :L]
        x = self.act(x)
        
        x_dbl = self.x_proj(x.transpose(1, 2))
        dt_rank, B_param, C_param = torch.split(x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        dt_content = self.dt_proj(dt_rank)
        dt_physics = self.physics_bias_proj(diff_coeff_flat)
        
        dt_sum = dt_content + torch.tanh(dt_physics) * self.physics_scale
        
        dt = F.softplus(dt_sum.float()) 
        dt = torch.clamp(dt, min=1e-4, max=0.1)

        if selective_scan_fn is not None:
            A_param = -torch.exp(self.A_log.float()) 
            y = selective_scan_fn(
                x, dt.transpose(1, 2), A_param, B_param.transpose(1, 2), C_param.transpose(1, 2),
                self.D.float(), z=None, delta_bias=None, delta_softplus=False, return_last_state=False,
            )
        else:
            raise ImportError("Please install mamba_ssm")

        y = y * self.act(z)
        out = self.out_proj(y.transpose(1, 2))
        return out + residual

class ResidualPIDGroup(nn.Module):
    def __init__(self, dim, depth, d_state=32):
        super().__init__()
        self.blocks = nn.ModuleList([
            PhysicsMambaBlock(d_model=dim, d_state=d_state) for _ in range(depth)
        ])
        self.conv_post = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x_flat, diff_flat, H, W):
        residual = x_flat
        for blk in self.blocks:
            x_flat = blk(x_flat, diff_flat)
        x_2d = rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)
        x_2d = self.conv_post(x_2d)
        x_flat_out = rearrange(x_2d, 'b c h w -> b (h w) c')
        return x_flat_out + residual

class PIDMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, depth=4, group_num=4):
        super().__init__()
        self.edge_net = EdgeGradientExtractor(in_channels)
        
        self.thermal_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layers = nn.ModuleList([
            ResidualPIDGroup(dim=dim, depth=depth) for _ in range(group_num)
        ])
        
        self.final_norm = nn.LayerNorm(dim) 
        self.final_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        
    def forward(self, thermal_lr, rgb_hr):
        H, W = thermal_lr.shape[2:]
        rgb_lr = F.interpolate(rgb_hr, size=(H, W), mode='bilinear')
        
        diff_coeff, edge_map = self.edge_net(rgb_lr)
        
        x = self.thermal_embed(thermal_lr)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        diff_flat = rearrange(diff_coeff, 'b c h w -> b (h w) c')
        
        for layer in self.layers:
            x_flat = layer(x_flat, diff_flat, H, W)
            
        x_flat = self.final_norm(x_flat)
        x = rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.final_conv(x)
        return x, edge_map, diff_coeff

class SpatialPivotAttention(nn.Module):
    def __init__(self, dim, num_heads=8, topk_ratio=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_ratio = topk_ratio

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_thermal, x_rgb, edge_map_flat):
        B, N, C = x_thermal.shape
        
        num_pivots = max(int(N * self.topk_ratio), 1)
        scores = edge_map_flat.squeeze(-1) 
        _, pivot_indices = torch.topk(scores, num_pivots, dim=1) 
        
        batch_indices = torch.arange(B, device=x_thermal.device).unsqueeze(1)
        k_pivot = x_rgb[batch_indices, pivot_indices, :] 
        v_pivot = x_rgb[batch_indices, pivot_indices, :] 

        q = self.q_proj(x_thermal).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        k = self.k_proj(k_pivot).reshape(B, num_pivots, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        v = self.v_proj(v_pivot).reshape(B, num_pivots, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.norm(x + x_thermal)

class StructuralInpaintingBlock(nn.Module):
    def __init__(self, dim, mask_ratio=0.25):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.pivot_attn = SpatialPivotAttention(dim, topk_ratio=0.2)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 8),
            nn.GELU(),
            nn.Linear(dim * 8, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x_thermal, x_rgb, edge_map_flat):
        H, W = x_thermal.shape[2:]
        x_t_flat = rearrange(x_thermal, 'b c h w -> b (h w) c')
        x_r_flat = rearrange(x_rgb, 'b c h w -> b (h w) c')
        edge_flat = rearrange(edge_map_flat, 'b c h w -> b (h w) c')
        
        B, N, C = x_t_flat.shape

        mask_prob = edge_flat.squeeze(-1) + torch.rand_like(edge_flat.squeeze(-1)) * 0.5
        num_mask = int(N * self.mask_ratio)
        _, mask_indices = torch.topk(mask_prob, num_mask, dim=1)
        
        mask = torch.zeros(B, N, dtype=torch.bool, device=x_thermal.device)
        mask.scatter_(1, mask_indices, True)
        
        mask_tokens = self.mask_token.expand(B, N, -1)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x_t_flat)

        attended = self.pivot_attn(x_masked, x_r_flat, edge_flat)
        out = attended + self.ffn(self.norm(attended))
        
        return rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

class ChannelConsistencyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(1, dim, 1))
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_conv = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, thermal, rgb):
            B, C, H, W = thermal.shape
            N = H * W
            q = self.q_conv(thermal).view(B, C, N)
            k = self.k_conv(rgb).view(B, C, N)
            v = self.v_conv(rgb).view(B, C, N)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) 
            attn = attn * self.temperature.view(1, -1, 1) 
            attn = attn.softmax(dim=-1) 
            out = (attn @ v).view(B, C, H, W)
            out = self.out_conv(out)
            return out

class ModalityConsistencyGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.r_proj = nn.Conv2d(dim, dim, 1, bias=False)
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, thermal, rgb):
        t_aligned = self.t_proj(thermal)
        r_aligned = self.r_proj(rgb)
        
        t_norm = F.normalize(t_aligned, dim=1, eps=1e-6)
        r_norm = F.normalize(r_aligned, dim=1, eps=1e-6)
        
        similarity = torch.sum(t_norm * r_norm, dim=1, keepdim=True)
        discrepancy = torch.clamp(1.0 - similarity, min=0.0, max=2.0) 
        
        gate = self.gate_conv(discrepancy)
        return gate

class UncertaintyAwareRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.consistency_gate = ModalityConsistencyGate(dim)
        
        self.manifold_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
        )
        self.retrieval_module = ChannelConsistencyAttention(dim)
        self.final_fusion = nn.Conv2d(dim, dim, 1)

    def forward(self, thermal_feat, rgb_feat):
        gate = self.consistency_gate(thermal_feat, rgb_feat)
        feat_manifold = self.manifold_branch(thermal_feat)
        feat_retrieved = self.retrieval_module(thermal_feat, rgb_feat)
        refined = (1 - gate) * feat_manifold + gate * feat_retrieved
        return self.final_fusion(refined)

class SuperResolutionHead(nn.Module):
    def __init__(self, dim, out_channels=3, scale_factor=4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
    
    def forward(self, x):
        return self.upsample(x)

class RGBEmbeding(nn.Module):
    def __init__(self, rgb_c, dim, scale_factor=8): # 注意传入 scale_factor
        super().__init__()
        
        # 计算 Unshuffle 后的通道数
        # 如果 scale=8, input=3, 则 unshuffle_c = 3 * 8 * 8 = 192
        self.unshuffle_dim = rgb_c * (scale_factor ** 2)
        
        # 1. 空间换通道 (无参数，零 FLOPs)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=scale_factor)
        
        # 2. 降维/特征映射 (在 LR 尺寸上进行卷积，FLOPs 极低)
        # 将 192 通道映射到 dim (256)
        self.stem = nn.Sequential(
            nn.Conv2d(self.unshuffle_dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. 深度特征提取 (在 LR 尺寸上进行，效率高)
        # 现在这里跑 dim=256 的 ResBlock 非常快
        # FLOPs: 64*64*256*256*9 = 2.4 G (之前是 154 G)
        self.deep_extraction = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim) # 可以加深到 5-6 层都没问题
        )
        
        # 4. Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_hr, target_size=None):
        # x_hr: [B, 3, H*8, W*8]
        
        # Step 1: Pixel Unshuffle -> [B, 192, H, W]
        x = self.pixel_unshuffle(x_hr)
        
        # Step 2: Stem Conv (on LR grid)
        x = self.stem(x)
        
        # Step 3: Deep Extraction
        x = self.deep_extraction(x)
        
        # Step 4: CA
        scale = self.ca(x)
        return x * scale

class PCM2Net(nn.Module):
    def __init__(self, thermal_c=3, rgb_c=3, dim=96, scale_factor=4): 
        super().__init__()
        
        self.pid_encoder = PIDMambaEncoder(in_channels=thermal_c, dim=dim, depth=6, group_num=4)
        
        self.rgb_embed = RGBEmbeding(rgb_c=rgb_c, dim=dim, scale_factor=scale_factor)
        
        self.structural_inpainter = nn.ModuleList([ 
            StructuralInpaintingBlock(dim=dim, mask_ratio=0.25),
            StructuralInpaintingBlock(dim=dim, mask_ratio=0.15), 
            StructuralInpaintingBlock(dim=dim, mask_ratio=0.05), 
        ])
        
        self.ca_ssr = UncertaintyAwareRefinement(dim=dim)
        self.reduce_dim = nn.Conv2d(dim, dim, 1) 
        
        self.sr_head = SuperResolutionHead(
            dim=dim,
            out_channels=thermal_c,
            scale_factor=scale_factor,
        )

    def forward(self, thermal_lr, rgb_hr):
        # Step 1: 物理感知特征提取
        thermal_feat, edge_map, _ = self.pid_encoder(thermal_lr, rgb_hr)
        
        # Step 2: 准备 RGB 特征 
        # [Modified] 传入 HR 图像和目标尺寸，由 RGBEmbeding 内部进行深度提取和安全下采样
        target_size = thermal_lr.shape[2:]
        rgb_feat = self.rgb_embed(rgb_hr, target_size)
        
        # Step 3: 结构引导流形修复 
        reconstructed_feat = thermal_feat
        for layer in self.structural_inpainter:
            reconstructed_feat = layer(reconstructed_feat, rgb_feat, edge_map)
        
        # Step 4: 模态一致性细化 
        refined_feat = self.ca_ssr(reconstructed_feat, rgb_feat)
        
        # Step 5: 上采样重建
        return self.sr_head(refined_feat + thermal_feat)

# ==============================================================================
# Main: 测试、参数统计与 FLOPs 计算
# ==============================================================================
if __name__ == '__main__':
    def format_params(num):
        if num < 1e3: return f"{num}"
        elif num < 1e6: return f"{num / 1e3:.2f} K"
        else: return f"{num / 1e6:.2f} M"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # 模拟输入尺寸 (计算FLOPs通常基于 64x64 或 128x128 的 LR 输入)
    TEST_H, TEST_W = 64, 64 
    SCALE = 8
    
    # 【参数控制】 base_dim 调整为 48
    # 维度变化: 48 -> 96 -> 192 -> 384
    try:
        model = PCM2Net(thermal_c=3, rgb_c=3, dim=256, scale_factor=SCALE).to(device)
        print("Model built successfully.")
    except Exception as e:
        print(f"Model build failed: {e}")
        exit()

    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 50)
    print(f"Total Parameters: {format_params(total_params)}")

    t_lr = torch.randn(1, 3, TEST_H, TEST_W).to(device)
    rgb_hr = torch.randn(1, 3, TEST_H * SCALE, TEST_W * SCALE).to(device)
    
    # 2. 计算 FLOPs (使用 thop)
    try:
        from thop import profile
        print("-" * 50)
        print(f"Calculating FLOPs for Input LR: {TEST_H}x{TEST_W} ...")
        
        # 注意: thop 可能无法追踪自定义的 mamba CUDA kernel，
        # 但它会统计 Linear 和 Conv 层，这通常占了 FLOPs 的绝大部分。
        flops, params = profile(model, inputs=(t_lr, rgb_hr), verbose=False)
        
        print(f"FLOPs: {flops / 1e9:.4f} G")
        print(f"Params (thop): {format_params(params)}")
    except ImportError:
        print("-" * 50)
        print("Warning: 'thop' not found. Install via 'pip install thop' to calculate FLOPs.")
    except Exception as e:
        print(f"FLOPs calculation error: {e}")

    # 3. 前向推理测试
    print("-" * 50)
    try:
        with torch.no_grad():
            out = model(t_lr, rgb_hr)
        print(f"Output Shape: {out.shape}")
        expected_shape = (1, 3, TEST_H * SCALE, TEST_W * SCALE)
        assert out.shape == expected_shape
        print("Verification: PASSED")
    except Exception as e:
        print(f"Forward error: {e}")
        import traceback
        traceback.print_exc()