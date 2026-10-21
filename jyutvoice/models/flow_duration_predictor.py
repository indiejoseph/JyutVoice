import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvNeXt1DBlock(nn.Module):
    """1D ConvNeXt block with depthwise conv, LayerNorm, and point-wise MLP."""

    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1, mlp_ratio: int = 4, p_dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim
        )
        self.norm = nn.GroupNorm(1, dim)  # Equivalent to LayerNorm across channels
        self.pwconv1 = nn.Conv1d(dim, dim * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * mlp_ratio, dim, kernel_size=1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), mask: (B, 1, T)
        residual = x
        x = self.dwconv(x * mask)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pwconv2(x)
        return (residual + x) * mask


class FlowDurationPredictor(nn.Module):
    """
    Conditional Flow Matching Duration Predictor with BERT, speaker, and phrase-level context.
    Models the distribution of log-durations: log(w) = log(frames).
    """

    def __init__(
        self,
        in_channels: int = 192,
        filter_channels: int = 256,
        n_layers: int = 4,
        kernel_size: int = 5,
        dilations=(1, 2, 4, 8),
        p_dropout: float = 0.1,
        bert_dim: int = 1024,
        spk_embed_dim: int = 192,
        sigma_min: float = 1e-4,
        cfg_dropout: float = 0.15,
        grad_scale: float = 0.1,
        infer_steps: int = 10,
        infer_cfg_scale: float = 1.5,
        infer_temperature: float = 1.0,
    ):
        super().__init__()
        self.infer_steps, self.infer_cfg_scale, self.infer_temperature = infer_steps, infer_cfg_scale, infer_temperature  # sampling defaults
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.sigma_min = sigma_min
        self.cfg_dropout = cfg_dropout
        self.grad_scale = grad_scale

        # Feature projections
        self.x_proj = nn.Conv1d(in_channels, filter_channels, 1)

        if bert_dim > 0:
            self.bert_proj = nn.Sequential(
                nn.LayerNorm(bert_dim, elementwise_affine=False),
                nn.Linear(bert_dim, filter_channels, bias=False),
            )
        else:
            self.bert_proj = None

        if spk_embed_dim > 0:
            self.spk_proj = nn.Linear(spk_embed_dim, filter_channels)
        else:
            self.spk_proj = None

        # Time step embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(filter_channels),
            nn.Linear(filter_channels, filter_channels),
            nn.GELU(),
            nn.Linear(filter_channels, filter_channels),
        )

        # Context backbone (dilated ConvNeXt blocks for wide receptive field across sentence)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = dilations[i % len(dilations)]
            self.blocks.append(
                ConvNeXt1DBlock(
                    dim=filter_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    mlp_ratio=4,
                    p_dropout=p_dropout,
                )
            )

        # Velocity net for Flow Matching
        self.dur_in_proj = nn.Conv1d(1, filter_channels, 1)
        self.time_proj = nn.Linear(filter_channels, filter_channels)
        self.out_proj = nn.Sequential(
            nn.GroupNorm(1, filter_channels),
            nn.GELU(),
            nn.Conv1d(filter_channels, 1, 1),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def extract_cond(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        bert: torch.Tensor = None,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Fused conditioning vector: (B, filter_channels, T)
        """
        if self.grad_scale < 1.0:
            x = x * self.grad_scale + x.detach() * (1.0 - self.grad_scale)

        cond = self.x_proj(x)

        if bert is not None and self.bert_proj is not None:
            # bert: (B, T, bert_dim) -> (B, filter_channels, T)
            b = self.bert_proj(bert.to(x.dtype)).transpose(1, 2)
            cond = cond + b

        if g is not None and self.spk_proj is not None:
            # g: (B, spk_embed_dim) -> (B, filter_channels, 1)
            s = self.spk_proj(g).unsqueeze(-1)
            cond = cond + s

        cond = cond * x_mask
        for block in self.blocks:
            cond = block(cond, x_mask)

        return cond

    def forward_velocity(
        self,
        w_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts d(w_t)/dt given w_t (B, 1, T), time t (B,), and condition (B, C, T).
        """
        h_w = self.dur_in_proj(w_t)
        t_emb = self.time_emb(t)  # (B, C)
        h_t = self.time_proj(t_emb).unsqueeze(-1)  # (B, C, 1)

        h = (h_w + cond + h_t) * x_mask
        v = self.out_proj(h) * x_mask
        return v

    def compute_loss(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        target_dur: torch.Tensor,
        x_lengths: torch.Tensor,
        bert: torch.Tensor = None,
        g: torch.Tensor = None,
        token_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        OT-Flow Matching loss over log-durations (`token_weight` (B, T): per-token weight, 0 drops a token from the loss):
        target_dur: (B, 1, T) integer frame counts from MAS (>= 1)
        """
        B, _, T = x.shape
        device = x.device

        # Target in log domain
        target_logw = torch.log(target_dur.float().clamp_min(1.0)) * x_mask

        # Condition extraction
        cond = self.extract_cond(x, x_mask, bert=bert, g=g)

        # Classifier-Free Guidance dropout during training
        if self.cfg_dropout > 0.0 and self.training:
            cfg_mask = (torch.rand(B, 1, 1, device=device) >= self.cfg_dropout).float()
            cond = cond * cfg_mask

        # Sample time t ~ U(0, 1)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)

        # Standard normal noise as prior w_0
        w_0 = torch.randn_like(target_logw) * x_mask

        # Flow path: w_t = (1 - (1 - sigma_min) * t) * w_0 + t * target_logw
        w_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * w_0 + t_expand * target_logw
        # Target vector field: u = target_logw - (1 - sigma_min) * w_0
        u = (target_logw - (1.0 - self.sigma_min) * w_0) * x_mask

        # Predict vector field
        pred_v = self.forward_velocity(w_t, t, cond, x_mask)

        # Duration loss: weighted by valid token lengths
        if token_weight is None:
            return torch.sum((pred_v - u) ** 2) / torch.sum(x_lengths).clamp_min(1.0)
        w = token_weight.unsqueeze(1).to(pred_v.dtype) * x_mask
        return torch.sum(((pred_v - u) ** 2) * w) / torch.sum(w).clamp_min(1.0)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        bert: torch.Tensor = None,
        g: torch.Tensor = None,
        n_steps: int = None,
        cfg_scale: float = None,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        Samples log-durations via Euler integration:
        Returns: (B, 1, T) predicted log-durations.
        """
        B, _, T = x.shape
        device = x.device
        n_steps = self.infer_steps if n_steps is None else n_steps
        cfg_scale = self.infer_cfg_scale if cfg_scale is None else cfg_scale
        temperature = self.infer_temperature if temperature is None else temperature

        cond = self.extract_cond(x, x_mask, bert=bert, g=g)

        if cfg_scale > 0.0:
            cond_null = torch.zeros_like(cond)
        else:
            cond_null = None

        # Sample prior w_0
        w = torch.randn(B, 1, T, device=device) * temperature * x_mask

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            v_cond = self.forward_velocity(w, t, cond, x_mask)
            if cfg_scale > 0.0:
                v_null = self.forward_velocity(w, t, cond_null, x_mask)
                v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_null
            else:
                v = v_cond

            w = w + dt * v

        return w * x_mask
