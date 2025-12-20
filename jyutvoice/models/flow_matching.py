import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from jyutvoice.models.estimator import Decoder


# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(
        self,
        noise_channels,
        cond_channels,
        hidden_channels,
        out_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels,
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4

        self.estimator = Decoder(
            noise_channels,
            cond_channels,
            hidden_channels,
            out_channels,
            filter_channels,
            p_dropout,
            n_layers,
            n_heads,
            kernel_size,
            gin_channels,
        )

    @torch.inference_mode()
    def forward(
        self,
        sfm_params,
        mask,
        n_timesteps,
        temperature=1.0,
        alpha=1.0,
        c=None,
        solver=None,
        cfg_kwargs=None,
    ):
        xp, tp, logsigma_p = sfm_params
        x0 = torch.randn_like(xp) * temperature
        sigma_p = torch.exp(logsigma_p)

        Delta = torch.clamp(
            alpha * (torch.sqrt(sigma_p) + (1 - self.sigma_min) * tp), min=1.0
        )
        tp = alpha * tp / Delta
        sigma_p = alpha**2 * sigma_p / Delta**2
        xp = alpha * xp / Delta

        x = (
            torch.sqrt(
                torch.clamp((1 - (1 - self.sigma_min) * tp) ** 2 - sigma_p, min=0.0)
            )
            * x0
            + xp
        )
        t_span = torch.linspace(tp.item(), 1, n_timesteps + 1, device=xp.device)

        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, c=c)
        else:
            estimator = functools.partial(
                self.cfg_wrapper, mask=mask, c=c, cfg_kwargs=cfg_kwargs
            )

        trajectory = odeint(estimator, x, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1], tp.item(), torch.sqrt(sigma_p).item()

    # cfg inference
    def cfg_wrapper(self, t, x, mask, c, cfg_kwargs):
        fake_speaker = cfg_kwargs["fake_speaker"].repeat(x.size(0), 1)
        cfg_strength = cfg_kwargs["cfg_strength"]

        cond_output = self.estimator(t, x, mask, c)
        uncond_output = self.estimator(t, x, mask, fake_speaker)

        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, sfm_params, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape
        xp, tp, logsigma_p = sfm_params

        th = torch.sum(xp.detach() * x1, dim=1) / (
            torch.sum(x1 * x1, dim=1) + 1e-8
        )  # [b, seq]
        th = torch.sum(th, dim=1) / torch.sum(mask.squeeze(1), dim=1)
        th = th[:, None, None]
        sigma_h = F.mse_loss(xp.detach(), th * x1, reduction="none")
        sigma_h = torch.sum(sigma_h, dim=[1, 2], keepdim=True) / (
            torch.sum(mask, dim=[1, 2], keepdim=True) * x1.shape[1]
        )

        Delta = torch.clamp(torch.sqrt(sigma_h) + (1 - self.sigma_min) * th, min=1.0)
        th = th / Delta
        sigma_h = sigma_h / Delta**2
        xp = xp / Delta

        mu_loss = F.mse_loss(xp, th * x1, reduction="none")
        mu_loss = torch.sum(mu_loss) / (torch.sum(mask) * x1.shape[1])

        t_loss = F.mse_loss(tp, th, reduction="mean")
        sigma_loss = F.mse_loss(logsigma_p, torch.log(sigma_h + 1e-8), reduction="mean")

        # random timestep
        # use cosine timestep scheduler from cosyvoice: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        # sample noise p(x_0)
        x0 = torch.randn_like(x1)

        xp = (
            torch.sqrt(
                torch.clamp((1 - (1 - self.sigma_min) * th) ** 2 - sigma_h, min=0.0)
            )
            * x0
            + xp
        )
        xt = (1 - t) * xp + t * (x1 + self.sigma_min * x0)
        ut = (x1 + self.sigma_min * x0 - xp) / (1 - th)

        xt = xt * mask
        ut = ut * mask
        t = (1 - th) * t + th

        vt = self.estimator(t.squeeze(), xt, mask, c)
        fm_loss = F.mse_loss(vt, ut, reduction="sum") / (torch.sum(mask) * ut.size(1))
        return {
            "fm_loss": fm_loss,
            "mu_loss": mu_loss,
            "t_loss": t_loss,
            "sigma_loss": sigma_loss,
        }, {
            "sigma_h": torch.mean(sigma_h),
            "sigma_p": torch.mean(torch.exp(logsigma_p)),
            "th": torch.mean(th),
            "tp": torch.mean(tp),
            "Delta": torch.mean(Delta),
        }
