import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiffusionProsodyModel(nn.Module):
    def __init__(self, n_feats=80, hidden_dim=256, n_layers=4, global_dim=256):
        super().__init__()
        self.n_feats = n_feats
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Conv1d(n_feats, hidden_dim, 1)

        # Global FiLM (scale/shift) from utterance vector g -> (γ, β)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(n_layers)
            ]
        )
        self.fuse_proj = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.output_proj = nn.Conv1d(hidden_dim, n_feats, 1)

    def _apply_global(self, h, g_embed):
        # g_embed: (B, 2*H) -> split scale/shift, broadcast to time
        B, H, T = h.shape
        gamma, beta = torch.split(g_embed, H, dim=-1)  # (B,H), (B,H)
        gamma = gamma.unsqueeze(-1)  # (B,H,1)
        beta = beta.unsqueeze(-1)  # (B,H,1)
        return h * (1 + gamma) + beta  # FiLM

    def forward(self, z_t, t, cond_local, cond_global):
        """
        z_t:         (B, n_feats, T)   noisy latent
        t:           (B,)               timesteps
        cond_local:  (B, n_feats, T)   μ_y baseline (aligned latent)
        cond_global: (B, Dg)           utterance-level style/prosody vector
        """
        # time embedding
        t_embed = self.time_embed(t[:, None].float())  # (B, H)

        # global FiLM base
        g_embed = self.global_mlp(cond_global)  # (B, 2H)

        g_embed = torch.cat([g_embed, t_embed], dim=-1)  # (B, 3H)
        g_embed = self.fuse_proj(g_embed)  # (B, 2H)

        # --- FiLM conditioning ---
        h = self.input_proj(z_t + cond_local)  # (B, H, T)
        h = self._apply_global(h, g_embed)  # uses split into γ/β

        # continue with residual blocks, etc.
        for blk in self.blocks:
            h = h + blk(h)

        return self.output_proj(h)

    @torch.inference_mode()
    def sample(self, cond_local, cond_global, scheduler, n_steps=50, noise_scale=1.0):
        self.eval()
        device = cond_local.device
        B, C, T = cond_local.shape
        z = torch.randn_like(cond_local) * noise_scale
        total_T = scheduler.T
        step_indices = torch.linspace(total_T - 1, 0, n_steps, device=device).long()
        for t in step_indices:
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            eps_pred = self.forward(z, t_batch, cond_local, cond_global)
            z = scheduler.step_reverse(z, eps_pred, int(t.item()))
        return z
