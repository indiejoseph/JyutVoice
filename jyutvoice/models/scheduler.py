import torch
import torch.nn as nn


class LinearNoiseSchedule(nn.Module):
    """
    Linear beta schedule for diffusion models.
    Keeps buffers device/dtype agnostic and automatically
    casts to match input tensors during forward calls.
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        dtype=torch.float32,
    ):
        super().__init__()
        self.T = num_train_timesteps

        # Register as buffer so Lightning/DDP moves automatically
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=dtype)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

    @torch.no_grad()
    def sample_noisy(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion q(z_t | x0).
        Automatically matches device and dtype of x0.

        Args:
            x0: (B, C, T)
            t: (B,) integer steps
        Returns:
            z_t: (B, C, T)
            eps: (B, C, T)
        """
        # Match dtype/device dynamically
        a = self.alpha_bar.index_select(0, t.to(self.alpha_bar.device))
        a = a.view(-1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        eps = torch.randn_like(x0)
        z_t = a.sqrt() * x0 + (1 - a).clamp_min(0).sqrt() * eps
        return z_t, eps

    @torch.no_grad()
    def step_reverse(self, z_t: torch.Tensor, eps_pred: torch.Tensor, t_int: int):
        """
        Reverse diffusion step.
        Matches dtype/device of z_t automatically.
        """
        device, dtype = z_t.device, z_t.dtype
        a_t = self.alpha_bar[t_int].to(device=device, dtype=dtype)
        a_prev = (
            self.alpha_bar[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=device)
        ).to(dtype=dtype)
        beta_t = 1 - (a_t / a_prev)
        mean = (z_t - beta_t / (1 - a_t).sqrt() * eps_pred) / a_t.sqrt()

        if t_int > 0:
            noise = torch.randn_like(z_t)
            z_prev = mean + beta_t.sqrt() * noise
        else:
            z_prev = mean
        return z_prev
