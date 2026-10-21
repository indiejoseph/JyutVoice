import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jyutvoice.models.flow_duration_predictor import SinusoidalPosEmb, ConvNeXt1DBlock


MARK_CLASSES = 6  # 0 no punctuation ahead, 1 comma, 2 full stop, 3 question mark, 4 exclamation mark, 5 other (… ' -)
PUNCT_CLASS = {",": 1, ".": 2, "?": 3, "!": 4, "…": 5, "'": 5, "-": 5}
MARK_MAX_DIST = 250  # frames (5 s): farther than this (or no mark ahead) counts as 250


def mark_class_table(symbols) -> torch.Tensor:
    """(n_symbols,) long: the punctuation class (see MARK_CLASSES) of every symbol id, 0 for everything that is not punctuation."""
    return torch.tensor([PUNCT_CLASS.get(s, 0) for s in symbols], dtype=torch.long)


def mark_features(x: torch.Tensor, attn: torch.Tensor, class_of_id: torch.Tensor):
    """What punctuation is coming up, for every frame: the class of the next punctuation token at or after the frame's own token, and the frames until that token starts.

    x: (B, T_text) symbol ids (blanks included, as fed to the encoder); attn: (B, [1,] T_text, T_mel) 0/1 alignment; class_of_id: from `mark_class_table`.
    Returns mark_y (B, T_mel) long (0: none ahead) and dist_y (B, 1, T_mel) float frames (inside the mark's own frames 0, capped at MARK_MAX_DIST).
    """
    a = attn.squeeze(1) if attn.dim() == 4 else attn
    B, T, Ty = a.shape
    cls = class_of_id.to(x.device)[x]  # (B, T_text)
    idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
    cand = torch.where(cls > 0, idx, torch.full_like(idx, T))
    nxt = torch.flip(torch.cummin(torch.flip(cand, [1]), dim=1).values, [1])  # index of the next mark token (T: none)
    has = nxt < T
    nxt_c = nxt.clamp(max=T - 1)
    nxt_cls = torch.where(has, cls.gather(1, nxt_c), torch.zeros_like(cls))
    dur = a.sum(-1)
    start = dur.cumsum(-1) - dur  # first frame of every token
    start_next = start.gather(1, nxt_c)
    tok = a.argmax(1)  # (B, T_mel): the token of every frame
    frames = torch.arange(Ty, device=x.device).unsqueeze(0)
    dist = (start_next.gather(1, tok) - frames).clamp(0, MARK_MAX_DIST).float()
    dist = torch.where(has.gather(1, tok), dist, torch.full_like(dist, float(MARK_MAX_DIST)))
    return nxt_cls.gather(1, tok), dist.unsqueeze(1)


class ProsodyCFM(nn.Module):
    """
    Frame-Level Conditional Flow Matching Prosody Generator at 50 Hz.
    Generates dynamic `decoder_h` trajectories from expanded phoneme + BERT + intra-phone fractional positions.
    """

    def __init__(
        self,
        in_channels: int = 80,
        text_channels: int = 192,
        hidden_channels: int = 256,
        n_blocks: int = 6,
        kernel_size: int = 7,
        dilations=(1, 2, 4, 1, 2, 4),
        p_dropout: float = 0.1,
        spk_embed_dim: int = 192,
        sigma_min: float = 1e-4,
        cfg_dropout: float = 0.15,
        residual: bool = False,
        use_mark: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.residual = residual  # True: the flow generates decoder_h - mu_y (the detail the encoder's average misses) and `forward` adds mu_y back
        self.hidden_channels = hidden_channels
        self.sigma_min = sigma_min
        self.cfg_dropout = cfg_dropout

        # Target decoder_h projection and condition projections
        self.target_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.mu_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.text_proj = nn.Conv1d(text_channels, hidden_channels, 1)

        # Intra-phone fractional position and duration embeddings
        self.tau_emb = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dur_emb = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.use_mark = use_mark  # True: every frame also gets the class of the next punctuation mark and the frames until it (`mark_features`)
        if use_mark:
            self.mark_emb = nn.Embedding(MARK_CLASSES, hidden_channels)
            self.dist_emb = nn.Sequential(
                SinusoidalPosEmb(64),
                nn.Linear(64, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
            )

        if spk_embed_dim > 0:
            self.spk_proj = nn.Linear(spk_embed_dim, hidden_channels)
        else:
            self.spk_proj = None

        # Time step embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # 50 Hz frame-level ConvNeXt backbone
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            self.blocks.append(
                ConvNeXt1DBlock(
                    dim=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    mlp_ratio=4,
                    p_dropout=p_dropout,
                )
            )

        # Velocity output projection
        self.out_proj = nn.Sequential(
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, 1),
        )
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    @staticmethod
    def compute_fractional_pos(attn: torch.Tensor):
        """
        attn: (B, 1, T_text, T_mel) or (B, T_text, T_mel) binary alignment matrix.
        Returns:
            tau: (B, 1, T_mel) fractional position within phone in [0, 1)
            dur: (B, 1, T_mel) total duration of phone in frames
        """
        if attn.dim() == 4:
            a = attn.squeeze(1)  # (B, T_text, T_mel)
        else:
            a = attn

        # Cumulative frame count per token:
        intra_tok = (torch.cumsum(a, dim=-1) - 1.0) * a  # (B, T_text, T_mel)
        intra_frame = intra_tok.sum(dim=1, keepdim=True).clamp_min(0.0)  # (B, 1, T_mel)

        # Total duration of phone for each frame:
        dur_tok = a.sum(dim=-1, keepdim=True)  # (B, T_text, 1)
        dur = torch.matmul(a.transpose(1, 2), dur_tok).transpose(1, 2).clamp_min(1.0)  # (B, 1, T_mel)

        tau = (intra_frame / dur).clamp(0.0, 1.0)
        return tau, dur

    def extract_cond(
        self,
        mu_y: torch.Tensor,
        h_text_y: torch.Tensor,
        tau: torch.Tensor,
        dur: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor = None,
        mark_y: torch.Tensor = None,
        dist_y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        mu_y: (B, 80, T_mel)
        h_text_y: (B, C, T_mel)
        tau: (B, 1, T_mel)
        dur: (B, 1, T_mel)
        y_mask: (B, 1, T_mel)
        g: (B, spk_dim)
        """
        B, _, T = mu_y.shape

        cond = self.mu_proj(mu_y) + self.text_proj(h_text_y)

        # Intra-phone sinusoidal features
        tau_flat = tau.squeeze(1).reshape(-1)  # (B * T,)
        dur_log = torch.log(dur.squeeze(1).reshape(-1).clamp_min(1.0))  # (B * T,)

        tau_features = self.tau_emb(tau_flat).view(B, T, -1).transpose(1, 2)
        dur_features = self.dur_emb(dur_log).view(B, T, -1).transpose(1, 2)

        cond = cond + tau_features + dur_features

        if self.use_mark:
            if mark_y is None or dist_y is None:
                raise ValueError("this ProsodyCFM uses punctuation (use_mark=True): pass mark_y and dist_y from mark_features()")
            cond = cond + self.mark_emb(mark_y).transpose(1, 2)
            dist_log = torch.log1p(dist_y.squeeze(1).reshape(-1)) * 20.0  # frames -> a range the sinusoids resolve
            cond = cond + self.dist_emb(dist_log).view(B, T, -1).transpose(1, 2)

        if g is not None and self.spk_proj is not None:
            s = self.spk_proj(g).unsqueeze(-1)
            cond = cond + s

        cond = cond * y_mask
        for block in self.blocks:
            cond = block(cond, y_mask)

        return cond

    def forward_velocity(
        self,
        h_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_in = self.target_proj(h_t)
        t_emb = self.time_emb(t).unsqueeze(-1)  # (B, C, 1)
        h = (x_in + cond + t_emb) * y_mask
        v = self.out_proj(h) * y_mask
        return v

    def compute_loss(
        self,
        decoder_h: torch.Tensor,
        mu_y: torch.Tensor,
        h_text_y: torch.Tensor,
        attn: torch.Tensor,
        y_mask: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor = None,
        prior_weight: torch.Tensor = None,
        mark_y: torch.Tensor = None,
        dist_y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Optimal-Transport Flow Matching Loss matching generated frames to true decoder_h.
        decoder_h: (B, 80, T_mel)
        mu_y: (B, 80, T_mel)
        h_text_y: (B, C, T_mel)
        """
        B, C, T = decoder_h.shape
        device = decoder_h.device

        tau, dur = self.compute_fractional_pos(attn)
        cond = self.extract_cond(mu_y, h_text_y, tau, dur, y_mask, g=g, mark_y=mark_y, dist_y=dist_y)

        # CFG dropout
        if self.cfg_dropout > 0.0 and self.training:
            cfg_mask = (torch.rand(B, 1, 1, device=device) >= self.cfg_dropout).float()
            cond = cond * cfg_mask

        # Time t ~ U(0, 1)
        t = torch.rand(B, device=device)
        t_expand = t.view(B, 1, 1)

        # Prior noise: standard normal (or initialized around mu_y)
        h_0 = torch.randn_like(decoder_h) * y_mask
        h_1 = (decoder_h - mu_y if self.residual else decoder_h) * y_mask

        h_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * h_0 + t_expand * h_1
        u = (h_1 - (1.0 - self.sigma_min) * h_0) * y_mask

        pred_v = self.forward_velocity(h_t, t, cond, y_mask)

        # Loss with optional sample prior_weight
        sample_mask = y_mask if prior_weight is None else y_mask * prior_weight.view(-1, 1, 1).to(y_mask.dtype)
        diff = (pred_v - u) * sample_mask
        loss = torch.sum(diff ** 2) / (torch.sum(sample_mask).clamp_min(1.0) * C)
        return loss

    @torch.inference_mode()
    def forward(
        self,
        mu_y: torch.Tensor,
        h_text_y: torch.Tensor,
        attn: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor = None,
        n_steps: int = 10,
        cfg_scale: float = 0.0,
        temperature: float = 0.5,
        mark_y: torch.Tensor = None,
        dist_y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Reverse Euler flow integration to generate dynamic 50 Hz decoder_h.

        The defaults are the settings chosen by listening and scoring (noise temperature 0.5, no guidance, 10 Euler steps): guidance above 0 inflates the
        movement of the features (std ratio 1.13 at 0.5), temperature 1.0 adds pronunciation errors for no gain in the pitch statistics.
        """
        B, C, T = mu_y.shape
        device = mu_y.device

        tau, dur = self.compute_fractional_pos(attn)
        cond = self.extract_cond(mu_y, h_text_y, tau, dur, y_mask, g=g, mark_y=mark_y, dist_y=dist_y)

        if cfg_scale > 0.0:
            cond_null = torch.zeros_like(cond)
        else:
            cond_null = None

        h = torch.randn(B, self.in_channels, T, device=device) * temperature * y_mask

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)

            v_cond = self.forward_velocity(h, t, cond, y_mask)
            if cfg_scale > 0.0:
                v_null = self.forward_velocity(h, t, cond_null, y_mask)
                v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_null
            else:
                v = v_cond

            h = h + dt * v

        if self.residual:
            h = h + mu_y
        return h * y_mask
