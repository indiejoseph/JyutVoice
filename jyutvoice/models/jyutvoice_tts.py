import os
import math
import datetime as dt
import random
import torch
from torch.nn import functional as F
from jyutvoice.utils.model import (
    sequence_mask,
    generate_path,
    duration_loss,
)
from jyutvoice.utils.mask import make_pad_mask
import jyutvoice.utils.monotonic_align as monotonic_align
from jyutvoice.models.baselightningmodule import BaseLightningClass
from jyutvoice.models.reference_encoder import MelStyleEncoder


class JyutVoiceTTS(BaseLightningClass):
    def __init__(
        self,
        encoder,
        ldpm,
        style_encoder,
        decoder,
        output_size=80,
        spk_embed_dim=192,
        gin_channels=256,
        freeze_encoder=False,
        freeze_decoder=False,
        use_precomputed_durations=False,
        optimizer=None,
        scheduler=None,
        pretrain_path=None,
        warmup_steps=100,
        noise_scheduler=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.decoder = decoder
        self.ldpm = ldpm
        self.noise_scheduler = noise_scheduler

        self.use_precomputed_durations = use_precomputed_durations
        self.n_feats = encoder.n_feats
        self.output_size = output_size

        # speaker projection -> mel dimension
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)

        # style encoder to extract global prosody vector
        self.style_encoder = style_encoder

        # linear map from style vector -> LDPM global conditioning dim
        self.style_to_global = torch.nn.Linear(gin_channels, 256)

        if freeze_encoder:
            self._freeze_encoder()
        if freeze_decoder:
            self._freeze_decoder()
        if pretrain_path:
            self.load_pretrain(pretrain_path)

    # ------------------------------
    # Freeze helpers
    # ------------------------------
    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def _freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.spk_embed_affine_layer.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.spk_embed_affine_layer.eval()

    # ------------------------------
    # Pretrain loading
    # ------------------------------
    def load_pretrain(self, pretrain_path):
        """
        Load pretrained weights from a checkpoint file.

        This method loads weights for transfer learning. It supports:
        1. Loading full model state_dict (encoder + decoder + speaker embedding layer)
        2. Partial loading with strict=False to handle missing keys gracefully
        3. Logging of loaded and skipped weights for debugging

        Args:
            pretrain_path (str): Path to the pretrained checkpoint file (.pt)

        Example:
            >>> model = JyutVoiceTTS(...)
            >>> model.load_pretrain('pretrained_models/pretrain.pt')
        """
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_path}")

        # Load the checkpoint
        checkpoint = torch.load(pretrain_path, map_location="cpu")

        # Handle both full checkpoints and state_dict-only checkpoints
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load the state dict with strict=False to allow for some missing keys
        # (e.g., keys that might be specific to training setup)
        incompatible_keys = self.load_state_dict(state_dict, strict=False)

        return incompatible_keys

    # ------------------------------
    # Inference
    # ------------------------------
    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_lengths,
        lang,
        tone,
        word_pos,
        syllable_pos,
        spk_embed,
        prompt_feat,
        prompt_h=None,
        n_timesteps=10,
        temperature=1.0,
        length_scale=1.0,
        use_ldpm=True,
        ldpm_steps=50,
        ldpm_temp=1.0,
    ):
        t0 = dt.datetime.now()

        # speaker & style conditioning
        spk_embed = F.normalize(spk_embed, dim=1)
        spk_embed = self.spk_embed_affine_layer(spk_embed)

        # extract global prosody vector g from prompt mel
        style_vec = self.style_encoder(prompt_feat.transpose(1, 2), None)
        g = self.style_to_global(style_vec)  # (B, 256)
        g = F.normalize(g, dim=1)

        # text encoder -> phoneme-level μ_x, logw
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed
        )

        # durations -> attention
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # align to mel time
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        ).transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max]

        # ---------------------------------------------------
        # Apply LDPM refinement (global prosody conditioning)
        # ---------------------------------------------------
        if use_ldpm and self.ldpm is not None and self.noise_scheduler is not None:
            mu_y = self.ldpm.sample(
                cond_local=mu_y,
                cond_global=g,
                scheduler=self.noise_scheduler,
                n_steps=ldpm_steps,
                noise_scale=ldpm_temp,
            )

        # prepare prompt/conditioning mel for flow decoder
        batch_size = x.size(0)
        if batch_size != 1:
            raise ValueError("synthesise() requires batch_size=1")

        if prompt_feat is not None and prompt_h is not None:
            mu_y = torch.cat([prompt_h.transpose(1, 2), mu_y], dim=2)
            mel_len1, mel_len2 = (
                prompt_feat.shape[1],
                mu_y.shape[2] - prompt_feat.shape[1],
            )
            conds = torch.zeros(
                [1, mel_len1 + mel_len2, self.output_size],
                device=x.device,
                dtype=mu_y.dtype,
            )
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)
            mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(conds.dtype)
        else:
            mel_len1 = 0
            conds = torch.zeros_like(mu_y)
            mask = (~make_pad_mask(y_lengths)).to(mu_y.dtype)

        # flow decoder synthesis
        decoder_outputs, _ = self.decoder(
            mu=mu_y,
            mask=mask.unsqueeze(1),
            spks=spk_embed,
            cond=conds,
            n_timesteps=n_timesteps,
            temperature=temperature,
            streaming=False,
        )
        decoder_outputs = decoder_outputs[:, :, mel_len1:]

        t1 = (dt.datetime.now() - t0).total_seconds()
        rtf = t1 * 24000 / (decoder_outputs.shape[-1] * 480)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn,
            "mel": decoder_outputs,
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    # ------------------------------
    # Training
    # ------------------------------
    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        lang,
        tone,
        word_pos,
        syllable_pos,
        spk_embed,
        decoder_h,
        z,
        z_lengths,
        durations=None,
    ):
        # speaker embedding
        spk_embed = F.normalize(spk_embed, dim=1)
        spk_embed = self.spk_embed_affine_layer(spk_embed)

        # style / global prosody vector
        ref_mask = sequence_mask(z_lengths, z.shape[-1]).unsqueeze(1).to(z.dtype)
        style_vec = self.style_encoder(z, ref_mask)
        g = self.style_to_global(style_vec)  # (B, 256)
        g = F.normalize(g, dim=1)

        # text encoder: phoneme-level features
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed
        )
        y_max = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # alignment
        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones_like(mu_x)
                h = decoder_h.transpose(1, 2)
                h_sq = torch.matmul(factor.transpose(1, 2), h**2)
                h_mu = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), h)
                mu_sq = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = h_sq - h_mu + mu_sq + const
                attn = monotonic_align.maximum_path(
                    log_prior, attn_mask.squeeze(1)
                ).detach()

        # duration loss
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # align to mel latent
        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)

        # ------------------------------------------------
        # LDPM latent diffusion ε-pred loss (global prosody)
        # ------------------------------------------------
        ldpm_loss = torch.tensor(0.0, device=x.device)
        if self.ldpm is not None and self.noise_scheduler is not None:
            B = mu_y.size(0)
            T_noise = self.noise_scheduler.T
            t = torch.randint(1, T_noise, (B,), device=x.device)
            z_t, eps = self.noise_scheduler.sample_noisy(decoder_h.transpose(1, 2), t)
            eps_pred = self.ldpm(z_t, t, cond_local=mu_y, cond_global=g)
            ldpm_loss = F.mse_loss(eps_pred, eps)

        # --------------------------------
        # flow matching loss (frozen decoder)
        # --------------------------------
        conds = torch.zeros_like(y)
        for i, j in enumerate(y_lengths):
            if random.random() < 0.5:
                continue
            idx = random.randint(0, int(0.3 * j))
            conds[i, :, :idx] = y[i, :, :idx]

        diff_loss, _ = self.decoder.compute_loss(
            x1=y,
            mask=y_mask,
            mu=mu_y,
            spks=spk_embed,
            cond=conds,
            streaming=(random.random() < 0.5),
        )

        # prior loss (match flow encoder hidden)
        decoder_h_mask = (
            sequence_mask(y_lengths, decoder_h.shape[1]).unsqueeze(1).to(mu_y.dtype)
        )
        mu_y_t = mu_y.transpose(1, 2)
        prior_loss = torch.sum(
            0.5
            * ((decoder_h - mu_y_t) ** 2 + math.log(2 * math.pi))
            * decoder_h_mask.squeeze(1).unsqueeze(-1)
        )
        prior_loss = prior_loss / (torch.sum(decoder_h_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss, ldpm_loss, attn
