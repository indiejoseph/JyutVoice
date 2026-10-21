import os
import math
import warnings
import datetime as dt
import torch
from torch.nn import functional as F
from typing import TYPE_CHECKING, Optional
from jyutvoice.utils.model import (
    sequence_mask,
    generate_path,
)
from jyutvoice.utils.mask import make_pad_mask
import jyutvoice.utils.monotonic_align as monotonic_align
from jyutvoice.models.baselightningmodule import BaseLightningClass
from jyutvoice.models.duration_predictor import DurationPredictor, duration_loss
from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
from jyutvoice.models.prosody_cfm import ProsodyCFM, mark_features
from jyutvoice.models.slp import (
    SpeechLengthPredictor,
    ce_logits_to_length,
    distribute_frames,
)

if TYPE_CHECKING:
    from jyutvoice.models.text_encoder import TextEncoder
    from jyutvoice.flow.flow_matching import CausalConditionalCFM


class JyutVoiceTTS(BaseLightningClass):
    def __init__(
        self,
        encoder: "TextEncoder",
        decoder: Optional["CausalConditionalCFM"],
        dp: "DurationPredictor",
        output_size=80,
        spk_embed_dim=192,
        freeze_encoder=False,
        freeze_decoder=False,
        optimizer=None,
        scheduler=None,
        warmup_steps=100,
        n_frame_per_class=10,
        slp: Optional["SpeechLengthPredictor"] = None,
        prosody_cfm: Optional["ProsodyCFM"] = None,
        pause_bonus=0.0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder
        self.n_feats = encoder.n_feats
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.output_size = output_size
        self.freeze_decoder = freeze_decoder
        self.freeze_encoder = freeze_encoder
        self.pause_bonus = pause_bonus  # log-prior reward per silent frame given to a pause token (0 = plain MAS)
        self.dp = dp
        self.prosody_cfm = prosody_cfm
        # Optional total-length predictor. When set, the duration predictor's per-token durations are
        # rescaled to the length it predicts; its weights live under `slp.*` (what train_slp.py saves).
        self.slp = slp
        self.n_frame_per_class = n_frame_per_class
        if freeze_encoder:  # train only what is not the encoder (e.g. the duration predictor)
            for prm in self.encoder.parameters():
                prm.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # a frozen encoder must not keep applying dropout
        return self

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        # A checkpoint trained without BERT has no `encoder.bert_proj.*`: start the projection at its (zero) initial value instead of failing
        # the strict load, so such a model can be fine-tuned with BERT features.
        state = checkpoint["state_dict"]
        for key, value in self.state_dict().items():
            if key.startswith("encoder.bert_proj.") and key not in state:
                state[key] = value.clone()

    def _predict_total_length(self, slp, x, x_mask, **kwargs):
        """Total mel length in frames, shape (B, 1), from an SLP with CE logits or an L1 scalar head."""
        slp_out = slp(x, x_mask, **kwargs)
        if slp_out.dim() > 1 and slp_out.shape[-1] > 1:  # CE logits
            total_length = ce_logits_to_length(slp_out, self.n_frame_per_class)
        else:  # L1 scalar
            total_length = slp_out.float()

        # Ensure total_length is (B, 1)
        if total_length.dim() == 1:
            total_length = total_length.unsqueeze(-1)
        return total_length

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
        prompt_feat=None,
        prompt_h=None,
        n_timesteps=10,
        temperature=1.0,
        length_scale=1.0,
        streaming=False,
        bert=None,
        duration_rounding="ceil",
        dp_temperature=None,
        dp_pause_temperature=None,
        dp_pause_mask=None,
        use_prosody=True,
        prosody_kwargs=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
                shape: (batch_size,)
            lang (torch.Tensor): language tokens.
                shape: (batch_size, max_text_length)
            tone (torch.Tensor): tone tokens.
                shape: (batch_size, max_text_length)
            word_pos (torch.Tensor): word position tokens.
                shape: (batch_size, max_text_length)
            syllable_pos (torch.Tensor): syllable position tokens.
                shape: (batch_size, max_text_length)
            prompt_feat (torch.Tensor): prompt mel-spectrogram for conditioning.
                shape: (batch_size, max_prompt_length, n_feats)
            prompt_h (torch.Tensor): prompt hidden states for conditioning.
                shape: (batch_size, max_prompt_length, n_feats)
            spk_embed (torch.Tensor): speaker embeddings.
                shape: (batch_size, spk_emb_dim)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of diffusion. Defaults to 1.0.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa. Defaults to 1.0.
            streaming (bool, optional)
            bert (torch.Tensor, optional): BERT features on the phone positions, as in training.
                shape: (batch_size, max_text_length, bert_dim). Required in practice when the encoder has `bert_dim > 0`.
            duration_rounding (str): how the predicted per-token durations become whole frames (ignored when an `slp` is set).
                "ceil" (default, the original behaviour) rounds every token up: at 50 Hz most tokens last 1-2 frames, so it adds about 0.6 frames
                per token and makes the utterance 20-25% longer than the recording. "round" rounds to nearest (at least one frame). "cumulative"
                rounds the running total (`distribute_frames`), so the utterance length is the sum of the predictions with no per-token bias.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        if duration_rounding not in ("ceil", "round", "cumulative"):
            raise ValueError(f"duration_rounding must be 'ceil', 'round' or 'cumulative', got {duration_rounding!r}")
        # For RTF computation
        t = dt.datetime.now()

        # Project speaker embedding for FM conditioning
        spk_embed = F.normalize(spk_embed, dim=1)
        c = self.spk_embed_affine_layer(spk_embed)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        if bert is None and getattr(self.encoder, "bert_dim", 0) > 0:
            warnings.warn(
                "the encoder was built with bert_dim > 0 but no BERT features were given; it will run as if there were no BERT input",
                stacklevel=2,
            )
        x_enc, mu_x, x_mask = self.encoder(
            x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed, **({} if bert is None else {"bert": bert})
        )  # x_enc = [B, n_feats, T_text], mu_x = [1, n_mel, T_text], x_mask = [1, 1, T_text]

        slp_inputs = dict(
            lang=lang,
            tone=tone,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_embed=spk_embed,
        )

        # Handle duration prediction
        if isinstance(self.dp, SpeechLengthPredictor):
            # Legacy: an SLP in the dp slot only gives a total length, so there are no per-token
            # durations to keep and every token (blanks included) gets the same share. The encoder was
            # trained on MAS durations, which are anything but uniform.
            warnings.warn(
                "dp is a SpeechLengthPredictor, so durations are spread uniformly over the tokens. "
                "Keep the DurationPredictor as `dp` and pass the SLP as `slp` to rescale its durations "
                "to the predicted length instead.",
                UserWarning,
                stacklevel=2,
            )
            total_length = self._predict_total_length(
                self.dp, x, x_mask, **slp_inputs
            )
            avg_dur = total_length / (x_lengths.float().unsqueeze(-1) + 1e-6)
            w = avg_dur.unsqueeze(-1).expand(-1, 1, x_mask.size(2)) * x_mask
        else:
            # FlowDurationPredictor or Standard DurationPredictor
            if isinstance(self.dp, FlowDurationPredictor):
                dp_kw = {}
                if dp_temperature is not None or dp_pause_temperature is not None:
                    # noise scale of the duration sampler per token: `dp_temperature` for speech tokens, `dp_pause_temperature` (default: the same) for the
                    # tokens flagged in `dp_pause_mask` (B, T_text; punctuation and edge pads), which keeps phone durations steady and pauses variable
                    t_speech = self.dp.infer_temperature if dp_temperature is None else dp_temperature
                    t_pause = t_speech if dp_pause_temperature is None else dp_pause_temperature
                    temp = torch.full_like(x_mask, float(t_speech))
                    if dp_pause_mask is not None:
                        temp = torch.where(dp_pause_mask.to(x_mask.device).bool().unsqueeze(1), torch.full_like(x_mask, float(t_pause)), temp)
                    dp_kw["temperature"] = temp
                logw = self.dp(x_enc, x_mask, bert=bert, g=spk_embed, **dp_kw)
            else:
                logw = self.dp(x_enc, x_mask)  # B, 1, T_text
            w = torch.exp(logw) * x_mask  # B, 1, T_text
            if self.slp is not None:
                # Keep the DP's relative rhythm, take the total from the SLP
                total_length = self._predict_total_length(
                    self.slp, x, x_mask, **slp_inputs
                )
                w = distribute_frames(w, x_mask, total_length)
            elif duration_rounding == "cumulative":
                w = distribute_frames(w, x_mask, w.sum(dim=(1, 2)).round().clamp_min(1.0).unsqueeze(-1))
            elif duration_rounding == "round":
                w = torch.clamp_min(torch.round(w), 1.0) * x_mask

        w_ceil = torch.ceil(w) * length_scale  # B, 1, T_text
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # B,
        y_max_length = y_lengths.max()

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        ).transpose(
            1, 2
        )  # B, n_feats, T_mel

        if self.prosody_cfm is not None and use_prosody:  # `use_prosody=False` keeps the encoder's mean; `prosody_kwargs` = n_steps / cfg_scale / temperature
            h_text_y = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2)
            ).transpose(1, 2)
            mark_kwargs = {}
            if getattr(self.prosody_cfm, "use_mark", False):  # punctuation-aware model: what mark comes next, and how far away
                if getattr(self, "_mark_table", None) is None:
                    from jyutvoice.models.prosody_cfm import mark_class_table
                    from jyutvoice.text.symbols import symbols

                    self._mark_table = mark_class_table(symbols)
                mark_kwargs["mark_y"], mark_kwargs["dist_y"] = mark_features(x, attn, self._mark_table)
            mu_y = self.prosody_cfm(
                mu_y=mu_y,
                h_text_y=h_text_y,
                attn=attn,
                y_mask=y_mask,
                g=spk_embed,
                **mark_kwargs,
                **(prosody_kwargs or {}),
            )

        encoder_outputs = mu_y[:, :, :y_max_length]

        # Enforce batch_size=1 for inference (decoder requirement)
        batch_size = x.size(0)
        if batch_size != 1:
            raise ValueError(
                f"synthesise() requires batch_size=1, got batch_size={batch_size}. "
                "Please pass one sample at a time."
            )

        if prompt_feat is not None and prompt_h is not None:
            n_mel = prompt_feat.shape[2]
            mel_len1, mel_len2 = prompt_feat.shape[1], mu_y.shape[2]
            mu_y = torch.cat([prompt_h.transpose(1, 2), mu_y], dim=2)
            conds = torch.zeros([1, mel_len1 + mel_len2, n_mel], device=x.device).to(
                mu_y.dtype
            )
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)

            mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(device=conds.device, dtype=conds.dtype)
        else:
            mel_len1 = 0
            conds = torch.zeros_like(mu_y).to(mu_y.dtype)
            mask = (~make_pad_mask(y_lengths)).to(mu_y.dtype)

        decoder_outputs = None
        rtf = None

        if self.decoder is not None:
            # Decoder forward pass
            decoder_outputs, _ = self.decoder(
                mu=mu_y,
                mask=mask.unsqueeze(1),
                spks=c,
                cond=conds,
                n_timesteps=n_timesteps,
                temperature=temperature,
                streaming=streaming,
            )
            decoder_outputs = decoder_outputs[:, :, mel_len1:]

            t = (dt.datetime.now() - t).total_seconds()
            rtf = t * 24000 / (decoder_outputs.shape[-1] * 480)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn,
            "mel": decoder_outputs,
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

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
        bert=None,
        prior_weight=None,
        silence=None,
        pause_mask=None,
        dur_weight=None,
    ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: the loss between the hidden state of CosyVoice2's flow encoder and the output of the text encoder.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x: Input token IDs (B, T_text)
            x_lengths: Lengths of input token sequences (B,)
            y: Target mel-spectrograms (B, n_mel_channels, T_mel)
            y_lengths: Lengths of target mel-spectrograms (B,)
            lang: Language information (B, T_text)
            tone: Tone information (B, T_text)
            word_pos: Word position information (B, T_text)
            syllable_pos: Syllable position information (B, T_text)
            spk_embed: Speaker embedding (B, spk_embed_dim)
            decoder_h: Hidden states from the flow encoder of CosyVoice2 (B, T_mel, n_feats)
            bert: Optional BERT features on the phone positions (B, T_text, bert_dim); zeros for samples without
            prior_weight: Optional (B,) weight of each sample in the prior loss (0 = the sample only trains the durations); default 1
            silence: Optional (B, T_mel) 1 where the frame is silent in the audio (see SilenceProbe); with pause_mask and `pause_bonus` > 0 makes MAS
                give silent frames to pause tokens (punctuation, edge pads) and not to the neighbouring phones
            pause_mask: Optional (B, T_text) True on the tokens that may hold a pause
            dur_weight: Optional (B, T_text) weight of each token in the duration loss (0 = not a target); used by the flow-matching duration predictor
            durations: Optional ground truth durations for teacher forcing
        """
        # xvec projection
        spk_embed = F.normalize(spk_embed, dim=1)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        # compute style conditioning
        x, mu_x, x_mask = self.encoder(
            x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed, **({} if bert is None else {"bert": bert})
        )  # x = [B, n_feats, T_text], mu_x = [1, n_feats, T_text], x_mask = [1, 1, T_text]

        y_max_length = decoder_h.shape[1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text tokens and decoder_h
        # The text encoder is trained to map text → decoder_h space, so align to decoder_h
        # Done in fp32 even under bf16 autocast: the matmuls below sum over n_feats and bf16 rounding
        # (~0.4% of values around -40) is as large as the score gaps MAS chooses paths by.
        with torch.no_grad(), torch.autocast(device_type=mu_x.device.type, enabled=False):
            mu_x_f = mu_x.float()
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones_like(mu_x_f)
            # Use decoder_h (flow encoder hidden states) for alignment computation
            # decoder_h: (batch, T_mel, n_feats) → transpose to (batch, n_feats, T_mel)
            h = decoder_h.transpose(1, 2).float()
            h_square = torch.matmul(factor.transpose(1, 2), (h**2))
            h_mu_double = torch.matmul(2.0 * (factor * mu_x_f).transpose(1, 2), h)
            mu_square = torch.sum(factor * (mu_x_f**2), 1).unsqueeze(-1)
            log_prior = h_square - h_mu_double + mu_square + const
            if self.pause_bonus > 0 and silence is not None and pause_mask is not None:
                sil = silence.float().unsqueeze(1)  # (B, 1, T_mel)
                log_prior = log_prior + self.pause_bonus * pause_mask.float().unsqueeze(2) * (sil - 0.5 * (1.0 - sil))

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1).float())
            attn = attn.detach()  # b, T_text, T_mel

        # Compute duration loss
        target_dur = torch.sum(attn.unsqueeze(1), -1)  # (B, 1, T_text)
        if isinstance(self.dp, FlowDurationPredictor):
            dur_loss = self.dp.compute_loss(
                x=x,
                x_mask=x_mask,
                target_dur=target_dur,
                x_lengths=x_lengths,
                bert=bert,
                g=spk_embed,
                token_weight=dur_weight,
            )
        else:
            logw = self.dp(x, x_mask)
            logw_ = torch.log(1e-8 + target_dur) * x_mask
            dur_loss = duration_loss(logw, logw_, x_lengths)

        # Align encoded text with mel-spectrogram and get mu_y segment
        # attn: [b, T_text, T_mel] -> [b, T_mel, T_text] x mu_x: [b, n_feats, T_text]
        # attn is an exact 0/1 path, so casting it to mu_x's dtype loses nothing
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2).to(mu_x.dtype), mu_x.transpose(1, 2)
        )  # (B, T_mel, n_feats)
        mu_y = mu_y.transpose(1, 2)

        # Compute prior / prosody loss
        prior_mask = y_mask if prior_weight is None else y_mask * prior_weight.view(-1, 1, 1).to(y_mask.dtype)
        if self.prosody_cfm is not None:
            h_text_y = torch.matmul(
                attn.squeeze(1).transpose(1, 2).to(x.dtype), x.transpose(1, 2)
            ).transpose(1, 2)
            prosody_loss = self.prosody_cfm.compute_loss(
                decoder_h=decoder_h.transpose(1, 2),
                mu_y=mu_y,
                h_text_y=h_text_y,
                attn=attn,
                y_mask=y_mask,
                y_lengths=y_lengths,
                g=spk_embed,
                prior_weight=prior_weight,
            )
            # anchor mu_x so MAS remains well-conditioned
            anchor_loss = torch.sum(
                0.5 * ((decoder_h.transpose(1, 2) - mu_y) ** 2) * prior_mask
            ) / (torch.sum(prior_mask).clamp_min(1.0) * self.n_feats)
            prior_loss = prosody_loss + 0.1 * anchor_loss
        else:
            prior_loss = torch.sum(
                0.5
                * ((decoder_h.transpose(1, 2) - mu_y) ** 2 + math.log(2 * math.pi))
                * prior_mask
            )
            prior_loss = prior_loss / (torch.sum(prior_mask).clamp_min(1.0) * self.n_feats)

        return dur_loss, prior_loss, attn

