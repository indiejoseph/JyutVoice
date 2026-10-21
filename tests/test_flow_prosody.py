import os
import sys
import pytest
import torch
from omegaconf import DictConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
from jyutvoice.models.prosody_cfm import ProsodyCFM
from jyutvoice.models.text_encoder import TextEncoder
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS


def test_flow_duration_predictor_train_and_infer():
    B, T, C, bert_dim, spk_dim = 2, 12, 192, 1024, 192
    model = FlowDurationPredictor(
        in_channels=C,
        filter_channels=128,
        n_layers=3,
        kernel_size=5,
        bert_dim=bert_dim,
        spk_embed_dim=spk_dim,
    )

    x = torch.randn(B, C, T, requires_grad=True)
    x_mask = torch.ones(B, 1, T)
    x_mask[:, :, 10:] = 0.0
    x_lengths = torch.tensor([10, 8])
    target_dur = torch.randint(1, 10, (B, 1, T)).float() * x_mask
    bert = torch.randn(B, T, bert_dim)
    g = torch.randn(B, spk_dim)

    # 1. Training loss & gradient flow to x
    loss = model.compute_loss(x, x_mask, target_dur, x_lengths, bert=bert, g=g)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # 2. Inference forward
    model.eval()
    with torch.no_grad():
        pred_logw = model(x, x_mask, bert=bert, g=g, n_steps=5, cfg_scale=1.5)
    assert pred_logw.shape == (B, 1, T)
    assert (pred_logw * (1.0 - x_mask)).abs().max() == 0.0


def test_prosody_cfm_train_and_infer():
    B, T_text, T_mel, C, bert_dim, spk_dim = 2, 8, 40, 192, 1024, 192
    prosody = ProsodyCFM(
        in_channels=80,
        text_channels=C,
        hidden_channels=128,
        n_blocks=3,
        kernel_size=5,
        spk_embed_dim=spk_dim,
    )

    decoder_h = torch.randn(B, 80, T_mel)
    mu_y = torch.randn(B, 80, T_mel)
    h_text_y = torch.randn(B, C, T_mel)
    y_mask = torch.ones(B, 1, T_mel)
    y_lengths = torch.tensor([40, 35])
    g = torch.randn(B, spk_dim)

    # Synthesize a valid monotonic alignment
    attn = torch.zeros(B, 1, T_text, T_mel)
    for b in range(B):
        # 5 frames per token
        for t in range(T_text):
            attn[b, 0, t, t * 5 : (t + 1) * 5] = 1.0

    # 1. Intra-phone fractional pos test
    tau, dur = ProsodyCFM.compute_fractional_pos(attn)
    assert tau.shape == (B, 1, T_mel)
    assert (tau >= 0.0).all() and (tau <= 1.0).all()
    assert (dur == 5.0).all()

    # 2. Training loss
    loss = prosody.compute_loss(
        decoder_h=decoder_h,
        mu_y=mu_y,
        h_text_y=h_text_y,
        attn=attn,
        y_mask=y_mask,
        y_lengths=y_lengths,
        g=g,
    )
    assert torch.isfinite(loss)

    # 3. Inference forward
    prosody.eval()
    with torch.no_grad():
        gen_h = prosody(mu_y, h_text_y, attn, y_mask, g=g, n_steps=5, cfg_scale=1.0)
    assert gen_h.shape == (B, 80, T_mel)
    assert torch.isfinite(gen_h).all()


def test_jyutvoice_tts_with_flow_prosody():
    B, T_text, T_mel, C, spk_dim, bert_dim = 2, 8, 30, 64, 16, 64

    enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig(
            {
                "n_feats": 80,
                "n_channels": C,
                "filter_channels": 64,
                "filter_channels_dp": 32,
                "n_heads": 2,
                "n_layers": 3,
                "kernel_size": 3,
                "p_dropout": 0.0,
                "prenet": False,
                "gin_channels": spk_dim,
            }
        ),
        n_vocab=97,
        n_lang=4,
        n_tone=7,
        bert_dim=bert_dim,
    )

    flow_dp = FlowDurationPredictor(
        in_channels=C,
        filter_channels=64,
        n_layers=2,
        bert_dim=bert_dim,
        spk_embed_dim=spk_dim,
    )

    prosody = ProsodyCFM(
        in_channels=80,
        text_channels=C,
        hidden_channels=64,
        n_blocks=2,
        spk_embed_dim=spk_dim,
    )

    tts = JyutVoiceTTS(
        encoder=enc,
        decoder=None,
        dp=flow_dp,
        prosody_cfm=prosody,
        output_size=80,
        spk_embed_dim=spk_dim,
    )

    x = torch.randint(1, 97, (B, T_text))
    x_lengths = torch.full((B,), T_text)
    lang = torch.zeros(B, T_text, dtype=torch.long)
    tone = torch.zeros(B, T_text, dtype=torch.long)
    word_pos = torch.zeros(B, T_text, dtype=torch.long)
    syllable_pos = torch.zeros(B, T_text, dtype=torch.long)
    spk_embed = torch.randn(B, spk_dim)
    decoder_h = torch.randn(B, T_mel, 80)
    bert = torch.randn(B, T_text, bert_dim)

    # 1. Forward training pass
    dur_loss, prior_loss, attn = tts(
        x=x,
        x_lengths=x_lengths,
        y=torch.zeros(B, 80, T_mel),
        y_lengths=torch.full((B,), T_mel),
        lang=lang,
        tone=tone,
        word_pos=word_pos,
        syllable_pos=syllable_pos,
        spk_embed=spk_embed,
        decoder_h=decoder_h,
        bert=bert,
    )
    assert torch.isfinite(dur_loss) and torch.isfinite(prior_loss)
    (dur_loss + prior_loss).backward()

    # 2. Synthesise pass (batch_size=1)
    tts.eval()
    with torch.no_grad():
        out = tts.synthesise(
            x=x[:1],
            x_lengths=x_lengths[:1],
            lang=lang[:1],
            tone=tone[:1],
            word_pos=word_pos[:1],
            syllable_pos=syllable_pos[:1],
            spk_embed=spk_embed[:1],
            bert=bert[:1],
        )
    assert out["encoder_outputs"].shape[1] == 80
    assert out["mel_lengths"].item() > 0
