"""from https://github.com/jaywalnut310/glow-tts"""

import math

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
from jyutvoice.models.diffusion_transformer import DiTConVBlock
from jyutvoice.utils.common import sequence_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        n_lang,  # PAD + 1: Cantonese, 2: Mandarin, 3: English
        n_tone,  # PAD + 6 tones
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.scale = self.hidden_channels**0.5

        # token-level embeddings
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # language embedding (we'll pool to global later)
        self.lang_emb = nn.Embedding(n_lang, self.hidden_channels)
        nn.init.normal_(self.lang_emb.weight, 0.0, self.hidden_channels**-0.5)

        self.tone_emb = nn.Embedding(n_tone, self.hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, self.hidden_channels**-0.5)

        self.encoder = nn.ModuleList(
            [
                DiTConVBlock(
                    hidden_channels,
                    filter_channels,
                    n_heads,
                    kernel_size,
                    p_dropout,
                    gin_channels,
                )
                for _ in range(n_layers)
            ]
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        tone: torch.Tensor,
        lang: torch.Tensor,
        c: torch.Tensor,
        x_lengths: torch.Tensor,
    ):
        x = (
            self.emb(x) + self.tone_emb(tone) + self.lang_emb(lang)
        ) * self.scale  # [B, T, D]

        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        # mask: (B, 1, T)
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        for layer in self.encoder:
            x = layer(x, c, x_mask)
        mu_x = self.proj(x) * x_mask

        return x, mu_x, x_mask


if __name__ == "__main__":
    from omegaconf import DictConfig

    text_encoder = TextEncoder(
        n_vocab=38,  # 49
        n_lang=3,
        n_tone=7,
        out_channels=192,
        hidden_channels=256,
        filter_channels=1024,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        gin_channels=128,
    )

    x = torch.randint(0, 38, (2, 10))
    x_lengths = torch.tensor([10, 8])
    tone = torch.randint(0, 7, (2, 10))
    lang = torch.randint(0, 3, (2, 10))
    c = torch.randn(2, 128)

    mu, logw, x_mask = text_encoder(x, tone, lang, c, x_lengths)

    print(mu.shape)
    print(logw.shape)
    print(x_mask.shape)
