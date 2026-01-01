import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class SpeechLengthPredictor(nn.Module):
    """
    Transformer Encoder-Decoder architecture for speech length prediction.
    Uses linguistic embeddings (phoneme, tone, lang, etc.) as input.
    """

    def __init__(
        self,
        n_vocab=100,
        n_lang=4,
        n_tone=7,
        n_mel=80,
        hidden_dim=192,  # Should match TextEncoder.hidden_channels
        n_text_layer=4,  # layers to re-interpret text features
        n_cross_layer=4,
        n_head=4,
        output_dim=1,
        spk_embed_dim=192,
    ):
        super().__init__()

        # Linguistic Embeddings
        self.emb = nn.Embedding(n_vocab, hidden_dim)
        self.lang_emb = nn.Embedding(n_lang, hidden_dim)
        self.tone_emb = nn.Embedding(n_tone, hidden_dim)
        self.word_pos_emb = nn.Embedding(4, hidden_dim)
        self.syllable_pos_emb = nn.Embedding(4, hidden_dim)
        self.spk_proj = nn.Linear(spk_embed_dim, hidden_dim)

        # Text Feature Re-interpreter (Memory Processing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_text_layer
        )
        self.text_pe = PositionalEncoding(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)

        # Mel Spectrogram Embedder (used during training)
        self.mel_embedder = nn.Linear(n_mel, hidden_dim)
        self.mel_norm = nn.LayerNorm(hidden_dim)
        self.mel_pe = PositionalEncoding(hidden_dim)

        # Transformer Decoder Layers with Cross-Attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_cross_layer)

        # Final Prediction Layer (predicts remaining length or total length)
        self.predictor = nn.Linear(hidden_dim, output_dim)

        # Query for inference (to predict total length)
        self.length_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        x,
        x_mask,
        lang,
        tone,
        word_pos,
        syllable_pos,
        spk_embed,
        mel=None,
    ):
        """
        Args:
            x: (B, T_text) phoneme IDs
            x_mask: (B, 1, T_text)
            lang: (B, T_text) language IDs
            tone: (B, T_text) tone IDs
            word_pos: (B, T_text) word position IDs
            syllable_pos: (B, T_text) syllable position IDs
            spk_embed: (B, spk_embed_dim) speaker embeddings
            mel: (B, n_mel, T_mel) target mel (optional, for training)
        """
        B = x.size(0)

        # Embed linguistic features
        memory = self.emb(x)
        memory = memory + self.lang_emb(lang)
        memory = memory + self.tone_emb(tone)
        memory = memory + self.word_pos_emb(word_pos)
        memory = memory + self.syllable_pos_emb(syllable_pos)

        # Add speaker embedding as global condition
        if spk_embed is not None:
            g = self.spk_proj(spk_embed).unsqueeze(1)  # (B, 1, C)
            memory = memory + g

        # memory is (B, T_text, C)
        # memory_mask: True for padding positions
        memory_key_padding_mask = x_mask.squeeze(1) == 0

        # Add positional encoding and re-interpret text features for duration task
        memory = self.text_pe(memory)
        memory = self.text_encoder(memory, src_key_padding_mask=memory_key_padding_mask)
        memory = self.text_norm(memory)

        if mel is not None:
            # Training mode: Predict remaining length at each mel frame
            # mel: (B, n_mel, T_mel) -> (B, T_mel, n_mel)
            mel_features = self.mel_norm(self.mel_embedder(mel.transpose(1, 2)))

            # Prepend length_query to mel_features to train it for total length prediction
            query = self.mel_norm(self.length_query.expand(B, -1, -1))
            tgt = torch.cat([query, mel_features], dim=1)

            # Add speaker embedding as global condition to decoder input
            if spk_embed is not None:
                tgt = tgt + g

            tgt = self.mel_pe(tgt)

            seq_len = tgt.size(1)
            # Causal mask for mel sequence
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=mel.device), diagonal=1
            ).bool()

            decoder_out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            # Inference mode: Predict total length using a query
            query = self.mel_norm(self.length_query.expand(B, -1, -1))
            tgt = query

            # Add speaker embedding as global condition to decoder input
            if spk_embed is not None:
                tgt = tgt + g

            tgt = self.mel_pe(tgt)
            decoder_out = self.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        length_logits = self.predictor(decoder_out)
        if length_logits.shape[-1] == 1:
            length_logits = length_logits.squeeze(-1)
        return length_logits


def calculate_remaining_lengths(mel_lengths, max_L=None):
    B = mel_lengths.shape[0]
    if max_L is None:
        max_L = mel_lengths.max().item()
    range_tensor = torch.arange(max_L, device=mel_lengths.device).expand(B, max_L)
    remain_lengths = (mel_lengths[:, None] - 1 - range_tensor).clamp(min=0)
    # Prepend total length as the target for the length_query
    remain_lengths = torch.cat([mel_lengths[:, None], remain_lengths], dim=1)
    return remain_lengths


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    n_mel = 80
    text_len = 10
    mel_len = 15
    hidden_dim = 192

    text_features = torch.randn(batch_size, hidden_dim, text_len)
    text_mask = torch.ones(batch_size, 1, text_len)
    mel = torch.randn(batch_size, n_mel, mel_len)
    mel_lengths = torch.tensor([15, 12])

    model = SpeechLengthPredictor(n_mel=n_mel, hidden_dim=hidden_dim)
    pred_remain_lengths = model(text_features, text_mask, mel=mel)
    y_mask = torch.arange(mel_len).expand(batch_size, mel_len) < mel_lengths.unsqueeze(
        1
    )
    y_mask = y_mask.unsqueeze(1).float()

    print("Predicted Remaining Lengths:", pred_remain_lengths)
