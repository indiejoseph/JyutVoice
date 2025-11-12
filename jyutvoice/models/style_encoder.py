"""Style encoder of GST-Tacotron."""

from typing import Sequence

import torch
from typeguard import typechecked
import math
import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer without SequenceModule.

    Args:
        q_dim (int): Dimension of query input.
        k_dim (int): Dimension of key input.
        v_dim (int): Dimension of value input.
        n_head (int): Number of heads.
        n_feat (int): Total feature size after projection (must be divisible by n_head).
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert n_feat % n_head == 0, "n_feat must be divisible by n_head"

        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head

        # projections
        self.linear_q = nn.Linear(q_dim, n_feat)
        self.linear_k = nn.Linear(k_dim, n_feat)
        self.linear_v = nn.Linear(v_dim, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        # kept for possible compatibility with code expecting this attr
        self.d_output = n_feat

    # -------- qkv projection --------
    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (Tensor): (#batch, time1, q_dim)
            key   (Tensor): (#batch, time2, k_dim)
            value (Tensor): (#batch, time2, v_dim)

        Returns:
            q, k, v each as (#batch, n_head, time*, d_k)
        """
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)

        # (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    # -------- attention core --------
    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (Tensor): (#batch, n_head, time2, d_k)
            scores (Tensor): (#batch, n_head, time1, time2)
            mask (Tensor or None): (#batch, 1, time2) or (#batch, time1, time2)

        Returns:
            Tensor: (#batch, time1, n_feat)
        """
        n_batch = value.size(0)

        if mask is not None:
            # Expect (#batch, 1, time2) or (#batch, time1, time2)
            # -> (#batch, 1, *, time2) so it can broadcast with scores
            if mask.dim() == 3:
                # (B, T1/Ty, T2) -> (B, 1, T1/Ty, T2)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                # (B, T2) -> (B, 1, 1, T2)
                mask = mask.unsqueeze(1).unsqueeze(2)
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

            # convert to boolean "is_pad"
            pad_mask = mask.eq(0)  # True where we want to mask

            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(pad_mask, min_value)

            self.attn = torch.softmax(scores, dim=-1).masked_fill(pad_mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (B, H, T1, d_k)

        # (B, H, T1, d_k) -> (B, T1, H * d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)  # (B, T1, n_feat)

    # -------- main forward --------
    def forward(self, query, key, value, mask=None, *args, **kwargs):
        """Compute scaled dot-product attention.

        Args:
            query (Tensor): (#batch, time1, q_dim)
            key   (Tensor): (#batch, time2, k_dim)
            value (Tensor): (#batch, time2, v_dim)
            mask  (Tensor or None): (#batch, 1, time2) or (#batch, time1, time2)

        Returns:
            Tensor: (#batch, time1, n_feat)
        """
        q, k, v = self.forward_qkv(query=query, key=key, value=value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

    # Optional: keep a 'step' API if you want to mimic the SequenceModule version
    def step(self, query, state=None, memory=None, mask=None, **kwargs):
        """One-step autoregressive interface (optional helper).

        query: (#batch, 1, q_dim)
        memory: (#batch, T_mem, k_dim/v_dim) or None (defaults to query)
        """
        if memory is None:
            memory = query
        out = self.forward(query, memory, memory, mask=mask)
        # return (B, d_output), state
        return out.squeeze(1), state


class StyleEncoder(torch.nn.Module):
    """Style encoder.

    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    Todo:
        * Support manual weight specification in inference.

    """

    @typechecked
    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize global style encoder module."""
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )
        self.gst_token_dim = gst_token_dim

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)

        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    @typechecked
    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    @typechecked
    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)

        return style_embs.squeeze(1)


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- config ----
    BATCH_SIZE = 2
    T_MEL = 200  # time frames
    IDIM = 80  # mel dim (must match StyleEncoder default)

    # ---- create dummy mel-spectrogram ----
    # shape: (B, Lmax, idim)
    speech = torch.randn(BATCH_SIZE, T_MEL, IDIM)
    print(f"Input speech shape: {speech.shape}")

    # ---- build model ----
    model = StyleEncoder(
        idim=IDIM,
        gst_tokens=10,
        gst_token_dim=256,
        gst_heads=4,
        conv_layers=6,
        conv_chans_list=(32, 32, 64, 64, 128, 128),
        conv_kernel_size=3,
        conv_stride=2,
        gru_layers=1,
        gru_units=128,
    )

    # ---- forward pass through full StyleEncoder ----
    with torch.no_grad():
        style_embs = model(speech)

    print("\n=== StyleEncoder debug ===")
    print(f"Style embeddings shape: {style_embs.shape}")  # expect (B, gst_token_dim)
    print(f"Style embeddings dtype: {style_embs.dtype}")
    print(f"Any NaNs in style_embs? {torch.isnan(style_embs).any().item()}")

    # check ref encoder output
    with torch.no_grad():
        ref_embs = model.ref_enc(speech)
    print(f"Reference embedding shape: {ref_embs.shape}")  # (B, gru_units)

    # ---- inspect attention inside StyleTokenLayer ----
    mha = model.stl.mha
    attn = mha.attn
    print("\n=== MHA inside StyleTokenLayer ===")
    if attn is None:
        print("mha.attn is None (this is expected BEFORE a forward call).")
    else:
        print(f"Attention weights shape: {attn.shape}")
        print(f"Any NaNs in attn? {torch.isnan(attn).any().item()}")

    # ---- direct MHA test (with and without mask) ----
    print("\n=== Direct MultiHeadedAttention test ===")
    B = 2
    Tq = 3
    Tk = 5
    q_dim = 128
    k_dim = 64
    v_dim = 64
    n_head = 4
    n_feat = 256

    mha_test = MultiHeadedAttention(
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=0.0,
    )

    q = torch.randn(B, Tq, q_dim)
    k = torch.randn(B, Tk, k_dim)
    v = torch.randn(B, Tk, v_dim)

    # no mask
    out_nomask = mha_test(q, k, v, mask=None)
    print(f"Output (no mask) shape: {out_nomask.shape}")  # (B, Tq, n_feat)
    print(f"Any NaNs in out_nomask? {torch.isnan(out_nomask).any().item()}")

    # with mask: shape (B, 1, Tk) with some zeros as padding
    mask = torch.ones(B, 1, Tk, dtype=torch.long)
    mask[:, :, -2:] = 0  # last 2 positions masked out
    out_mask = mha_test(q, k, v, mask=mask)
    print(f"Output (with mask) shape: {out_mask.shape}")  # (B, Tq, n_feat)
    print(f"Any NaNs in out_mask? {torch.isnan(out_mask).any().item()}")
    print(f"Attention weights shape (test MHA): {mha_test.attn.shape}")
    print("Mask test OK if last two key positions have near-zero attention.\n")

    print("Debug run finished.")
