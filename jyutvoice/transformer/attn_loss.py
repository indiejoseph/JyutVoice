import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedAttentionLoss(nn.Module):
    """
    Guided Attention Loss encourages diagonal attention alignment between input and output sequences.

    Modified version for decoder-only models where input and output tokens are concatenated,
    and attention operates on the full combined sequence.

    Args:
        guided_attn_weight (float): Weighting factor for the guided attention loss.
        reduction_factor (int): Factor by which output length is reduced (typically 1 in decoder-only).
        attn_sigma (float): Standard deviation controlling the diagonal width of the attention guide.
    """

    def __init__(self, guided_attn_weight, reduction_factor, attn_sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.guided_attn_weight = guided_attn_weight
        self.reduction_factor = reduction_factor
        self.attn_sigma = attn_sigma

    def forward(self, att_ws_stack, input_length, output_length):
        """
        Computes the guided attention loss for decoder-only sequence alignment.

        Args:
            att_ws_stack (Tensor): Attention weights of shape [B, H, T, T].
            input_length (Tensor): Lengths of input portion of the sequence [B].
            output_length (Tensor): Lengths of output portion of the sequence [B].

        Returns:
            Tensor: Scalar loss value.
        """
        output_offset = input_length + 2  # Account for sos and task_id
        text_offset = 1
        attn_masks = self._create_attention_masks(input_length, output_length)
        length_masks = self._create_length_masks(input_length, output_length)

        if att_ws_stack.dim() == 3:
            att_ws_stack = att_ws_stack.unsqueeze(1)  # [B, 1, T, T]

        # Crop to output tokens attending to input tokens
        att_ws_stack = torch.stack(
            [
                att_ws[
                    :,
                    output_offset[i] : output_offset[i] + output_length[i],
                    text_offset : text_offset + input_length[i],
                ]
                for i, att_ws in enumerate(att_ws_stack)
            ],
            dim=0,
        )  # assumes padding aligned
        losses = attn_masks * att_ws_stack
        losses *= length_masks
        loss = losses.sum()
        total_size = length_masks.sum().clamp(min=1.0)  # avoid division by zero

        return self.guided_attn_weight * loss / total_size

    def _create_attention_masks(self, input_length, output_length):
        """
        Create guided attention masks for decoder-only models:
        aligns output (right half) attending to input (left half).

        Returns:
            Tensor: [B, 1, output_length, input_length] mask
        """
        batch_size = input_length.size(0)
        input_max_len = input_length.max()
        output_max_len = output_length.max()

        grid_x = torch.arange(output_max_len, device=input_length.device).view(1, -1, 1)
        grid_y = torch.arange(input_max_len, device=input_length.device).view(1, 1, -1)

        grid_x = grid_x.expand(batch_size, -1, input_max_len)
        grid_y = grid_y.expand(batch_size, output_max_len, -1)

        input_length = input_length.view(-1, 1, 1).float()
        output_length = output_length.view(-1, 1, 1).float()

        masks = 1.0 - torch.exp(
            -((grid_y / input_length - grid_x / output_length) ** 2)
            / (2 * self.attn_sigma**2)
        )
        return masks.unsqueeze(1)

    def _create_length_masks(self, input_length, output_length):
        """
        Create validity masks based on non-padding areas of input and output regions.

        Returns:
            Tensor: [B, 1, output_length, input_length] mask
        """
        input_max_len = input_length.max()
        output_max_len = output_length.max()

        input_mask = torch.arange(input_max_len, device=input_length.device).unsqueeze(
            0
        ) < input_length.unsqueeze(1)
        output_mask = torch.arange(
            output_max_len, device=output_length.device
        ).unsqueeze(0) < output_length.unsqueeze(1)

        input_mask = input_mask.unsqueeze(1).expand(-1, output_max_len, -1)
        output_mask = output_mask.unsqueeze(2).expand(-1, -1, input_max_len)

        return (input_mask & output_mask).unsqueeze(1).float()


class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """
    Multi-layer, multi-head variant of Guided Attention Loss for decoder-only attention.

    Args:
        guided_attn_weight (float): Weighting factor for the guided attention loss.
        reduction_factor (int): Output frame reduction factor.
        attn_sigma (float): Standard deviation for the diagonal alignment.
        num_heads (int): Number of attention heads to apply the loss to.
        num_layers (int): Number of recent layers to apply the loss on.
    """

    def __init__(
        self,
        guided_attn_weight,
        reduction_factor,
        attn_sigma=0.4,
        num_heads=2,
        num_layers=2,
    ):
        super().__init__(guided_attn_weight, reduction_factor, attn_sigma)
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, att_ws_stack, input_length, output_length):
        """
        Computes guided attention loss across selected layers and heads.

        Args:
            att_ws_stack (List[Tensor]): List of attention weights per layer [B, H, T, T].
            input_length (Tensor): Input segment lengths [B].
            output_length (Tensor): Output segment lengths [B].

        Returns:
            Tensor: Total guided attention loss across layers and heads.
        """
        total_loss = 0
        total_layers = len(att_ws_stack)
        for index, layer_index in enumerate(reversed(range(total_layers))):
            if index >= self.num_layers:
                break
            att_ws_layer = att_ws_stack[layer_index]  # [B, H, T, T]
            total_loss += super().forward(
                att_ws_layer[:, : self.num_heads], input_length, output_length
            )
        return total_loss
