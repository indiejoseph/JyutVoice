import os
import argparse
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from hyperpyyaml import load_hyperpyyaml

from jyutvoice.models.slp import SpeechLengthPredictor, calculate_remaining_lengths
from jyutvoice.data.text_mel_datamodule import TextMelDataModule
from jyutvoice.utils import instantiators


def masked_l1_loss(est_lengths, tar_lengths, mask):
    """
    L1 loss masked to only include valid frames.
    """
    loss = F.l1_loss(est_lengths, tar_lengths, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum()


def masked_cross_entropy_loss(est_length_logits, tar_length_labels, mask):
    """
    Cross entropy loss masked to only include valid frames.
    """
    B, L, C = est_length_logits.shape
    loss = F.cross_entropy(
        est_length_logits.reshape(-1, C),
        tar_length_labels.reshape(-1),
        reduction="none",
    ).reshape(B, L)
    loss = loss * mask
    return loss.sum() / mask.sum()


class SLPLightningModule(L.LightningModule):
    def __init__(self, tts_model, slp_model, config):
        super().__init__()
        self.save_hyperparameters(ignore=["tts_model", "slp_model"])
        self.tts = tts_model
        self.slp = slp_model

        self.loss_fn = config.get("loss_fn", "L1")
        self.lambda_L1 = config.get("lambda_L1", 1.0)
        self.n_frame_per_class = config.get("n_frame_per_class", 10)
        self.n_class = config.get("n_class", 301)
        self.gumbel_tau = config.get("gumbel_tau", 0.5)
        self.lr = config.get("lr", 1e-4)
        self.warmup_updates = config.get("warmup_updates", 5000)

        # Audio params for sec_error
        self.hop_length = config.get("hop_length", 480)
        self.sample_rate = config.get("sample_rate", 24000)

        # Freeze TTS
        self.tts.eval()
        for param in self.tts.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Override train to ensure TTS encoder stays in eval mode (no dropout).
        """
        super().train(mode)
        self.tts.eval()

    def _compute_loss_and_metrics(self, batch):
        x = batch["x"]
        x_lengths = batch["x_lengths"]
        y = batch["y"]
        y_lengths = batch["y_lengths"]
        lang = batch["lang"]
        tone = batch["tone"]
        word_pos = batch["word_pos"]
        syllable_pos = batch["syllable_pos"]
        spk_embed = batch["spk_embed"]

        # Create x_mask (B, 1, T_text)
        from jyutvoice.utils.model import sequence_mask

        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1).to(x.dtype)

        # Predict remaining lengths using raw linguistic features
        predictions = self.slp(
            x,
            x_mask,
            lang=lang,
            tone=tone,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_embed=spk_embed,
            mel=y,
        )

        # Target remaining lengths: (B, T_mel + 1)
        target_remain_lengths = calculate_remaining_lengths(y_lengths, max_L=y.size(-1))

        # Create mask for targets (B, T_mel + 1)
        B, T_plus_1 = target_remain_lengths.shape
        mask = torch.zeros_like(target_remain_lengths, dtype=torch.bool)
        mask[:, 0] = True  # Total length query
        for i in range(B):
            mask[i, 1 : y_lengths[i] + 1] = True

        if self.loss_fn == "L1":
            loss = masked_l1_loss(predictions, target_remain_lengths.float(), mask)
            frame_error = loss
        elif self.loss_fn == "CE":
            tar_labels = (target_remain_lengths // self.n_frame_per_class).clamp(
                0, self.n_class - 1
            )
            loss = masked_cross_entropy_loss(predictions, tar_labels, mask)
            with torch.no_grad():
                est_lengths = torch.argmax(predictions, dim=-1) * self.n_frame_per_class
                frame_error = masked_l1_loss(
                    est_lengths.float(), target_remain_lengths.float(), mask
                )
        elif self.loss_fn == "L1_and_CE":
            tar_labels = (target_remain_lengths // self.n_frame_per_class).clamp(
                0, self.n_class - 1
            )
            loss_CE = masked_cross_entropy_loss(predictions, tar_labels, mask)

            # Use Gumbel-Softmax for differentiable L1 loss on top of CE
            est_1hots = F.gumbel_softmax(
                predictions, tau=self.gumbel_tau, hard=True, dim=-1
            )
            values = (
                torch.arange(self.n_class, device=predictions.device).float()
                * self.n_frame_per_class
            )
            est_lengths = (est_1hots * values).sum(-1)
            loss_L1 = masked_l1_loss(est_lengths, target_remain_lengths.float(), mask)

            loss = loss_CE + self.lambda_L1 * loss_L1
            frame_error = loss_L1
        else:
            raise ValueError(f"Unsupported loss_fn: {self.loss_fn}")

        # Calculate error in seconds
        sec_error = frame_error * self.hop_length / self.sample_rate

        return loss, sec_error

    def training_step(self, batch, batch_idx):
        loss, sec_error = self._compute_loss_and_metrics(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train/sec_error", sec_error, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, sec_error = self._compute_loss_and_metrics(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/sec_error", sec_error, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.slp.parameters(), lr=self.lr)

        # Scheduler setup: Linear Warmup + Linear Decay
        warmup_steps = self.warmup_updates
        # estimated_stepping_batches is available after trainer is attached
        total_steps = self.trainer.estimated_stepping_batches
        decay_steps = max(total_steps - warmup_steps, 1)

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
        )
        decay_scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=1e-5, total_iters=decay_steps
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Train Speech Length Predictor (SLP)")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pretrained_models/20251231.pt",
        help="Path to TTS checkpoint",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="L1_and_CE",
        choices=["L1", "CE", "L1_and_CE"],
        help="Loss function",
    )
    parser.add_argument(
        "--lambda_L1", type=float, default=1.0, help="Weight for L1 loss in L1_and_CE"
    )
    parser.add_argument(
        "--n_class", type=int, default=301, help="Number of classes for CE loss"
    )
    parser.add_argument(
        "--n_frame_per_class", type=int, default=10, help="Frames per class for CE loss"
    )
    parser.add_argument(
        "--gumbel_tau", type=float, default=0.5, help="Tau for Gumbel-Softmax"
    )
    parser.add_argument("--warmup_updates", type=int, default=5000, help="Warmup steps")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SLP checkpoint to load weights",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Filter samples longer than this duration (seconds)",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=None,
        help="Filter samples with text longer than this length",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = load_hyperpyyaml(f)

    # Initialize TTS and load weights
    tts = cfg["tts"]
    if os.path.exists(args.ckpt):
        print(f"Loading TTS weights from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        # Remove 'tts.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("tts."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        tts.load_state_dict(new_state_dict, strict=False)
    else:
        print(
            f"Warning: Checkpoint {args.ckpt} not found. Training with random encoder weights."
        )

    # Initialize SLP with correct output_dim
    output_dim = args.n_class if args.loss_fn in ["CE", "L1_and_CE"] else 1
    hidden_dim = tts.encoder.n_channels
    slp = SpeechLengthPredictor(
        n_vocab=tts.encoder.n_vocab,
        n_lang=tts.encoder.lang_emb.num_embeddings,
        n_tone=tts.encoder.tone_emb.num_embeddings,
        n_mel=80,
        hidden_dim=hidden_dim,
        n_text_layer=4,
        n_cross_layer=4,
        n_head=8,
        output_dim=output_dim,
        spk_embed_dim=192,
    )
    # Setup DataModule
    datamodule = cfg["data"]
    if args.batch_size:
        datamodule.hparams.batch_size = args.batch_size
    if args.max_duration:
        datamodule.hparams.max_duration = args.max_duration
    if args.max_text_length:
        datamodule.hparams.max_text_length = args.max_text_length

    # Prepare config for LightningModule
    slp_config = {
        "loss_fn": args.loss_fn,
        "lambda_L1": args.lambda_L1,
        "n_class": args.n_class,
        "n_frame_per_class": args.n_frame_per_class,
        "gumbel_tau": args.gumbel_tau,
        "lr": args.lr,
        "warmup_updates": args.warmup_updates,
        "hop_length": cfg.get("hop_length", 480),
        "sample_rate": cfg.get("sample_rate", 24000),
    }

    # Initialize Lightning Module
    model = SLPLightningModule(tts, slp, slp_config)

    # Logger and Callbacks
    logger = instantiators.instantiate_loggers(cfg.get("logger"))
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/slp",
        filename="slp-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    callbacks = [checkpoint_callback]
    if logger:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
