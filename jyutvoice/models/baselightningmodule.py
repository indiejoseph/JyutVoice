"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
import math
from abc import ABC
from typing import Any, Dict
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim.lr_scheduler import LambdaLR, SequentialLR
import wandb

from jyutvoice import utils
from jyutvoice.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())

        # Get warmup steps from config (default to 1000 if not specified)
        warmup_steps = getattr(self.hparams, "warmup_steps", 1000)

        # Define warmup function
        def warmup_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        # Create warmup scheduler
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # Check if main scheduler is configured
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if (
                "last_epoch"
                in inspect.signature(self.hparams.scheduler.scheduler).parameters
            ):
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            main_scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            main_scheduler.last_epoch = current_epoch

            # Combine warmup + main scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update LR every step
                    "frequency": 1,
                    "name": "learning_rate",
                },
            }
        else:
            # Only warmup scheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate",
                },
            }

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        z, z_lengths = batch["z"], batch["z_lengths"]
        lang, tone, word_pos, syllable_pos = (
            batch["lang"],
            batch["tone"],
            batch["word_pos"],
            batch["syllable_pos"],
        )
        spk_embed = batch["spk_embed"]
        decoder_h = batch["decoder_h"]

        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            lang=lang,
            tone=tone,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_embed=spk_embed,
            decoder_h=decoder_h,
            z=z,
            z_lengths=z_lengths,
            durations=batch.get("durations", None),
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint[
            "epoch"
        ]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        step = float(self.global_step)

        # Log current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr",
            lr,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/train_ldpm_loss",
            loss_dict.get("ldpm_loss", 0.0),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )

        # ----------------------------
        #  LDPM weighting schedule
        # ----------------------------
        ldpm_cap = 0.5  # max weight
        ldpm_warmup = 5_000  # steps before LDPM starts
        ldpm_ramp = 20_000  # ramp length
        if step < ldpm_warmup:
            w_ldpm = 0.0
        else:
            p = min(1.0, (step - ldpm_warmup) / ldpm_ramp)
            w_ldpm = ldpm_cap * (0.5 * (1 - math.cos(math.pi * p)))  # cosine ramp

        self.log(
            "w_ldpm",
            w_ldpm,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )

        # total_loss = sum(loss_dict.values())

        total_loss = (
            loss_dict["dur_loss"]
            + loss_dict["prior_loss"]
            + w_ldpm * loss_dict.get("ldpm_loss", 0.0)
        )

        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )

        # total_loss = sum(loss_dict.values())

        total_loss = (
            loss_dict["dur_loss"] + loss_dict["prior_loss"] + loss_dict["ldpm_loss"]
        )
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["x"].shape[0],
        )

        return total_loss

    def on_validation_end(self) -> None:
        # Wrap visualization and synthesis in a try/except to prevent hook errors from
        # crashing the training loop. Any exception here will be logged and skipped.
        if not self.trainer.is_global_zero:
            return

        try:
            one_batch = next(iter(self.trainer.val_dataloaders))

            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(min(2, one_batch["y"].shape[0])):
                    try:
                        y = one_batch["y"][i].unsqueeze(0).to(self.device)
                        image = plot_tensor(y.squeeze().cpu())
                        # Check if using wandb logger
                        if hasattr(self.logger.experiment, "log"):
                            self.logger.experiment.log(
                                {f"original/{i}": wandb.Image(image)},
                                step=self.current_epoch,
                            )
                        else:
                            # Fallback to tensorboard API
                            self.logger.experiment.add_image(
                                f"original/{i}",
                                image,
                                self.current_epoch,
                                dataformats="HWC",
                            )
                    except Exception:
                        log.exception("Failed to plot original sample %d", i)

            log.debug("Synthesising...")
            for i in range(2):
                try:
                    if one_batch["x"].shape[0] <= i:
                        break
                    x = one_batch["x"][i].unsqueeze(0).to(self.device)
                    x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    lang = one_batch["lang"][i].unsqueeze(0).to(self.device)
                    tone = one_batch["tone"][i].unsqueeze(0).to(self.device)
                    word_pos = one_batch["word_pos"][i].unsqueeze(0).to(self.device)
                    syllable_pos = (
                        one_batch["syllable_pos"][i].unsqueeze(0).to(self.device)
                    )
                    spk_embed = (
                        one_batch["spk_embed"][i].unsqueeze(0).to(self.device)
                        if one_batch.get("spk_embed") is not None
                        else None
                    )
                    output = self.synthesise(
                        x[:, : x_lengths.item()],
                        x_lengths,
                        lang[:, : x_lengths.item()],
                        tone[:, : x_lengths.item()],
                        word_pos[:, : x_lengths.item()],
                        syllable_pos[:, : x_lengths.item()],
                        spk_embed=spk_embed,
                        prompt_feat=y,
                        n_timesteps=10,
                    )
                    y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                    attn = output["attn"]

                    # helper to log an image and swallow per-image errors
                    def _log_image(name, img_arr):
                        try:
                            if hasattr(self.logger.experiment, "log"):
                                self.logger.experiment.log(
                                    {name: wandb.Image(img_arr)},
                                    step=self.current_epoch,
                                )
                            else:
                                self.logger.experiment.add_image(
                                    name, img_arr, self.current_epoch, dataformats="HWC"
                                )
                        except Exception:
                            log.exception("Failed to log image %s", name)

                    image_enc = plot_tensor(y_enc.squeeze().cpu())
                    _log_image(f"generated_enc/{i}", image_enc)

                    image_dec = plot_tensor(y_dec.squeeze().cpu())
                    _log_image(f"generated_dec/{i}", image_dec)

                    image_attn = plot_tensor(attn.squeeze().cpu())
                    _log_image(f"alignment/{i}", image_attn)
                except Exception:
                    log.exception("Failed to synthesise/log sample %d", i)

        except Exception:
            log.exception(
                "Unexpected error in on_validation_end, skipping visualizations"
            )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(
            {f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()}
        )
