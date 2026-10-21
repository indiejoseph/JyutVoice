import argparse
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from hyperpyyaml import load_hyperpyyaml

from jyutvoice.utils import instantiators


def attach_callbacks(trainer: Trainer, callbacks: list) -> None:
    """Give the configured callbacks to a trainer that hyperpyyaml already built without them.

    A configured ModelCheckpoint replaces the one Lightning adds by default, so checkpoints are not written twice.
    """
    if not callbacks:
        return
    if any(isinstance(c, ModelCheckpoint) for c in callbacks):
        trainer.callbacks[:] = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
    trainer.callbacks.extend(callbacks)


def load_init_weights(model: LightningModule, path: str, prefixes=None) -> None:
    """Warm start from a checkpoint: weights only, the optimizer and the epoch counter start fresh.

    `prefixes` (e.g. ["encoder.", "spk_embed_affine_layer."]) loads only the checkpoint keys that start with one of them, for a model whose other
    modules differ from the checkpoint's (a different duration predictor); every model parameter under those prefixes must then be in the checkpoint.

    Unlike `ckpt_path` (which resumes everything) this works when the model gained parameters since the checkpoint was written, e.g.
    `encoder.bert_proj.*` (the model's `on_load_checkpoint` starts those at zero). Accepts a Lightning checkpoint or a bare state dict.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # Lightning checkpoints pickle the hyper-parameters: use trusted files
    if "state_dict" not in ckpt:
        ckpt = {"state_dict": ckpt}
    ckpt["epoch"] = 0
    model.on_load_checkpoint(ckpt)
    if prefixes:
        prefixes = tuple(prefixes)
        state = {k: v for k, v in ckpt["state_dict"].items() if k.startswith(prefixes)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k.startswith(prefixes)]
        assert not missing and not unexpected, (missing, unexpected)
        return
    model.load_state_dict(ckpt["state_dict"])


def train(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    :param cfg: Configuration dictionary loaded from hyperpyyaml config file.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg["seed"], workers=True)

    print(f"Instantiating datamodule...")
    datamodule: LightningDataModule = cfg["data"]

    print(f"Instantiating model...")
    model: LightningModule = cfg["tts"]
    if cfg.get("init_ckpt"):
        print(f"Initialising weights from {cfg['init_ckpt']}")
        load_init_weights(model, cfg["init_ckpt"], cfg.get("init_ckpt_prefixes"))

    print("Instantiating callbacks...")
    callbacks: list[Callback] = instantiators.instantiate_callbacks(
        cfg.get("callbacks")
    )

    print("Instantiating loggers...")
    logger: list[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    print(f"Instantiating trainer...")
    trainer: Trainer = cfg.get("trainer", Trainer(max_epochs=100))
    attach_callbacks(trainer, callbacks)

    # Attach logger to trainer if instantiated
    if logger:
        trainer.logger = logger if len(logger) > 1 else logger[0]

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    print("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        print("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def main():
    """Main entry point for training using hyperpyyaml configuration."""
    parser = argparse.ArgumentParser(
        description="Train Jyutvoice TTS model with hyperpyyaml configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to hyperpyyaml configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run only validation, not training",
    )
    args = parser.parse_args()

    # Load configuration from hyperpyyaml
    with open(args.config) as f:
        cfg = load_hyperpyyaml(f)

    # Set device
    if args.device:
        cfg["device"] = args.device

    if args.validate_only:
        # Run only validation
        validate_only(cfg)
    else:
        # Train the model
        metric_dict, object_dict = train(cfg)

        print(f"Training completed!")
        print(f"Metrics: {metric_dict}")

    return {}


def validate_only(cfg: Dict[str, Any]):
    """Run only validation."""
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg["seed"], workers=True)

    print(f"Instantiating datamodule...")
    datamodule: LightningDataModule = cfg["data"]

    print(f"Instantiating model...")
    model: LightningModule = cfg["tts"]
    model.eval()

    print(f"Instantiating trainer...")
    trainer: Trainer = cfg.get("trainer", Trainer(max_epochs=100))

    # Load checkpoint if provided
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}")
        # trainer.validate will load the checkpoint automatically if ckpt_path is provided

    print("Starting validation!")
    trainer.validate(model, datamodule, ckpt_path=ckpt_path)

    print("Validation completed!")


if __name__ == "__main__":
    main()
