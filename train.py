import argparse
import torch
from typing import Any, Dict, Tuple

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from hyperpyyaml import load_hyperpyyaml

from jyutvoice.utils import instantiators


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

    try:
        model.load_state_dict(
            torch.load("pretrained_models/checkpoint_0.pt"), strict=False
        )
        print("Loaded pretrained model weights from pretrained_models/checkpoint_0.pt")
    except Exception as e:
        pass

    print("Instantiating callbacks...")
    callbacks: list[Callback] = instantiators.instantiate_callbacks(
        cfg.get("callbacks")
    )

    print("Instantiating loggers...")
    logger: list[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    print(f"Instantiating trainer...")
    trainer: Trainer = cfg.get("trainer", Trainer(max_epochs=100))

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
