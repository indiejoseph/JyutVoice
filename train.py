#!/usr/bin/env python3
"""
Training script for JyutVoice TTS
"""
import sys
import torch
from hyperpyyaml import load_hyperpyyaml
import lightning


def main():
    print("=" * 80)
    print("JyutVoice TTS Training")
    print("=" * 80)

    # Load configuration
    print("\n📁 Loading configuration from configs/base.yaml...")
    with open("configs/base.yaml") as f:
        cfg = load_hyperpyyaml(f)
    print("✅ Configuration loaded!")

    # Get components from config
    datamodule = cfg["data"]
    model = cfg["tts"]
    trainer = cfg.get("trainer")

    print(f"\n📦 Components:")
    print(f"   Datamodule: {type(datamodule).__name__}")
    print(f"   Model: {type(model).__name__}")
    print(f"   Trainer: {type(trainer).__name__ if trainer else 'None'}")

    # Setup datamodule
    print("\n⚙️  Setting up datamodule...")
    try:
        datamodule.setup(stage="fit")
        print(f"✅ Datamodule setup complete!")
        print(f"   Train set size: {len(datamodule.trainset)}")
        print(f"   Valid set size: {len(datamodule.validset)}")
    except Exception as e:
        print(f"❌ Error during datamodule setup: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    model.to(device)

    # Test a forward pass
    print("\n🧪 Testing forward pass...")
    try:
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))

        print(f"   Batch keys: {batch.keys()}")
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                print(f"   - {key}: {batch[key].shape} {batch[key].dtype}")
            elif isinstance(batch[key], list):
                print(f"   - {key}: list of {len(batch[key])} items")
            else:
                print(f"   - {key}: {type(batch[key])}")

        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        with torch.no_grad():
            dur_loss, prior_loss, diff_loss, attn = model(
                x=batch["x"],
                x_lengths=batch["x_lengths"],
                y=batch["y"],
                y_lengths=batch["y_lengths"],
                lang=batch["lang"],
                tone=batch["tone"],
                word_pos=batch["word_pos"],
                syllable_pos=batch["syllable_pos"],
                spk_embed=batch["spk_embed"],
                decoder_h=batch["decoder_h"],
            )

        print(f"✅ Forward pass successful!")
        print(f"   Duration loss: {dur_loss.item():.6f}")
        print(f"   Prior loss: {prior_loss.item():.6f}")
        print(f"   Diff loss: {diff_loss.item():.6f}")
        print(f"   Total loss: {(dur_loss + prior_loss + diff_loss).item():.6f}")

    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Run training
    print("\n🚂 Starting training...")
    try:
        trainer.fit(model, datamodule=datamodule)
        print(f"✅ Training completed!")
        return True
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
