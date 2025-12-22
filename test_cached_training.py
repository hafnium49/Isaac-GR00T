#!/usr/bin/env python3
"""
Quick test of training loop with cached features.
Tests just a few steps to verify the pipeline works.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.data.dataset.cached_dataset import (
    CachedFeatureCollator,
    CachedWebDatasetIterable,
)


def test_cached_training(
    cached_path: str,
    model_path: str = "nvidia/GR00T-N1.6-3B",
    num_steps: int = 10,
    batch_size: int = 2,
):
    """Test the training loop with cached features."""
    print(f"\n{'='*60}")
    print("Testing Cached Training Pipeline")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Cached features: {cached_path}")
    print(f"Test steps: {num_steps}")
    print(f"Batch size: {batch_size}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model with frozen backbone
    print("\nLoading model (frozen backbone)...")
    model = Gr00tN1d6.from_pretrained(
        model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=False,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.train()

    # Verify backbone is frozen
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    action_trainable = sum(p.numel() for p in model.action_head.parameters() if p.requires_grad)
    print(f"Backbone trainable params: {backbone_trainable:,}")
    print(f"Action head trainable params: {action_trainable:,}")

    if backbone_trainable > 0:
        print("WARNING: Backbone has trainable parameters!")

    # Create dataset
    print("\nLoading cached features...")
    dataset = CachedWebDatasetIterable(
        cached_path=cached_path,
        shuffle_shards=False,
    )
    collator = CachedFeatureCollator()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collator,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # Training loop
    print(f"\nRunning {num_steps} training steps...")
    step_times = []
    losses = []

    data_iter = iter(loader)

    for step in range(num_steps):
        step_start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move to device
        inputs = batch["inputs"]
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, time={step_time*1000:.1f}ms")

    # Statistics
    avg_step_time = sum(step_times) / len(step_times)
    avg_loss = sum(losses) / len(losses)
    steps_per_sec = 1.0 / avg_step_time
    samples_per_sec = steps_per_sec * batch_size

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average step time: {avg_step_time*1000:.1f} ms")
    print(f"Throughput: {steps_per_sec:.1f} steps/sec")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")

    # Check if we achieved target
    target_steps_per_sec = 20
    if steps_per_sec >= target_steps_per_sec:
        print(f"\n✓ SUCCESS: Achieved >{target_steps_per_sec} steps/sec target!")
    else:
        print(f"\n⚠ Below target: {steps_per_sec:.1f} < {target_steps_per_sec} steps/sec")
        print("  (This may be due to small batch size or warmup effects)")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-path", type=str,
                       default="/home/h_fujiwara/projects/Isaac-GR00T/test_outputs/cached_features")
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    try:
        success = test_cached_training(
            cached_path=args.cached_path,
            model_path=args.model_path,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
