#!/usr/bin/env python3
"""
Test script to verify cached feature dataset loading.
"""
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from gr00t.data.dataset.cached_dataset import (
    CachedFeatureDataset,
    CachedFeatureCollator,
    CachedWebDatasetIterable,
    get_cached_dataloader,
)


def test_cached_dataset(cached_path: str):
    """Test the cached feature dataset loading."""
    print(f"\n{'='*60}")
    print("Testing CachedFeatureDataset (random access)")
    print(f"{'='*60}")

    try:
        dataset = CachedFeatureDataset(cached_path=cached_path)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Format: {dataset.format}")
        print(f"  - Metadata: {dataset.metadata}")

        # Load first sample
        sample = dataset[0]
        print(f"\n✓ First sample loaded:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {value}")

        # Test random access speed
        print(f"\n  Testing random access speed (100 samples)...")
        start = time.time()
        for i in range(min(100, len(dataset))):
            _ = dataset[i]
        elapsed = time.time() - start
        print(f"  - Time: {elapsed:.3f}s ({100/elapsed:.1f} samples/sec)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iterable_dataset(cached_path: str):
    """Test the iterable dataset streaming."""
    print(f"\n{'='*60}")
    print("Testing CachedWebDatasetIterable (streaming)")
    print(f"{'='*60}")

    try:
        dataset = CachedWebDatasetIterable(
            cached_path=cached_path,
            shuffle_shards=False,
        )
        print(f"✓ Iterable dataset created")
        print(f"  - Number of samples: {dataset.num_samples}")

        # Stream a few samples
        print(f"\n  Streaming first 5 samples...")
        count = 0
        for sample in dataset:
            print(f"  - Sample {count}:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"      {key}: {value}")
            count += 1
            if count >= 5:
                break

        # Test streaming speed
        print(f"\n  Testing streaming speed (all samples)...")
        dataset = CachedWebDatasetIterable(cached_path=cached_path, shuffle_shards=False)
        start = time.time()
        total = 0
        for sample in dataset:
            total += 1
        elapsed = time.time() - start
        print(f"  - Streamed {total} samples in {elapsed:.3f}s ({total/elapsed:.1f} samples/sec)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(cached_path: str):
    """Test the full DataLoader pipeline."""
    print(f"\n{'='*60}")
    print("Testing DataLoader Pipeline")
    print(f"{'='*60}")

    try:
        # Test with iterable dataset
        dataloader = get_cached_dataloader(
            cached_path=cached_path,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            shuffle=False,
            use_iterable=True,
        )
        print(f"✓ DataLoader created")

        # Get first batch
        batch = next(iter(dataloader))
        print(f"\n✓ First batch loaded:")
        inputs = batch["inputs"]
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {value}")

        # Test batch loading speed
        print(f"\n  Testing batch loading speed...")
        start = time.time()
        total_samples = 0
        for batch in dataloader:
            total_samples += batch["inputs"]["cached_features"].shape[0]
        elapsed = time.time() - start
        print(f"  - Loaded {total_samples} samples in {elapsed:.3f}s ({total_samples/elapsed:.1f} samples/sec)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collator():
    """Test the collator function."""
    print(f"\n{'='*60}")
    print("Testing CachedFeatureCollator")
    print(f"{'='*60}")

    try:
        collator = CachedFeatureCollator()

        # Create fake samples
        samples = [
            {
                "features": torch.randn(256, 2048, dtype=torch.float16),
                "attention_mask": torch.ones(256, dtype=torch.bool),
                "image_mask": torch.ones(256, dtype=torch.bool),
                "state": torch.randn(1, 29),
                "action": torch.randn(16, 29),
                "action_mask": torch.ones(16, 29, dtype=torch.bool),
                "embodiment_id": 0,
            },
            {
                "features": torch.randn(256, 2048, dtype=torch.float16),
                "attention_mask": torch.ones(256, dtype=torch.bool),
                "image_mask": torch.ones(256, dtype=torch.bool),
                "state": torch.randn(1, 29),
                "action": torch.randn(16, 29),
                "action_mask": torch.ones(16, 29, dtype=torch.bool),
                "embodiment_id": 1,
            },
        ]

        batch = collator(samples)
        print(f"✓ Collator works correctly")
        print(f"  Output keys: {batch.keys()}")
        inputs = batch["inputs"]
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-path", type=str, default="/tmp/test_cached_features",
                       help="Path to cached features directory")
    args = parser.parse_args()

    print(f"Testing cached dataset loading from: {args.cached_path}")

    results = []

    # Test collator (no data needed)
    results.append(("Collator", test_collator()))

    # Test dataset loading
    results.append(("CachedFeatureDataset", test_cached_dataset(args.cached_path)))
    results.append(("CachedWebDatasetIterable", test_iterable_dataset(args.cached_path)))
    results.append(("DataLoader", test_dataloader(args.cached_path)))

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
