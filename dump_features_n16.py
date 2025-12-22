#!/usr/bin/env python3
"""
dump_features_n16.py - Offline Feature Caching for GR00T N1.6

Extracts Eagle backbone features from video frames and caches them to disk.
This eliminates the 2B-parameter forward pass during training, enabling
>20 steps/sec throughput (vs ~0.2 steps/sec with live backbone).

Usage:
    python dump_features_n16.py \
        --input-dir /path/to/lerobot_dataset \
        --output-dir /path/to/cached_features \
        --model-path nvidia/GR00T-N1.6-3B \
        --batch-size 32 \
        --format webdataset

Requirements:
    - Frozen backbone: tune_visual=False, tune_llm=False
    - Pre-exploded frames recommended (video_backend=preextracted)
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add Isaac-GR00T to path
ISAAC_GROOT_PATH = Path(__file__).parent
if str(ISAAC_GROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_PATH))

from gr00t.configs.data.embodiment_configs import get_embodiment_config
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor


def get_repo_commit() -> str:
    """Get the current git commit hash for cache validation."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ISAAC_GROOT_PATH,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def compute_config_hash(config: dict) -> str:
    """Compute a hash of the preprocessing config for validation."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class FeatureDumper:
    """
    Extracts and caches Eagle backbone features from a LeRobot dataset.

    The dumper:
    1. Loads video frames through the standard preprocessing pipeline
    2. Runs the Eagle backbone in inference mode
    3. Saves features to WebDataset shards or LMDB
    4. Includes metadata for cache validation
    """

    def __init__(
        self,
        model_path: str,
        input_dir: str,
        output_dir: str,
        embodiment_tag: str,
        modality_config_path: str | None = None,
        batch_size: int = 32,
        output_format: str = "webdataset",
        shard_size: int = 1000,
        video_backend: str = "preextracted",
        num_workers: int = 4,
        device: str = "cuda",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.device = device

        # Load modality config if provided
        if modality_config_path:
            self._load_modality_config(modality_config_path)

        # Get embodiment config
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.modality_config = get_embodiment_config(self.embodiment_tag)

        print(f"Loading model from {model_path}...")
        self.model = Gr00tN1d6.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        # Verify frozen backbone
        backbone_trainable = any(
            p.requires_grad for p in self.model.backbone.parameters()
        )
        if backbone_trainable:
            print("WARNING: Backbone has trainable parameters. Features may not be reusable!")

        # Create processor for data loading
        self.processor = Gr00tN1d6Processor.from_pretrained(model_path)
        self.processor.eval()  # Disable training augmentations

        # Create dataset
        print(f"Loading dataset from {input_dir}...")
        self.dataset = ShardedSingleStepDataset(
            dataset_path=input_dir,
            embodiment_tag=self.embodiment_tag,
            modality_configs=self.modality_config,
            video_backend=video_backend,
            shard_size=2048,
            episode_sampling_rate=1.0,  # Use all data
            seed=42,
        )
        self.dataset.processor = self.processor

        # Set dataset statistics
        stats = self.dataset.get_dataset_statistics()
        self.processor.set_statistics({embodiment_tag: stats})

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata for cache validation
        self.metadata = {
            "backbone_ckpt": model_path,
            "repo_commit": get_repo_commit(),
            "preprocessing": {
                "video_backend": video_backend,
                "image_size": getattr(self.processor, "image_target_size", None),
            },
            "layer": "backbone_output_post_ln",
            "dtype": "fp16",
            "embodiment_tag": embodiment_tag,
            "input_dir": str(input_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.metadata["config_hash"] = compute_config_hash(self.metadata)

    def _load_modality_config(self, path: str):
        """Load custom modality config from a Python file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("modality_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate batch for model input."""
        return self.processor.collator(batch)

    @torch.inference_mode()
    def extract_features(self, batch: dict) -> dict:
        """
        Extract backbone features from a batch.

        Returns:
            dict with:
                - features: [B, seq_len, 2048] fp16
                - attention_mask: [B, seq_len]
                - image_mask: [B, seq_len]
        """
        # Prepare inputs
        inputs = batch["inputs"]
        backbone_inputs, action_inputs = self.model.prepare_input(inputs)

        # Run backbone
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_output = self.model.backbone(backbone_inputs)

        # Extract features
        features = backbone_output["backbone_features"].half().cpu()
        attention_mask = backbone_output["backbone_attention_mask"].cpu()
        image_mask = backbone_output["image_mask"].cpu()

        # Extract action inputs for later training
        return {
            "features": features,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "state": inputs["state"].cpu() if "state" in inputs else None,
            "action": inputs["action"].cpu() if "action" in inputs else None,
            "action_mask": inputs["action_mask"].cpu() if "action_mask" in inputs else None,
            "embodiment_id": inputs["embodiment_id"] if "embodiment_id" in inputs else None,
        }

    def _write_webdataset_sample(self, tar: tarfile.TarFile, idx: int, sample: dict):
        """Write a single sample to a WebDataset tar file."""
        key = f"{idx:08d}"

        # Save features as fp16 tensor
        features_buf = BytesIO()
        torch.save(sample["features"].half(), features_buf)
        features_buf.seek(0)
        features_info = tarfile.TarInfo(name=f"{key}.features.pt")
        features_info.size = len(features_buf.getvalue())
        tar.addfile(features_info, features_buf)

        # Save attention mask
        mask_buf = BytesIO()
        torch.save(sample["attention_mask"], mask_buf)
        mask_buf.seek(0)
        mask_info = tarfile.TarInfo(name=f"{key}.attention_mask.pt")
        mask_info.size = len(mask_buf.getvalue())
        tar.addfile(mask_info, mask_buf)

        # Save image mask
        img_mask_buf = BytesIO()
        torch.save(sample["image_mask"], img_mask_buf)
        img_mask_buf.seek(0)
        img_mask_info = tarfile.TarInfo(name=f"{key}.image_mask.pt")
        img_mask_info.size = len(img_mask_buf.getvalue())
        tar.addfile(img_mask_info, img_mask_buf)

        # Save state
        if sample["state"] is not None:
            state_buf = BytesIO()
            torch.save(sample["state"], state_buf)
            state_buf.seek(0)
            state_info = tarfile.TarInfo(name=f"{key}.state.pt")
            state_info.size = len(state_buf.getvalue())
            tar.addfile(state_info, state_buf)

        # Save action
        if sample["action"] is not None:
            action_buf = BytesIO()
            torch.save(sample["action"], action_buf)
            action_buf.seek(0)
            action_info = tarfile.TarInfo(name=f"{key}.action.pt")
            action_info.size = len(action_buf.getvalue())
            tar.addfile(action_info, action_buf)

        # Save action mask
        if sample["action_mask"] is not None:
            action_mask_buf = BytesIO()
            torch.save(sample["action_mask"], action_mask_buf)
            action_mask_buf.seek(0)
            action_mask_info = tarfile.TarInfo(name=f"{key}.action_mask.pt")
            action_mask_info.size = len(action_mask_buf.getvalue())
            tar.addfile(action_mask_info, action_mask_buf)

        # Save embodiment_id as JSON
        if sample["embodiment_id"] is not None:
            emb_data = json.dumps({"embodiment_id": int(sample["embodiment_id"])}).encode()
            emb_buf = BytesIO(emb_data)
            emb_info = tarfile.TarInfo(name=f"{key}.json")
            emb_info.size = len(emb_data)
            tar.addfile(emb_info, emb_buf)

    def dump_webdataset(self):
        """Dump features to WebDataset format (tar shards)."""
        print(f"Dumping features to WebDataset format at {self.output_dir}")

        total_samples = sum(self.dataset.shard_lengths)
        print(f"Total samples: {total_samples}")
        print(f"Shard size: {self.shard_size}")

        # Save metadata
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")

        current_tar = None
        current_shard_idx = 0
        samples_in_shard = 0
        global_idx = 0

        # Process all dataset shards
        pbar = tqdm(total=total_samples, desc="Extracting features")

        for shard_idx in range(len(self.dataset)):
            # Load shard data
            shard_data = self.dataset.get_shard(shard_idx)

            # Create batches
            for batch_start in range(0, len(shard_data), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(shard_data))
                batch = shard_data[batch_start:batch_end]

                # Collate and extract
                collated = self.collate_fn(batch)
                extracted = self.extract_features(collated)

                # Write samples
                batch_size = extracted["features"].shape[0]
                for i in range(batch_size):
                    # Create new shard if needed
                    if current_tar is None or samples_in_shard >= self.shard_size:
                        if current_tar is not None:
                            current_tar.close()
                        tar_path = self.output_dir / f"shard-{current_shard_idx:06d}.tar"
                        current_tar = tarfile.open(tar_path, "w")
                        current_shard_idx += 1
                        samples_in_shard = 0

                    # Extract single sample
                    sample = {
                        "features": extracted["features"][i],
                        "attention_mask": extracted["attention_mask"][i],
                        "image_mask": extracted["image_mask"][i],
                        "state": extracted["state"][i] if extracted["state"] is not None else None,
                        "action": extracted["action"][i] if extracted["action"] is not None else None,
                        "action_mask": extracted["action_mask"][i] if extracted["action_mask"] is not None else None,
                        "embodiment_id": extracted["embodiment_id"] if extracted["embodiment_id"] is not None else None,
                    }

                    self._write_webdataset_sample(current_tar, global_idx, sample)
                    global_idx += 1
                    samples_in_shard += 1
                    pbar.update(1)

                # Free memory
                del extracted
                torch.cuda.empty_cache()

            # Free shard memory
            del shard_data
            gc.collect()

        if current_tar is not None:
            current_tar.close()

        pbar.close()

        # Write index file
        index = {
            "num_samples": global_idx,
            "num_shards": current_shard_idx,
            "shard_size": self.shard_size,
            "metadata": self.metadata,
        }
        index_path = self.output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"\nDone! Wrote {global_idx} samples to {current_shard_idx} shards")
        print(f"Index: {index_path}")

    def dump_lmdb(self):
        """Dump features to LMDB format."""
        try:
            import lmdb
        except ImportError:
            raise ImportError("LMDB not installed. Run: pip install lmdb")

        print(f"Dumping features to LMDB format at {self.output_dir}")

        total_samples = sum(self.dataset.shard_lengths)
        print(f"Total samples: {total_samples}")

        # Estimate map size (1MB per sample, 50% overhead)
        map_size = int(total_samples * 1.5 * 1024 * 1024)

        # Save metadata separately
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        lmdb_path = self.output_dir / "features.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

        global_idx = 0
        pbar = tqdm(total=total_samples, desc="Extracting features")

        for shard_idx in range(len(self.dataset)):
            shard_data = self.dataset.get_shard(shard_idx)

            for batch_start in range(0, len(shard_data), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(shard_data))
                batch = shard_data[batch_start:batch_end]

                collated = self.collate_fn(batch)
                extracted = self.extract_features(collated)

                batch_size = extracted["features"].shape[0]

                with env.begin(write=True) as txn:
                    for i in range(batch_size):
                        key = f"{global_idx:08d}".encode()

                        sample = {
                            "features": extracted["features"][i].numpy(),
                            "attention_mask": extracted["attention_mask"][i].numpy(),
                            "image_mask": extracted["image_mask"][i].numpy(),
                            "state": extracted["state"][i].numpy() if extracted["state"] is not None else None,
                            "action": extracted["action"][i].numpy() if extracted["action"] is not None else None,
                            "action_mask": extracted["action_mask"][i].numpy() if extracted["action_mask"] is not None else None,
                            "embodiment_id": int(extracted["embodiment_id"]) if extracted["embodiment_id"] is not None else None,
                        }

                        # Serialize with pickle
                        import pickle
                        value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
                        txn.put(key, value)

                        global_idx += 1
                        pbar.update(1)

                del extracted
                torch.cuda.empty_cache()

            del shard_data
            gc.collect()

        env.close()
        pbar.close()

        # Write index file
        index = {
            "num_samples": global_idx,
            "format": "lmdb",
            "metadata": self.metadata,
        }
        index_path = self.output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"\nDone! Wrote {global_idx} samples to LMDB")

    def run(self):
        """Run the feature dumping process."""
        if self.output_format == "webdataset":
            self.dump_webdataset()
        elif self.output_format == "lmdb":
            self.dump_lmdb()
        else:
            raise ValueError(f"Unknown format: {self.output_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump Eagle backbone features for GR00T N1.6 training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to LeRobot format dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output cached features",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/GR00T-N1.6-3B",
        help="Path to GR00T N1.6 model",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag for the dataset",
    )
    parser.add_argument(
        "--modality-config",
        type=str,
        default=None,
        help="Path to custom modality config Python file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["webdataset", "lmdb"],
        default="webdataset",
        help="Output format",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per WebDataset shard",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="preextracted",
        help="Video backend (preextracted, ffmpeg, decord)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GR00T N1.6 Feature Dumper")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Batch:  {args.batch_size}")
    print("=" * 60)

    dumper = FeatureDumper(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        embodiment_tag=args.embodiment_tag,
        modality_config_path=args.modality_config,
        batch_size=args.batch_size,
        output_format=args.format,
        shard_size=args.shard_size,
        video_backend=args.video_backend,
        num_workers=args.num_workers,
        device=args.device,
    )

    start_time = time.time()
    dumper.run()
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
