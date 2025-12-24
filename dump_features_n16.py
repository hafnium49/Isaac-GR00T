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
import logging
import os
import shutil
import signal
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


def _patch_eagle3_vl_config():
    """
    Patch Eagle3_VLConfig for compatibility with newer transformers versions.

    The _attn_implementation_autoset attribute was added in later transformers
    versions. This patch monkey-patches the class at runtime.

    Also patches FlashAttention2 check to gracefully fall back to eager.
    """
    from transformers import PretrainedConfig, modeling_utils

    # Store original to_dict method
    _original_to_dict = PretrainedConfig.to_dict

    def _patched_to_dict(self):
        """Patched to_dict that handles missing _attn_implementation_autoset."""
        result = _original_to_dict(self)
        # Ensure _attn_implementation_autoset is present
        if '_attn_implementation_autoset' not in result:
            result['_attn_implementation_autoset'] = getattr(
                self, '_attn_implementation_autoset', True
            )
        return result

    # Also patch __getattribute__ to handle the attribute access
    _original_getattribute = PretrainedConfig.__getattribute__

    def _patched_getattribute(self, name):
        if name == '_attn_implementation_autoset':
            try:
                return _original_getattribute(self, name)
            except AttributeError:
                return True
        return _original_getattribute(self, name)

    PretrainedConfig.__getattribute__ = _patched_getattribute

    # Check if flash_attn is available
    flash_attn_available = False
    try:
        import flash_attn  # noqa: F401
        flash_attn_available = True
    except ImportError:
        pass

    # If flash_attn not available, patch to force eager attention
    if not flash_attn_available and hasattr(modeling_utils, 'PreTrainedModel'):

        def _patched_check(self, *args, **kwargs):
            """Force eager attention when flash_attn is not available."""
            # Simply return 'eager' - don't bother with original check
            # This avoids any flash attention verification
            if hasattr(self.config, '_attn_implementation'):
                self.config._attn_implementation = 'eager'
            if hasattr(self.config, '_attn_implementation_autoset'):
                self.config._attn_implementation_autoset = False
            return 'eager'

        modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = _patched_check

    print("Applied Eagle3 VL compatibility patch")


# Apply patch before importing any model code
_patch_eagle3_vl_config()

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
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


class LoggingManager:
    """Manages file and console logging with performance metrics."""

    def __init__(self, output_dir: Path, log_level: str = "INFO", log_file: str | None = None):
        self.output_dir = output_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create logger
        self.logger = logging.getLogger("FeatureDumper")
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.handlers.clear()  # Remove any existing handlers

        # Log file path
        if log_file:
            self.log_path = Path(log_file)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = output_dir / f"dumper_{timestamp}.log"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File handler (DEBUG level - captures everything)
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # Console handler (user-specified level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_format = logging.Formatter("%(levelname)-5s | %(message)s")
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log_config(self, config: dict):
        """Log configuration settings."""
        self.info("Configuration:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")

    def log_metrics(self, samples_processed: int, total_samples: int, elapsed: float, shards: int):
        """Log periodic performance metrics."""
        pct = (samples_processed / total_samples) * 100 if total_samples > 0 else 0
        throughput = samples_processed / elapsed if elapsed > 0 else 0
        self.info(f"Progress: {samples_processed:,}/{total_samples:,} ({pct:.1f}%) | {shards} shards | {throughput:.1f} samples/sec")

    def log_final_summary(self, total_samples: int, total_shards: int, total_time: float, peak_throughput: float):
        """Log completion summary."""
        avg_throughput = total_samples / total_time if total_time > 0 else 0
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        self.info("=" * 60)
        self.info("FEATURE DUMPER COMPLETE")
        self.info("=" * 60)
        self.info(f"Total samples:    {total_samples:,}")
        self.info(f"Total shards:     {total_shards:,}")
        self.info(f"Total time:       {hours}h {minutes}m {seconds}s")
        self.info(f"Avg throughput:   {avg_throughput:.2f} samples/sec")
        self.info(f"Peak throughput:  {peak_throughput:.2f} samples/sec")
        self.info(f"Log file:         {self.log_path}")
        self.info("=" * 60)


class PerformanceTracker:
    """Tracks timing and throughput metrics."""

    def __init__(self):
        self.start_time: float | None = None
        self.batch_times: list[float] = []
        self.shard_times: list[float] = []
        self.samples_processed: int = 0
        self._batch_start: float | None = None
        self._shard_start: float | None = None
        self._window_samples: list[tuple[float, int]] = []  # (timestamp, cumulative_samples)

    def start(self):
        """Start overall timing."""
        self.start_time = time.time()
        self._shard_start = self.start_time

    def start_batch(self):
        """Start timing a batch."""
        self._batch_start = time.time()

    def end_batch(self, batch_size: int):
        """End batch timing and record metrics."""
        if self._batch_start is not None:
            elapsed = time.time() - self._batch_start
            self.batch_times.append(elapsed)
            self.samples_processed += batch_size
            self._window_samples.append((time.time(), self.samples_processed))
            # Keep only last 100 samples for windowed throughput
            if len(self._window_samples) > 100:
                self._window_samples.pop(0)

    def end_shard(self):
        """End shard timing and record metrics."""
        if self._shard_start is not None:
            elapsed = time.time() - self._shard_start
            self.shard_times.append(elapsed)
            self._shard_start = time.time()

    def get_elapsed(self) -> float:
        """Get total elapsed time."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_throughput(self) -> float:
        """Get overall average throughput (samples/sec)."""
        elapsed = self.get_elapsed()
        if elapsed <= 0:
            return 0.0
        return self.samples_processed / elapsed

    def get_recent_throughput(self) -> float:
        """Get recent throughput from sliding window."""
        if len(self._window_samples) < 2:
            return self.get_throughput()
        oldest = self._window_samples[0]
        newest = self._window_samples[-1]
        time_diff = newest[0] - oldest[0]
        sample_diff = newest[1] - oldest[1]
        if time_diff <= 0:
            return 0.0
        return sample_diff / time_diff

    def get_peak_throughput(self) -> float:
        """Get peak throughput from batch times."""
        if not self.batch_times:
            return 0.0
        # Estimate samples per batch from total
        if len(self.batch_times) > 0:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            if avg_batch_time > 0:
                # Use recent throughput as approximation
                return max(self.get_recent_throughput(), self.get_throughput())
        return self.get_throughput()

    def get_summary(self) -> dict:
        """Get performance summary."""
        return {
            "total_time": self.get_elapsed(),
            "samples_processed": self.samples_processed,
            "avg_throughput": self.get_throughput(),
            "peak_throughput": self.get_peak_throughput(),
            "num_batches": len(self.batch_times),
            "num_shards": len(self.shard_times),
        }


class GracefulShutdown:
    """Handles graceful shutdown on SIGINT/SIGTERM."""

    def __init__(self):
        self._shutdown_requested = False
        self._original_sigint = None
        self._original_sigterm = None

    def register(self):
        """Register signal handlers."""
        self._original_sigint = signal.signal(signal.SIGINT, self._handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        """Signal handler that sets shutdown flag."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[{sig_name}] Graceful shutdown requested. Finishing current shard...")
        self._shutdown_requested = True

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def restore(self):
        """Restore original signal handlers."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)


class CheckpointManager:
    """Manages checkpoint save/load for resumable feature extraction."""

    CHECKPOINT_FILE = ".dumper_checkpoint.json"
    VERSION = 1

    def __init__(self, output_dir: Path, config_hash: str, logger: LoggingManager | None = None):
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / self.CHECKPOINT_FILE
        self.config_hash = config_hash
        self._created_at = None
        self._logger = logger

    def _log(self, level: str, msg: str):
        """Log message using logger if available, else print."""
        if self._logger:
            getattr(self._logger, level)(msg)
        else:
            print(f"{level.upper()}: {msg}" if level != "info" else msg)

    def checkpoint_exists(self) -> bool:
        """Check if a valid checkpoint file exists."""
        return self.checkpoint_path.exists()

    def load(self) -> dict | None:
        """Load and validate checkpoint. Returns None if invalid."""
        if not self.checkpoint_exists():
            return None

        with open(self.checkpoint_path, "r") as f:
            ckpt = json.load(f)

        # Validate version
        if ckpt.get("version") != self.VERSION:
            self._log("warning", f"Checkpoint version mismatch (got {ckpt.get('version')}, expected {self.VERSION})")
            return None

        # Validate config hash
        if ckpt.get("config_hash") != self.config_hash:
            self._log("warning", "Config hash mismatch. Cannot resume.")
            self._log("warning", f"  Checkpoint: {ckpt.get('config_hash')}")
            self._log("warning", f"  Current:    {self.config_hash}")
            return None

        return ckpt

    def save(self, state: dict):
        """Save checkpoint state atomically."""
        if self._created_at is None:
            self._created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        state["version"] = self.VERSION
        state["config_hash"] = self.config_hash
        state["created_at"] = self._created_at
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, "item"):  # numpy scalar types
                return obj.item()
            return obj

        state = convert_numpy(state)

        # Write atomically using temp file
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        tmp_path.rename(self.checkpoint_path)

    def delete(self):
        """Remove checkpoint file on successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def validate_existing_shards(self, expected_count: int) -> bool:
        """Verify that expected shard files exist and are valid."""
        for i in range(expected_count):
            shard_path = self.output_dir / f"shard-{i:06d}.tar"
            if not shard_path.exists():
                self._log("error", f"Missing shard file: {shard_path}")
                return False
            # Verify TAR is readable
            try:
                with tarfile.open(shard_path, "r") as tar:
                    _ = tar.getnames()
            except Exception as e:
                self._log("error", f"Corrupt shard file {shard_path}: {e}")
                return False
        return True

    def cleanup_incomplete_shards(self, last_complete: int):
        """Remove any shard files beyond the last complete one."""
        for shard_path in self.output_dir.glob("shard-*.tar"):
            try:
                idx = int(shard_path.stem.split("-")[1])
                if idx >= last_complete:
                    self._log("info", f"Removing incomplete shard: {shard_path}")
                    shard_path.unlink()
            except (ValueError, IndexError):
                pass


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
        checkpoint_interval: int = 5,
        log_level: str = "INFO",
        log_file: str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.shutdown_handler = GracefulShutdown()

        # Initialize logging and performance tracking
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = LoggingManager(self.output_dir, log_level, log_file)
        self.perf = PerformanceTracker()

        # Load modality config if provided
        if modality_config_path:
            self._load_modality_config(modality_config_path)

        # Get embodiment config
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.modality_config = MODALITY_CONFIGS.get(self.embodiment_tag.value)
        if self.modality_config is None:
            raise ValueError(f"No modality config found for embodiment tag: {embodiment_tag}")

        self.log.info(f"Loading model from {model_path}...")
        self.model = Gr00tN1d6.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention (no FlashAttention)
        )
        self.model.to(device)
        self.model.eval()
        self.log.info(f"Model loaded successfully")

        # Verify frozen backbone
        backbone_trainable = any(
            p.requires_grad for p in self.model.backbone.parameters()
        )
        if backbone_trainable:
            self.log.warning("Backbone has trainable parameters. Features may not be reusable!")

        # Create processor for data loading
        # Pass modality_config so the processor knows about our custom embodiment
        self.processor = Gr00tN1d6Processor.from_pretrained(
            model_path,
            modality_configs={embodiment_tag: self.modality_config}
        )
        self.processor.eval()  # Disable training augmentations

        # Create dataset
        self.log.info(f"Loading dataset from {input_dir}...")
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

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            config_hash=self.metadata["config_hash"],
            logger=self.log,
        )

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

    def _compute_resume_state(self, checkpoint: dict) -> tuple[int, int, int, int]:
        """
        Compute where to resume processing from checkpoint.

        Returns:
            (dataset_shard_idx, batch_start, output_shard_idx, global_idx)
        """
        last_global_idx = checkpoint["last_complete_global_idx"]
        output_shard_idx = checkpoint["output_state"]["current_shard_idx"]

        # Calculate which input dataset shard contains this sample
        cumulative = 0
        dataset_shard_idx = 0
        batch_start = 0

        for shard_idx, length in enumerate(self.dataset.shard_lengths):
            if cumulative + length > last_global_idx:
                dataset_shard_idx = shard_idx
                # Offset within this shard
                offset_in_shard = last_global_idx - cumulative
                # Round down to batch boundary
                batch_start = (offset_in_shard // self.batch_size) * self.batch_size
                break
            cumulative += length
        else:
            # All samples processed
            dataset_shard_idx = len(self.dataset)
            batch_start = 0

        return (dataset_shard_idx, batch_start, output_shard_idx, last_global_idx)

    def _save_checkpoint(
        self,
        global_idx: int,
        current_shard_idx: int,
        dataset_shard_idx: int,
        batch_start: int,
        total_samples: int,
    ):
        """Save checkpoint state."""
        state = {
            "total_samples": total_samples,
            "completed_shards": current_shard_idx,
            "last_complete_global_idx": global_idx,
            "input_dataset_state": {
                "dataset_shard_idx": dataset_shard_idx,
                "batch_start_within_shard": batch_start,
            },
            "output_state": {
                "current_shard_idx": current_shard_idx,
                "samples_in_shard": 0,
            },
        }
        self.checkpoint_manager.save(state)
        self.log.info(f"Checkpoint saved: {global_idx}/{total_samples} samples, {current_shard_idx} shards")

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
            emb_id = sample["embodiment_id"]
            # Handle tensor conversion
            if hasattr(emb_id, 'item'):
                emb_id = emb_id.item()
            emb_data = json.dumps({"embodiment_id": int(emb_id)}).encode()
            emb_buf = BytesIO(emb_data)
            emb_info = tarfile.TarInfo(name=f"{key}.json")
            emb_info.size = len(emb_data)
            tar.addfile(emb_info, emb_buf)

    def dump_webdataset(self, resume_checkpoint: dict | None = None):
        """Dump features to WebDataset format (tar shards) with resumption support."""
        # Register signal handlers for graceful shutdown
        self.shutdown_handler.register()

        total_samples = sum(self.dataset.shard_lengths)
        self.log.info(f"Dumping features to WebDataset format at {self.output_dir}")
        self.log.info(f"Total samples: {total_samples}")
        self.log.info(f"Shard size: {self.shard_size}")

        # Initialize state (fresh start or resume)
        if resume_checkpoint is not None:
            start_dataset_shard, start_batch, current_shard_idx, global_idx = \
                self._compute_resume_state(resume_checkpoint)
            self.log.info(f"Resuming from: global_idx={global_idx}, output_shard={current_shard_idx}")
        else:
            start_dataset_shard = 0
            start_batch = 0
            current_shard_idx = 0
            global_idx = 0

            # Save metadata for fresh run
            meta_path = self.output_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            self.log.info(f"Saved metadata to {meta_path}")

        current_tar = None
        samples_in_shard = 0
        shards_since_checkpoint = 0
        last_completed_shard_global_idx = global_idx

        # Start performance tracking
        self.perf.start()

        # Process all dataset shards
        pbar = tqdm(total=total_samples, initial=global_idx, desc="Extracting features")

        try:
            for shard_idx in range(start_dataset_shard, len(self.dataset)):
                # Load shard data
                shard_data = self.dataset.get_shard(shard_idx)

                # Determine batch start (for resume)
                batch_start_offset = start_batch if shard_idx == start_dataset_shard else 0

                # Create batches
                for batch_start in range(batch_start_offset, len(shard_data), self.batch_size):
                    # Check for shutdown request
                    if self.shutdown_handler.shutdown_requested:
                        raise KeyboardInterrupt("Graceful shutdown requested")

                    batch_end = min(batch_start + self.batch_size, len(shard_data))
                    batch = shard_data[batch_start:batch_end]

                    # Track batch timing
                    self.perf.start_batch()

                    # Collate and extract
                    collated = self.collate_fn(batch)
                    extracted = self.extract_features(collated)

                    # Write samples
                    batch_size_actual = extracted["features"].shape[0]
                    for i in range(batch_size_actual):
                        # Create new shard if needed
                        if current_tar is None or samples_in_shard >= self.shard_size:
                            if current_tar is not None:
                                current_tar.close()
                                self.perf.end_shard()
                                shards_since_checkpoint += 1

                                # Update tracking for checkpoint
                                last_completed_shard_global_idx = global_idx

                                # Save checkpoint and log metrics periodically
                                if shards_since_checkpoint >= self.checkpoint_interval:
                                    self._save_checkpoint(
                                        global_idx=global_idx,
                                        current_shard_idx=current_shard_idx,
                                        dataset_shard_idx=shard_idx,
                                        batch_start=batch_start,
                                        total_samples=total_samples,
                                    )
                                    # Log performance metrics
                                    self.log.log_metrics(
                                        samples_processed=global_idx,
                                        total_samples=total_samples,
                                        elapsed=self.perf.get_elapsed(),
                                        shards=current_shard_idx,
                                    )
                                    shards_since_checkpoint = 0

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
                            "embodiment_id": extracted["embodiment_id"][i] if extracted["embodiment_id"] is not None else None,
                        }

                        self._write_webdataset_sample(current_tar, global_idx, sample)
                        global_idx += 1
                        samples_in_shard += 1
                        pbar.update(1)

                    # End batch timing
                    self.perf.end_batch(batch_size_actual)

                    # Free memory
                    del extracted
                    torch.cuda.empty_cache()

                # Free shard memory
                del shard_data
                gc.collect()

        except KeyboardInterrupt:
            # Graceful shutdown: close current tar and save checkpoint
            self.log.warning("Shutdown: saving checkpoint...")
            if current_tar is not None:
                current_tar.close()
                # The current shard is incomplete, so checkpoint points to start of it
                current_shard_idx -= 1

            self._save_checkpoint(
                global_idx=last_completed_shard_global_idx,
                current_shard_idx=current_shard_idx,
                dataset_shard_idx=shard_idx,
                batch_start=batch_start,
                total_samples=total_samples,
            )
            pbar.close()
            self.shutdown_handler.restore()
            self.log.info("Checkpoint saved. Run with --resume to continue.")
            sys.exit(130)  # Standard exit code for SIGINT

        finally:
            self.shutdown_handler.restore()

        # Normal completion
        if current_tar is not None:
            current_tar.close()

        pbar.close()

        # Write index file
        index = {
            "num_samples": global_idx,
            "num_shards": current_shard_idx,
            "shard_size": self.shard_size,
            "format": "webdataset",
            "metadata": self.metadata,
        }
        index_path = self.output_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        # Remove checkpoint on success
        self.checkpoint_manager.delete()

        # Log final summary with performance metrics
        self.log.log_final_summary(
            total_samples=global_idx,
            total_shards=current_shard_idx,
            total_time=self.perf.get_elapsed(),
            peak_throughput=self.perf.get_peak_throughput(),
        )
        self.log.info(f"Index: {index_path}")

    def dump_lmdb(self):
        """Dump features to LMDB format."""
        try:
            import lmdb
        except ImportError:
            raise ImportError("LMDB not installed. Run: pip install lmdb")

        self.log.info(f"Dumping features to LMDB format at {self.output_dir}")

        total_samples = sum(self.dataset.shard_lengths)
        self.log.info(f"Total samples: {total_samples}")

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

        self.log.info(f"Done! Wrote {global_idx} samples to LMDB")

    def run(self, resume_checkpoint: dict | None = None):
        """Run the feature dumping process."""
        if self.output_format == "webdataset":
            self.dump_webdataset(resume_checkpoint=resume_checkpoint)
        elif self.output_format == "lmdb":
            if resume_checkpoint is not None:
                self.log.warning("LMDB format does not support resumption. Starting fresh.")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing checkpoint and start fresh (deletes previous output)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N completed shards (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for console output (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Custom log file path (default: output_dir/dumper_YYYYMMDD_HHMMSS.log)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / CheckpointManager.CHECKPOINT_FILE

    # Handle existing output directory
    if output_dir.exists():
        if args.force_restart:
            print(f"Force restart: removing existing output at {output_dir}")
            shutil.rmtree(output_dir)
        elif checkpoint_path.exists() and not args.resume:
            print(f"ERROR: Output directory exists with checkpoint at {output_dir}")
            print(f"  Use --resume to continue, or --force-restart to start fresh.")
            sys.exit(1)
        elif not checkpoint_path.exists() and args.resume:
            print(f"WARNING: --resume specified but no checkpoint found. Starting fresh.")
            args.resume = False

    print("=" * 60)
    print("GR00T N1.6 Feature Dumper")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Batch:  {args.batch_size}")
    if args.resume:
        print(f"Mode:   RESUME from checkpoint")
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
        checkpoint_interval=args.checkpoint_interval,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    # Load and validate checkpoint if resuming
    resume_checkpoint = None
    if args.resume and checkpoint_path.exists():
        dumper.log.info("Loading checkpoint...")
        resume_checkpoint = dumper.checkpoint_manager.load()
        if resume_checkpoint is None:
            dumper.log.error("Checkpoint validation failed. Use --force-restart to start fresh.")
            sys.exit(1)

        # Validate existing shards
        expected_shards = resume_checkpoint["completed_shards"]
        dumper.log.info(f"Validating {expected_shards} existing shards...")
        if not dumper.checkpoint_manager.validate_existing_shards(expected_shards):
            dumper.log.error("Some shard files are missing or corrupted.")
            sys.exit(1)

        # Clean up any incomplete shards
        dumper.checkpoint_manager.cleanup_incomplete_shards(expected_shards)
        dumper.log.info(f"Resuming from {resume_checkpoint['last_complete_global_idx']} samples, {expected_shards} shards")

    dumper.run(resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()
