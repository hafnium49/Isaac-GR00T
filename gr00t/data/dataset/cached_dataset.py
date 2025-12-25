"""
cached_dataset.py - Cached Feature Dataset for GR00T N1.6

Loads pre-computed Eagle backbone features from disk instead of running
the backbone at every training step. Enables >20 steps/sec throughput.

Supports two formats:
- WebDataset: TAR shards for sequential I/O
- LMDB: Fast random access database

Usage:
    dataset = CachedFeatureDataset(
        cached_path="/path/to/cached_features",
        format="webdataset",
    )
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from transformers.feature_extraction_utils import BatchFeature


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-computed backbone features from disk.

    This dataset replaces the standard video-loading dataset when using
    cached features. It loads:
    - backbone_features: [seq_len, 2048] fp16
    - attention_mask: [seq_len]
    - image_mask: [seq_len]
    - state: [state_horizon, state_dim]
    - action: [action_horizon, action_dim]
    - action_mask: [action_horizon, action_dim]
    - embodiment_id: int

    Metadata validation ensures cache was created with compatible settings.
    """

    def __init__(
        self,
        cached_path: str | Path,
        validate_metadata: bool = True,
        expected_model_path: str | None = None,
    ):
        """
        Initialize cached feature dataset.

        Args:
            cached_path: Path to cached features directory
            validate_metadata: Whether to validate cache metadata
            expected_model_path: Expected model path for validation
        """
        self.cached_path = Path(cached_path)

        if not self.cached_path.exists():
            raise FileNotFoundError(f"Cached features not found: {cached_path}")

        # Load index
        index_path = self.cached_path / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.num_samples = self.index["num_samples"]
        self.metadata = self.index.get("metadata", {})

        # Validate metadata if requested
        if validate_metadata and expected_model_path:
            cached_model = self.metadata.get("backbone_ckpt", "")
            if expected_model_path not in cached_model and cached_model not in expected_model_path:
                print(
                    f"WARNING: Model mismatch! Cache: {cached_model}, Expected: {expected_model_path}"
                )

        # Detect format
        self.format = self.index.get("format", "webdataset")

        if self.format == "webdataset":
            self._init_webdataset()
        elif self.format == "lmdb":
            self._init_lmdb()
        else:
            raise ValueError(f"Unknown format: {self.format}")

        print(f"Loaded cached features: {self.num_samples} samples, format={self.format}")
        print(f"Cache metadata: {self.metadata.get('config_hash', 'unknown')}")

    def _init_webdataset(self):
        """Initialize WebDataset format loading."""
        self.shard_size = self.index.get("shard_size", 1000)
        self.num_shards = self.index.get("num_shards", 0)

        # Build sample-to-shard mapping
        self.shard_files = sorted(
            self.cached_path.glob("shard-*.tar"),
            key=lambda x: int(x.stem.split("-")[1]),
        )

        if len(self.shard_files) == 0:
            raise FileNotFoundError(f"No shard files found in {self.cached_path}")

        # Validate cache completion
        completed = self.index.get("completed", False)
        if not completed:
            raise ValueError(
                f"Cache at {self.cached_path} is not complete. "
                f"Run ./groot/dump_features.sh or use --resume to continue an interrupted run."
            )

        # Validate shard count matches index
        actual_shards = len(self.shard_files)
        if actual_shards < self.num_shards:
            raise ValueError(
                f"Incomplete cache: found {actual_shards}/{self.num_shards} shards. "
                f"Run ./groot/dump_features.sh --resume to complete the feature extraction."
            )

        # Cache for loaded shards
        self._shard_cache = {}
        self._cache_max_shards = 2  # Keep at most 2 shards in memory

    def _init_lmdb(self):
        """Initialize LMDB format loading."""
        try:
            import lmdb
        except ImportError:
            raise ImportError("LMDB not installed. Run: pip install lmdb")

        lmdb_path = self.cached_path / "features.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB database not found: {lmdb_path}")

        self.env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

    def __len__(self) -> int:
        return self.num_samples

    def _load_webdataset_sample(self, idx: int) -> dict:
        """Load a sample from WebDataset shards."""
        import tarfile
        from io import BytesIO

        # Find shard
        shard_idx = idx // self.shard_size
        sample_key = f"{idx:08d}"

        # Load shard if not cached
        if shard_idx not in self._shard_cache:
            # Evict old shards
            if len(self._shard_cache) >= self._cache_max_shards:
                oldest = min(self._shard_cache.keys())
                del self._shard_cache[oldest]

            # Load shard
            shard_path = self.shard_files[shard_idx]
            shard_data = {}

            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    # Parse filename: {key}.{field}.{ext}
                    parts = member.name.split(".")
                    if len(parts) >= 2:
                        key = parts[0]
                        if key not in shard_data:
                            shard_data[key] = {}

                        f = tar.extractfile(member)
                        if f is not None:
                            if member.name.endswith(".pt"):
                                shard_data[key][parts[1]] = torch.load(
                                    BytesIO(f.read()),
                                    map_location="cpu",
                                    weights_only=False,
                                )
                            elif member.name.endswith(".json"):
                                shard_data[key]["json"] = json.load(f)

            self._shard_cache[shard_idx] = shard_data

        # Get sample
        shard_data = self._shard_cache[shard_idx]
        if sample_key not in shard_data:
            raise IndexError(f"Sample {idx} not found in shard {shard_idx}")

        sample = shard_data[sample_key]

        # Parse JSON metadata
        json_data = sample.get("json", {})

        return {
            "features": sample.get("features"),
            "attention_mask": sample.get("attention_mask"),
            "image_mask": sample.get("image_mask"),
            "state": sample.get("state"),
            "action": sample.get("action"),
            "action_mask": sample.get("action_mask"),
            "embodiment_id": json_data.get("embodiment_id"),
        }

    def _load_lmdb_sample(self, idx: int) -> dict:
        """Load a sample from LMDB."""
        import pickle

        key = f"{idx:08d}".encode()

        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Sample {idx} not found in LMDB")

            sample = pickle.loads(value)

        # Convert numpy to torch
        return {
            "features": torch.from_numpy(sample["features"]) if sample["features"] is not None else None,
            "attention_mask": torch.from_numpy(sample["attention_mask"]) if sample["attention_mask"] is not None else None,
            "image_mask": torch.from_numpy(sample["image_mask"]) if sample["image_mask"] is not None else None,
            "state": torch.from_numpy(sample["state"]) if sample["state"] is not None else None,
            "action": torch.from_numpy(sample["action"]) if sample["action"] is not None else None,
            "action_mask": torch.from_numpy(sample["action_mask"]) if sample["action_mask"] is not None else None,
            "embodiment_id": sample["embodiment_id"],
        }

    def __getitem__(self, idx: int) -> dict:
        """Load a single sample."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        if self.format == "webdataset":
            return self._load_webdataset_sample(idx)
        else:
            return self._load_lmdb_sample(idx)


class CachedFeatureCollator:
    """
    Collator for cached feature batches.

    Converts cached feature samples into BatchFeature format
    compatible with the action head forward pass.
    """

    def __init__(self, max_state_dim: int = 29, max_action_dim: int = 29):
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim

    def __call__(self, features: list[dict]) -> dict:
        """Collate batch of cached features."""
        batch = {}

        # Stack tensors
        batch["cached_features"] = torch.stack([f["features"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["image_mask"] = torch.stack([f["image_mask"] for f in features])

        # Stack state and action if present
        if features[0]["state"] is not None:
            batch["state"] = torch.stack([f["state"] for f in features])

        if features[0]["action"] is not None:
            batch["action"] = torch.stack([f["action"] for f in features])

        if features[0]["action_mask"] is not None:
            batch["action_mask"] = torch.stack([f["action_mask"] for f in features])

        # Handle embodiment_id
        if features[0]["embodiment_id"] is not None:
            batch["embodiment_id"] = torch.tensor(
                [f["embodiment_id"] for f in features], dtype=torch.long
            )

        return {"inputs": batch}


class CachedWebDatasetIterable(IterableDataset):
    """
    Iterable dataset for streaming WebDataset shards.

    More efficient than random access for sequential training.
    Loads shards one at a time and streams samples.
    """

    def __init__(
        self,
        cached_path: str | Path,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.cached_path = Path(cached_path)

        # Load index
        index_path = self.cached_path / "index.json"
        with open(index_path, "r") as f:
            self.index = json.load(f)

        self.num_samples = self.index["num_samples"]
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Find shard files
        self.shard_files = sorted(
            self.cached_path.glob("shard-*.tar"),
            key=lambda x: int(x.stem.split("-")[1]),
        )

        # Validate cache completion
        completed = self.index.get("completed", False)
        if not completed:
            raise ValueError(
                f"Cache at {cached_path} is not complete. "
                f"Run ./groot/dump_features.sh or use --resume to continue an interrupted run."
            )

        # Validate shard count matches index
        expected_shards = self.index.get("num_shards", 0)
        actual_shards = len(self.shard_files)
        if actual_shards < expected_shards:
            raise ValueError(
                f"Incomplete cache: found {actual_shards}/{expected_shards} shards. "
                f"Run ./groot/dump_features.sh --resume to complete the feature extraction."
            )

    def __iter__(self) -> Iterator[dict]:
        """Iterate through all shards and samples."""
        import tarfile
        from io import BytesIO

        worker_info = torch.utils.data.get_worker_info()

        # Determine which shards this worker handles
        if worker_info is None:
            shard_indices = list(range(len(self.shard_files)))
        else:
            # Split shards across workers
            per_worker = len(self.shard_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.shard_files)
            shard_indices = list(range(start, end))

        # Shuffle shards if requested
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(shard_indices)

        # Stream samples from shards
        for shard_idx in shard_indices:
            shard_path = self.shard_files[shard_idx]
            samples = []

            with tarfile.open(shard_path, "r") as tar:
                shard_data = {}
                for member in tar.getmembers():
                    parts = member.name.split(".")
                    if len(parts) >= 2:
                        key = parts[0]
                        if key not in shard_data:
                            shard_data[key] = {}

                        f = tar.extractfile(member)
                        if f is not None:
                            if member.name.endswith(".pt"):
                                shard_data[key][parts[1]] = torch.load(
                                    BytesIO(f.read()),
                                    map_location="cpu",
                                    weights_only=False,
                                )
                            elif member.name.endswith(".json"):
                                shard_data[key]["json"] = json.load(f)

                # Yield samples
                for key in sorted(shard_data.keys()):
                    sample = shard_data[key]
                    json_data = sample.get("json", {})

                    yield {
                        "features": sample.get("features"),
                        "attention_mask": sample.get("attention_mask"),
                        "image_mask": sample.get("image_mask"),
                        "state": sample.get("state"),
                        "action": sample.get("action"),
                        "action_mask": sample.get("action_mask"),
                        "embodiment_id": json_data.get("embodiment_id"),
                    }


def get_cached_dataloader(
    cached_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    use_iterable: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for cached features.

    Args:
        cached_path: Path to cached features
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        use_iterable: Use iterable dataset for better I/O

    Returns:
        DataLoader for cached features
    """
    collator = CachedFeatureCollator()

    if use_iterable:
        dataset = CachedWebDatasetIterable(
            cached_path=cached_path,
            shuffle_shards=shuffle,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
        )
    else:
        dataset = CachedFeatureDataset(cached_path=cached_path)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory,
        )
