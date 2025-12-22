#!/usr/bin/env python3
"""Create index.json for cached features directory."""
import json
import tarfile
from pathlib import Path

cached_path = Path("/home/h_fujiwara/projects/Isaac-GR00T/test_outputs/cached_features")

# Load metadata
with open(cached_path / "metadata.json") as f:
    metadata = json.load(f)

# Count shards and samples
shard_files = sorted(cached_path.glob("shard-*.tar"))
num_shards = len(shard_files)

total_samples = 0
samples_per_shard = []
for shard_path in shard_files:
    with tarfile.open(shard_path, "r") as tar:
        # Count unique sample keys (files ending with .json)
        count = len([m for m in tar.getnames() if m.endswith(".json")])
        samples_per_shard.append(count)
        total_samples += count

print(f"Found {num_shards} shards")
print(f"Total samples: {total_samples}")
print(f"Samples per shard (first 5): {samples_per_shard[:5]}")
print(f"Samples per shard (last 5): {samples_per_shard[-5:]}")

# Create index
index = {
    "num_samples": total_samples,
    "format": "webdataset",
    "shard_size": 50,  # expected from args
    "num_shards": num_shards,
    "metadata": metadata,
}

# Write index
with open(cached_path / "index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"\nCreated index.json:")
print(json.dumps(index, indent=2))
