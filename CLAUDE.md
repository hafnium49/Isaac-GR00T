# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac GR00T N1.5 is an open foundation model for generalized humanoid robot reasoning and skills. It's a vision-language-action (VLA) model that combines:
- **Vision-Language Model**: Eagle 2.5 (frozen during training) for multimodal understanding
- **Diffusion Transformer**: Flow matching-based action head for continuous action prediction
- **Cross-embodiment support**: Multiple pretrained action heads for different robot types

The model takes multimodal input (video, proprioceptive state, language) and outputs action sequences for robot control.

## Development Environment

### Installation
```bash
# Create conda environment (Python 3.10 required)
conda create -n gr00t python=3.10
conda activate gr00t

# Install dependencies
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

**Note**: CUDA 12.4 recommended, but CUDA 11.8 also works (use `flash-attn==2.8.2` for 11.8).

### Code Quality
Before committing, run these formatting and linting tools:
```bash
# Format code
isort .
black .

# Lint
ruff check . --fix
```

Code style: Black formatter with 100 char line length, isort for imports.

### Testing
```bash
# Run specific test module
pytest -v tests/path/to/test.py

# Run all tests
pytest
```

## Common Commands

### Dataset Loading
```bash
# Load and inspect a dataset
python scripts/load_dataset.py --dataset-path ./demo_data/robot_sim.PickNPlace
```

### Inference
```bash
# Start inference server
python scripts/inference_service.py --model-path nvidia/GR00T-N1.5-3B --server

# Run client (in separate terminal)
python scripts/inference_service.py --client

# Offline evaluation with plotting
python scripts/eval_policy.py --plot --model_path nvidia/GR00T-N1.5-3B
```

### Fine-tuning
```bash
# Basic fine-tuning (see all options with --help)
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/robot_sim.PickNPlace \
  --num-gpus 1

# Fine-tuning on RTX 4090 (memory constrained)
python scripts/gr00t_finetune.py \
  --dataset-path ./path/to/data \
  --num-gpus 1 \
  --no-tune_diffusion_model

# Multi-dataset training
python scripts/gr00t_finetune.py \
  --dataset-path <DATASET1> <DATASET2> \
  --num-gpus 1

# LoRA fine-tuning (lower memory, but reduced performance)
python scripts/gr00t_finetune.py \
  --dataset-path ./path/to/data \
  --lora_rank 64 \
  --lora_alpha 128 \
  --num-gpus 1
```

**Recommended**: Max batch size, train for 20k steps. Full model fine-tuning preferred over LoRA.

### Deployment (TensorRT)
```bash
# See deployment_scripts/README.md for full guide
# Export to ONNX
python deployment_scripts/export_onnx.py

# Run TensorRT inference
python deployment_scripts/gr00t_inference.py
```

## Architecture Overview

### Core Model Structure

```
GR00T_N1_5 (gr00t/model/gr00t_n1.py)
├── EagleBackbone (gr00t/model/backbone/eagle_backbone.py)
│   └── Eagle2.5 VLM (frozen during training)
│       ├── Vision Encoder (ViT)
│       ├── Language Model
│       └── Projector (MLP adapter with layer norm)
└── FlowmatchingActionHead (gr00t/model/action_head/flow_matching_action_head.py)
    ├── Embodiment-specific projectors (maps to action space)
    ├── Cross-attention DiT (diffusion transformer)
    └── Flow matching denoiser (FLARE objective)
```

### Data Pipeline

1. **LeRobot Dataset Format**: Extended LeRobot V2.0 with additional `meta/modality.json`
   - Videos: MP4 files in `videos/chunk-*/observation.images.<camera_name>/`
   - State/Action: Concatenated arrays in parquet files under `data/chunk-*/`
   - Metadata: `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/modality.json`

2. **Dataset Classes** (`gr00t/data/dataset.py`):
   - `LeRobotSingleDataset`: Single embodiment dataset
   - `LeRobotMixtureDataset`: Multi-embodiment training (auto-balances weights)
   - `CachedLeRobotSingleDataset`: With video frame caching

3. **Transform Pipeline** (applied in order):
   - `VideoTransform`: Crop, resize (224x224), color jitter, to tensor
   - `StateActionTransform`: Normalization (min_max, q99, mean_std, or binary)
   - `ConcatTransform`: Concatenates modalities in specified order
   - `GR00TTransform`: Pads sequences, creates final dict format

### Embodiment System

**Key Concept**: Each robot type has a dedicated action head selected via `EmbodimentTag`:

| Tag | Description | Control Space | Data Config |
|-----|-------------|---------------|-------------|
| `EmbodimentTag.GR1` | Fourier GR1 humanoid | Absolute joint | `fourier_gr1_arms_waist` |
| `EmbodimentTag.OXE_DROID` | Single arm robots | Delta EEF | `oxe_droid` |
| `EmbodimentTag.AGIBOT_GENIE1` | Humanoid with grippers | Absolute joint | `agibot_genie1` |
| `EmbodimentTag.NEW_EMBODIMENT` | Custom robots | User-defined | Custom config |

**Location**: `gr00t/data/embodiment_tags.py`
**Mapping**: Each tag maps to a specific projector index in the Action Expert Module via `EMBODIMENT_TAG_MAPPING`

### Data Config System

Data configs (`gr00t/experiment/data_config.py`) define:
- **Modality keys**: Which observation/action fields to use
- **Indices**: Which frames to sample (e.g., `[0]` for current, `[-1, 0]` for history)
- **Transforms**: Video augmentation and normalization parameters

**Important**: When creating new embodiments, define a custom data config inheriting from `BaseDataConfig`.

### Modality Configuration

`meta/modality.json` schema:
```json
{
  "state": {
    "<key>": {"start": int, "end": int, "rotation_type": str (optional)}
  },
  "action": {
    "<key>": {"start": int, "end": int, "absolute": bool (optional)}
  },
  "video": {
    "<new_key>": {"original_key": "<lerobot_key>"}
  },
  "annotation": {
    "<source>.<type>.<name>": {}
  }
}
```

Rotation types supported: `quaternion`, `axis_angle`, `rotation_6d`, `euler_angles_*`, etc.

## Key Training Parameters

When calling `GR00T_N1_5.from_pretrained`, control what gets fine-tuned:

- `tune_visual` (bool): Fine-tune vision encoder. Set to `true` only if visual domain differs significantly from pre-training. Expensive.
- `tune_llm` (bool): Fine-tune language model. Rarely needed (default: `false`).
- `tune_projector` (bool): Fine-tune projector. Default: `true` (recommended).
- `tune_diffusion_model` (bool): Fine-tune DiT. Default: `false`. Set to `true` for better performance if memory allows.

**Embodiment-specific heads**: Only the head matching your `embodiment_tag` is fine-tuned; others remain frozen.

## Inference Details

**Default denoising steps**: 4 (sufficient for most cases)

**Performance (H100, single sample)**:
- VLM Backbone: 23.18ms
- Action Head (4 steps): 24.7ms
- **Total**: ~48ms

**Entry Point**: `Gr00tPolicy` class in `gr00t/model/policy.py`

```python
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.5-3B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)

action_chunk = policy.get_action(observation_dict)
```

## Important Implementation Notes

### Data Format Requirements
- Videos must be **256x256 pixels with top/bottom padding** to preserve aspect ratio before training
- See `demo_data/robot_sim.PickNPlace/videos/` for examples
- Annotations stored as indices pointing to `meta/tasks.jsonl` (same as LeRobot V2 `task_index`)

### Multi-dataset Training
`LeRobotMixtureDataset` automatically balances:
- Dataset weights (`balance_dataset_weights=True`)
- Trajectory weights (`balance_trajectory_weights=True`)

Override by passing custom sampling weights to the class.

### Model Architecture Details
- **Flow Matching**: Uses FLARE (Future Latent Representation Alignment) objective alongside flow matching loss
- **Vision Encoder**: RADIO model from Eagle 2.5
- **Frozen VLM**: Preserves language understanding and improves generalization
- **Action Horizon**: Variable, defined in config (typically 10-20 steps)

## Testing Strategy

The codebase includes tests in `tests/` directory. When adding features:
1. Run existing tests: `pytest -v tests/path/to/relevant_test.py`
2. Add tests for new functionality in corresponding test modules
3. Ensure tests pass before opening PR

## Hardware Considerations

- **Finetuning**: H100/L40 optimal. A6000/RTX 4090 work but slower. RTX 4090 requires `--no-tune_diffusion_model`.
- **LoRA**: 2x A6000 or 2x RTX 4090
- **Inference**: Most modern GPUs similar performance (L40 ≈ RTX 4090 for single sample)
- **Jetson/Thor**: See `deployment_scripts/README.md` for TensorRT deployment

## Blackwell GPUs (RTX 6000 Ada)

Special setup required:
1. Install PyTorch for your CUDA version (e.g., CUDA 12.8)
2. Build Flash Attention from source with `TORCH_CUDA_ARCH_LIST="sm_120"`
3. See FAQ in README.md for full instructions

## Directory Structure

- `gr00t/`: Core library
  - `model/`: Model architecture (policy, backbone, action head)
  - `data/`: Dataset loading, transforms, embodiment tags
  - `experiment/`: Training configs and runner
  - `eval/`: Evaluation utilities and wrappers
  - `utils/`: Misc utilities
- `scripts/`: Executable scripts for common tasks
- `deployment_scripts/`: TensorRT/ONNX export and inference
- `getting_started/`: Jupyter notebooks and documentation
- `examples/`: Benchmark examples (RoboCasa, Libero, SimplerEnv, SO-100)
- `demo_data/`: Example dataset in LeRobot format

## External Resources

- Model: [huggingface.co/nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- Dataset: [huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
- Reference Architecture: `reference_architecture/reference_architecture.md`
- Tutorials: `getting_started/*.ipynb` notebooks
- Sim Evaluation: [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Pull request workflow
- Code formatting requirements (isort, black, ruff)
- Testing expectations
- Developer Certificate of Origin

All contributions must include proper tests and pass CI checks.
