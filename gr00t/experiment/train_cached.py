"""
train_cached.py - Training with Pre-computed Backbone Features

This module provides the training loop for GR00T N1.6 using cached
backbone features. It bypasses the Eagle backbone entirely, enabling
>20 steps/sec throughput.

Usage:
    Called automatically from launch_finetune.py when use_cached_features=True

Requirements:
    - Features must be pre-computed using dump_features_n16.py
    - Backbone must be frozen (tune_visual=False, tune_llm=False)
"""

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed
import wandb

from gr00t.configs.base_config import Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.data.dataset.cached_dataset import (
    CachedFeatureCollator,
    CachedFeatureDataset,
    CachedWebDatasetIterable,
)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )


def run_cached_training(
    config: Config,
    cached_features_path: str,
    ft_config,
):
    """
    Run training with cached backbone features.

    This function:
    1. Loads the GR00T model with frozen backbone
    2. Creates dataloader from cached features
    3. Runs training loop (action head only)

    Args:
        config: Training configuration
        cached_features_path: Path to cached features
        ft_config: Finetune configuration
    """
    # Initialize distributed if needed
    if dist.is_initialized():
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    setup_logging()
    set_seed(config.data.seed)

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if configured
    if config.training.use_wandb and global_rank == 0:
        wandb.init(
            project=config.training.wandb_project,
            name=f"cached_{output_dir.name}",
            config={
                "cached_features_path": cached_features_path,
                "max_steps": config.training.max_steps,
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.global_batch_size,
            },
        )

    # Load model
    logging.info(f"Loading model from {config.training.start_from_checkpoint}...")
    model = Gr00tN1d6.from_pretrained(
        config.training.start_from_checkpoint,
        tune_llm=False,  # Force frozen backbone
        tune_visual=False,  # Force frozen backbone
        tune_top_llm_layers=0,  # Force all LLM layers frozen (override config default of 4)
        tune_projector=config.model.tune_projector,
        tune_diffusion_model=config.model.tune_diffusion_model,
        tune_vlln=config.model.tune_vlln,
        state_dropout_prob=config.model.state_dropout_prob,
        trust_remote_code=True,
    )

    # Force freeze backbone - from_pretrained kwargs don't reliably update config
    # This explicitly calls eagle_backbone.py:set_trainable_parameters() to freeze all backbone params
    model.backbone.set_trainable_parameters(
        tune_llm=False,
        tune_visual=False,
        tune_top_llm_layers=0,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Verify backbone is frozen
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    if backbone_trainable:
        raise ValueError(
            "Backbone has trainable parameters! Cached features require frozen backbone."
        )

    # Create cached dataset
    logging.info(f"Loading cached features from {cached_features_path}...")

    # Use iterable dataset for better I/O performance
    train_dataset = CachedWebDatasetIterable(
        cached_path=cached_features_path,
        shuffle_shards=True,
        seed=config.data.seed,
    )

    collator = CachedFeatureCollator()

    per_device_batch_size = config.training.global_batch_size // max(config.training.num_gpus, 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        num_workers=config.training.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Create optimizer (action head only)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    num_training_steps = config.training.max_steps
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    logging.info("Starting training with cached features...")
    logging.info(f"  Max steps: {num_training_steps}")
    logging.info(f"  Batch size: {per_device_batch_size}")
    logging.info(f"  Learning rate: {config.training.learning_rate}")

    global_step = 0
    total_loss = 0.0
    log_interval = 10
    save_steps = config.training.save_steps

    # Timing for throughput measurement
    step_times = []
    warmup_steps = 50  # Skip first 50 steps for timing

    pbar = tqdm(total=num_training_steps, desc="Training", disable=global_rank != 0)

    data_iter = iter(train_loader)
    start_time = time.time()

    while global_step < num_training_steps:
        step_start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Move to device
        inputs = batch["inputs"]
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        # Forward pass (uses cached features)
        outputs = model(inputs)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        global_step += 1

        # Timing
        step_time = time.time() - step_start
        if global_step > warmup_steps:
            step_times.append(step_time)

        # Logging
        if global_step % log_interval == 0:
            avg_loss = total_loss / log_interval
            total_loss = 0.0

            # Calculate throughput
            if len(step_times) > 0:
                avg_step_time = sum(step_times[-100:]) / min(len(step_times), 100)
                steps_per_sec = 1.0 / avg_step_time
                samples_per_sec = steps_per_sec * per_device_batch_size
            else:
                # Skip throughput metrics during warmup (would be 0, dominating charts)
                steps_per_sec = None
                samples_per_sec = None

            if global_rank == 0:
                log_dict = {
                    "loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                # Only log throughput after warmup to avoid zeros in charts
                if steps_per_sec is not None:
                    log_dict["steps_per_sec"] = steps_per_sec
                    log_dict["samples_per_sec"] = samples_per_sec

                if config.training.use_wandb:
                    wandb.log(log_dict, step=global_step)

                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if steps_per_sec is not None:
                    postfix["steps_s"] = f"{steps_per_sec:.1f}"
                pbar.set_postfix(**postfix)

        # Save checkpoint
        if global_step % save_steps == 0 and global_rank == 0:
            checkpoint_dir = output_dir / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model.save_pretrained(checkpoint_dir)

            # Save training state
            torch.save(
                {
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_dir / "training_state.pt",
            )

            logging.info(f"Saved checkpoint to {checkpoint_dir}")

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time

    # Print final stats
    if global_rank == 0:
        logging.info("Training complete!")
        logging.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logging.info(f"  Steps: {global_step}")

        if len(step_times) > 0:
            avg_step_time = sum(step_times) / len(step_times)
            logging.info(f"  Avg step time: {avg_step_time*1000:.1f} ms")
            logging.info(f"  Throughput: {1.0/avg_step_time:.1f} steps/sec")
            logging.info(f"  Throughput: {per_device_batch_size/avg_step_time:.1f} samples/sec")

        # Save final model
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        logging.info(f"Saved final model to {final_dir}")

        if config.training.use_wandb:
            wandb.finish()
