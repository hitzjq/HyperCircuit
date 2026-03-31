#!/usr/bin/env python3
"""
Batch evaluation of pretrain checkpoints on the evaluation set.

Usage:
    # Multi-GPU (recommended)
    torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
        scripts/eval/eval_checkpoints.py \
        --checkpoint_dir checkpoints/ARC-AGI-1 \
        --data_path data/arc1concept-aug-1000

    # Single GPU
    python scripts/eval/eval_checkpoints.py \
        --checkpoint_dir checkpoints/ARC-AGI-1 \
        --data_path data/arc1concept-aug-1000
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so we can import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from evaluators.arc import ARC


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluate pretrain checkpoints")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory containing step_xxxx checkpoint files")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to evaluation dataset, e.g. data/arc1concept-aug-1000")
    p.add_argument("--arch_config", type=str, default="trm",
                   help="Architecture config name under config/arch/ (default: trm)")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Global batch size for evaluation (default: 256)")
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step numbers to evaluate. Default: all found.")

    # Architecture overrides (to match pretrain settings)
    p.add_argument("--H_cycles", type=int, default=None, help="Override arch.H_cycles")
    p.add_argument("--L_cycles", type=int, default=None, help="Override arch.L_cycles")
    p.add_argument("--L_layers", type=int, default=None, help="Override arch.L_layers")
    return p.parse_args()


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def load_arch_config(arch_name: str) -> dict:
    """Load architecture config YAML and resolve Hydra-style self-references."""
    config_path = os.path.join(PROJECT_ROOT, "config", "arch", f"{arch_name}.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve simple ${.key} self-references (e.g. puzzle_emb_ndim: ${.hidden_size})
    for key, val in list(cfg.items()):
        if isinstance(val, str) and val.startswith("${.") and val.endswith("}"):
            ref_key = val[3:-1]
            cfg[key] = cfg[ref_key]
    return cfg


def find_checkpoints(checkpoint_dir: str, steps: Optional[List[int]] = None) -> List[Tuple[int, str]]:
    """Discover step_xxxx checkpoint files and return sorted (step, path) list."""
    results = []
    for entry in os.listdir(checkpoint_dir):
        m = re.match(r"^step_(\d+)$", entry)
        if m:
            step = int(m.group(1))
            path = os.path.join(checkpoint_dir, entry)
            if os.path.isfile(path) and (steps is None or step in steps):
                results.append((step, path))
    results.sort(key=lambda x: x[0])
    return results


# ──────────────────────────────────────────
# Data
# ──────────────────────────────────────────

def create_eval_dataloader(data_path: str, batch_size: int, rank: int, world_size: int):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[data_path],
        rank=rank,
        num_replicas=world_size,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=batch_size,
    ), split="test")
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata


# ──────────────────────────────────────────
# Model
# ──────────────────────────────────────────

def create_model(arch_config: dict, metadata: PuzzleDatasetMetadata,
                 batch_size: int, world_size: int) -> nn.Module:
    # Separate model params from meta keys
    model_cfg = {k: v for k, v in arch_config.items() if k not in ("name", "loss")}
    model_cfg.update(
        batch_size=batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(arch_config["name"])
    loss_head_cls = load_model_class(arch_config["loss"]["name"])
    loss_kwargs = {k: v for k, v in arch_config["loss"].items() if k != "name"}

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_kwargs)
    return model


def load_checkpoint_into(model: nn.Module, path: str) -> Tuple[int, int]:
    """Load checkpoint weights. Returns (n_missing, n_unexpected)."""
    sd = torch.load(path, map_location="cuda")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Handle puzzle-embedding shape mismatch
    emb_key = "model.inner.puzzle_emb.weights"
    if emb_key in sd:
        expected = model.model.puzzle_emb.weights.shape  # type: ignore
        if sd[emb_key].shape != expected:
            print(f"  ⚠ Resizing puzzle_emb: {sd[emb_key].shape} -> {expected}")
            sd[emb_key] = torch.mean(sd[emb_key], dim=0, keepdim=True).expand(expected).contiguous()

    info = model.load_state_dict(sd, assign=True, strict=False)
    return len(info.missing_keys), len(info.unexpected_keys)


# ──────────────────────────────────────────
# Evaluation loop (mirrors lora_finetune.evaluate)
# ──────────────────────────────────────────

def run_evaluation(model, eval_loader, eval_metadata, evaluators,
                   rank, world_size, cpu_group):
    reduced_metrics = None
    with torch.inference_mode():
        return_keys = set()
        for ev in evaluators:
            ev.begin_eval()
            return_keys.update(ev.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        metric_keys: list = []
        metric_values = None

        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            # ACT multi-step inference
            while True:
                carry, _loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                if all_finish:
                    break

            for ev in evaluators:
                ev.update_batch(batch, preds)

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metric_keys)), dtype=torch.float32, device="cuda"
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del carry, _loss, preds, batch, all_finish, metrics

        # Reduce standard metrics
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            if rank == 0:
                arr = metric_values.cpu().numpy()
                reduced_metrics = {
                    sname: {
                        mname: arr[sid, mid]
                        for mid, mname in enumerate(metric_keys)
                    }
                    for sid, sname in enumerate(set_ids)
                }
                for sname, m in reduced_metrics.items():
                    cnt = m.pop("count")
                    reduced_metrics[sname] = {k: v / cnt for k, v in m.items()}

        # Run evaluators (ARC pass@K)
        for ev in evaluators:
            ev_metrics = ev.result(None, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and ev_metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(ev_metrics)

    return reduced_metrics


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed init ──
    RANK, WORLD_SIZE, CPU_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_GROUP = dist.new_group(backend="gloo")

    # ── Discover checkpoints ──
    step_filter = [int(s.strip()) for s in args.steps.split(",")] if args.steps else None
    checkpoints = find_checkpoints(args.checkpoint_dir, step_filter)

    if RANK == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 60)
        print(f"[{now}] 📊 Batch Checkpoint Evaluator")
        print(f"  Checkpoint Dir : {args.checkpoint_dir}")
        print(f"  Data Path      : {args.data_path}")
        print(f"  Architecture   : {args.arch_config}")
        print(f"  Batch Size     : {args.batch_size}")
        print(f"  GPUs           : {WORLD_SIZE}")
        print(f"  Checkpoints    : {len(checkpoints)}")
        for step, _ in checkpoints:
            print(f"    - step_{step}")
        print("=" * 60, flush=True)

    # ── Load arch config with CLI overrides ──
    arch_config = load_arch_config(args.arch_config)
    if args.H_cycles is not None:
        arch_config["H_cycles"] = args.H_cycles
    if args.L_cycles is not None:
        arch_config["L_cycles"] = args.L_cycles
    if args.L_layers is not None:
        arch_config["L_layers"] = args.L_layers

    # ── Dataloader (created once, reused) ──
    eval_loader, eval_metadata = create_eval_dataloader(
        args.data_path, args.batch_size, RANK, WORLD_SIZE
    )

    # ── Evaluator ──
    # aggregated_voting=False so state is cleared between checkpoints via begin_eval()
    evaluators = [
        ARC(data_path=args.data_path, eval_metadata=eval_metadata, aggregated_voting=False)
    ]

    # ── Create model (once, reload weights for each checkpoint) ──
    if RANK == 0:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model...", flush=True)
    with torch.device("cuda"):
        model = create_model(arch_config, eval_metadata, args.batch_size, WORLD_SIZE)

    # ── Evaluate each checkpoint ──
    all_results: Dict[int, dict] = {}

    for i, (step, ckpt_path) in enumerate(checkpoints):
        if RANK == 0:
            print(f"\n{'=' * 60}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Evaluating checkpoint {i + 1}/{len(checkpoints)}: step_{step}")
            print(f"{'=' * 60}", flush=True)

        # Load weights on rank 0, then broadcast
        if RANK == 0:
            n_miss, n_unexp = load_checkpoint_into(model, ckpt_path)
            print(f"  [{n_miss} missing, {n_unexp} unexpected]: Loaded.", flush=True)

        if WORLD_SIZE > 1:
            with torch.no_grad():
                for p in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(p, src=0)

        model.eval()
        metrics = run_evaluation(
            model, eval_loader, eval_metadata, evaluators,
            rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_GROUP
        )

        if RANK == 0 and metrics is not None:
            all_results[step] = metrics
            print("=" * 40)
            print(f"📊 RESULT step_{step}: {metrics}")
            print("=" * 40, flush=True)

        # Free CUDA cache between checkpoints
        torch.cuda.empty_cache()

    # ── Summary table ──
    if RANK == 0 and all_results:
        print(f"\n\n{'=' * 70}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'=' * 70}")

        # Collect ARC pass@K keys
        first = next(iter(all_results.values()))
        pass_keys = sorted([k for k in first if k.startswith("ARC/")])

        # Also collect standard metric keys (accuracy, exact_accuracy, lm_loss, etc.)
        std_keys = []
        for v in first.values():
            if isinstance(v, dict):
                std_keys = sorted(v.keys())
                break

        # Print pass@K table
        header = f"{'Step':>12}"
        for k in pass_keys:
            header += f" | {k:>12}"
        print(header)
        print("-" * len(header))
        for step in sorted(all_results):
            row = f"{step:>12}"
            for k in pass_keys:
                val = all_results[step].get(k, 0.0)
                row += f" | {val:>12.4f}"
            print(row)

        # Print standard metrics table
        if std_keys:
            print()
            header2 = f"{'Step':>12}"
            for k in std_keys:
                header2 += f" | {k:>16}"
            print(header2)
            print("-" * len(header2))
            for step in sorted(all_results):
                row = f"{step:>12}"
                for sname, smetrics in all_results[step].items():
                    if isinstance(smetrics, dict):
                        for k in std_keys:
                            row += f" | {smetrics.get(k, 0.0):>16.4f}"
                        break
                print(row)

        print("=" * 70, flush=True)

        # Save results as JSON
        save_path = os.path.join(args.checkpoint_dir, "eval_results.json")
        serializable = {}
        for step, m in all_results.items():
            serializable[step] = {}
            for k, v in m.items():
                if isinstance(v, dict):
                    serializable[step][k] = {kk: float(vv) for kk, vv in v.items()}
                else:
                    serializable[step][k] = float(v)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n💾 Results saved to: {save_path}")

    # ── Cleanup ──
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
