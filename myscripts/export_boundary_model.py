#!/usr/bin/env python3
"""Merge Qwen3 base权重与句边界头部，导出新的 HuggingFace 目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Qwen3 主干与句边界头导出为单一模型目录"
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/data/workspace/model/Qwen/Qwen3-0.6B"),
        help="原始 Qwen3 模型目录",
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=Path("/data/workspace/stream/checkpoints/head/best_boundary_head.pt"),
        help="二分类句边界头 checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/workspace/model/Qwen/Qwen3Boundary-Stream"),
        help="导出目标目录",
    )
    return parser.parse_args()


def copy_base_files(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in base_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)


def load_head_state(head_path: Path) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    head_state = ckpt["boundary_head"]
    config = ckpt.get("config", {})
    return head_state, config


def save_head_state(head_state: Dict[str, torch.Tensor], output_dir: Path) -> Path:
    head_file = output_dir / "boundary_head.pt"
    torch.save(head_state, head_file)
    return head_file


def update_config(
    config_path: Path, head_meta: Dict[str, Any], head_file: Path
) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["sentence_boundary"] = {
        "num_labels": 2,
        "hidden_size": head_meta.get("hidden_size"),
        "dropout": head_meta.get("dropout"),
        "max_length": head_meta.get("max_length"),
        "head_file": head_file.name,
    }
    config.setdefault("architectures", [])
    # VLLM 只需要 boundary 架构，去掉原有的 CausalLM
    config["architectures"] = [
        arch for arch in config["architectures"] if arch != "Qwen3ForCausalLM"
    ]
    if "Qwen3BoundaryForStreaming" not in config["architectures"]:
        config["architectures"].append("Qwen3BoundaryForStreaming")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def write_boundary_meta(output_dir: Path, head_meta: Dict[str, Any]) -> None:
    meta_path = output_dir / "boundary_meta.json"
    serializable = {}
    for key, value in head_meta.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    copy_base_files(args.base_model, args.output_dir)
    head_state, head_meta = load_head_state(args.head_checkpoint)
    head_file = save_head_state(head_state, args.output_dir)
    config_path = args.output_dir / "config.json"
    update_config(config_path, head_meta, head_file)
    write_boundary_meta(
        args.output_dir,
        {
            "checkpoint": str(args.head_checkpoint),
            **head_meta,
        },
    )
    print(
        f"Boundary model exported to {args.output_dir}. "
        f"Head weights stored at {head_file.name}."
    )


if __name__ == "__main__":
    main()
