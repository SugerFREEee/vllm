#!/usr/bin/env python3
"""示例脚本：加载训练好的二分类句边界头并完成一次推理。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_sentence_boundary import (
    BOUNDARY_LABEL2ID,
    SentenceBoundaryHead,
    ensure_pad_token,
)

LABEL_TEXT = {
    BOUNDARY_LABEL2ID["complete"]: "完整断句",
    BOUNDARY_LABEL2ID["incomplete"]: "未完结",
}


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="利用冻结 Qwen3 + 句边界头进行一次简单推理"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="直接输入待推理的文本（与 --file 互斥）",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="读取包含待推理文本的文件（整个文件作为单条输入）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/head_backup"),
        help="训练脚本保存的 tokenizer 与 boundary head 目录",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/data/workspace/model/Qwen/Qwen3-0.6B",
        help="冻结主干模型路径或 HF hub 标识",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="推理所用 dtype（默认读取训练配置）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="tokenizer 编码时的最大长度（默认读取训练配置）",
    )
    return parser.parse_args()


def load_components(
    checkpoint_dir: Path,
    base_model_name: str,
    dtype_name: str | None,
) -> Tuple[
    AutoTokenizer,
    AutoModelForCausalLM,
    SentenceBoundaryHead,
    torch.dtype,
    dict,
]:
    ckpt_path = checkpoint_dir / "best_boundary_head.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    dtype = dtype_from_name(dtype_name or config.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    ensure_pad_token(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    base_model.eval()

    head = SentenceBoundaryHead(
        vocab_size=base_model.lm_head.out_features,
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
    ).to(dtype=dtype)
    head.load_state_dict(ckpt["boundary_head"])
    head.eval()
    return tokenizer, base_model, head, dtype, config


def run_streaming_inference(
    text: str,
    tokenizer,
    base_model,
    head,
    torch_dtype: torch.dtype,
    max_length: int,
) -> List[Tuple[str, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device=device, dtype=torch_dtype)
    head = head.to(device=device, dtype=torch_dtype)

    stream_results: List[Tuple[str, str]] = []
    past_key_values = None
    total_tokens = 0

    with torch.no_grad():
        for ch in text:
            token_ids = tokenizer.encode(ch, add_special_tokens=False)
            if not token_ids:
                continue
            pred_id = None
            for token_id in token_ids:
                total_tokens += 1
                if total_tokens > max_length:
                    break
                token_tensor = torch.tensor(
                    [[token_id]], dtype=torch.long, device=device
                )
                outputs = base_model(
                    input_ids=token_tensor,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                probs = torch.softmax(outputs.logits[:, -1:, :], dim=-1)
                boundary_logits = head(probs)
                pred_id = boundary_logits[:, -1, :].argmax(dim=-1).item()
            if total_tokens > max_length:
                break
            if pred_id is None:
                continue
            stream_results.append(
                (ch, LABEL_TEXT.get(pred_id, f"label_{pred_id}"))
            )
    return stream_results


def main() -> None:
    args = parse_args()
    if not args.text and not args.file:
        args.text = "１２月３１日，中共中央总书记、国家主席江泽民发表新年讲话。新华社记者兰红光摄，这是一九九八年新年讲话的现场记录。"

    text = args.text
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    assert text is not None

    tokenizer, base_model, head, torch_dtype, config = load_components(
        args.checkpoint_dir,
        args.base_model,
        args.dtype,
    )
    max_length = args.max_length or config.get("max_length", 512)
    predictions = run_streaming_inference(
        text.strip(),
        tokenizer,
        base_model,
        head,
        torch_dtype,
        max_length,
    )

    print("Token\tBoundary")
    for token, label in predictions:
        print(f"{token}\t{label}")


if __name__ == "__main__":
    main()
# python simple—test.py --text "１２月３１日，中共中央总书记发表新年讲话。"

