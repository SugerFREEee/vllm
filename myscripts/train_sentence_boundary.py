#!/usr/bin/env python3
"""Train a frozen Qwen3 backbone with a binary sentence-boundary head."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BIES_TO_BOUNDARY = {
    "B": 0,  # character still inside a word
    "I": 0,
    "E": 1,  # character completes a word (boundary)
    "S": 1,
}
BOUNDARY_LABEL2ID = {"incomplete": 0, "complete": 1}
ID2BOUNDARY_LABEL = {v: k for k, v in BOUNDARY_LABEL2ID.items()}
NUM_BOUNDARY_LABELS = len(BOUNDARY_LABEL2ID)


def ensure_pad_token(tokenizer) -> int:
    """Guarantee that the tokenizer exposes a padding token."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer.pad_token_id


def chunk_sequence(
    ids: Sequence[int],
    labels: Sequence[int],
    chunk_len: int,
) -> List[Dict[str, List[int]]]:
    """Split long sequences into max_length chunks."""
    results: List[Dict[str, List[int]]] = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_len, len(ids))
        results.append(
            {
                "ids": list(ids[start:end]),
                "labels": list(labels[start:end]),
            }
        )
        start = end
    return results


def encode_record(record: Dict, tokenizer, max_length: int) -> List[Dict[str, List[int]]]:
    """Convert a JSONL record into model-ready chunks."""
    chars = record.get("chars")
    tags = record.get("tags")
    if not chars or not tags or len(chars) != len(tags):
        return []
    token_ids: List[int] = []
    label_ids: List[int] = []
    for ch, tag in zip(chars, tags):
        label = BIES_TO_BOUNDARY.get(tag)
        if label is None:
            continue
        encoded = tokenizer.encode(ch, add_special_tokens=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        label_ids.extend([label] * len(encoded))
    usable_len = max_length
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    specials = 0
    if bos is not None:
        specials += 1
    if eos is not None:
        specials += 1
    chunk_len = usable_len - specials
    if chunk_len <= 0:
        raise ValueError("max_length too small to accommodate special tokens.")
    return chunk_sequence(token_ids, label_ids, chunk_len)


@dataclass
class PackedSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class BoundaryJsonlDataset(Dataset):
    """Dataset wrapping BIES JSONL samples and mapping them to binary labels."""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = ensure_pad_token(tokenizer)
        self.samples: List[PackedSample] = []
        self._load()

    def _pack(self, ids: List[int], labels: List[int]) -> PackedSample:
        tokens: List[int] = []
        label_seq: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            tokens.append(self.tokenizer.bos_token_id)
            label_seq.append(-100)
        tokens.extend(ids)
        label_seq.extend(labels)
        if self.tokenizer.eos_token_id is not None:
            tokens.append(self.tokenizer.eos_token_id)
            label_seq.append(-100)
        if len(tokens) > self.max_length:
            raise ValueError("Packed sequence exceeds max_length after specials.")
        pad_len = self.max_length - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_id] * pad_len)
            label_seq.extend([-100] * pad_len)
        attention = [1] * (self.max_length - pad_len) + [0] * pad_len
        return PackedSample(tokens, attention, label_seq)

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found.")
        with self.path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for chunk in encode_record(record, self.tokenizer, self.max_length):
                    sample = self._pack(chunk["ids"], chunk["labels"])
                    self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"No usable samples built from {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(sample.attention_mask, dtype=torch.long),
            "labels": torch.tensor(sample.labels, dtype=torch.long),
        }


class SentenceBoundaryHead(nn.Module):
    """FC head fed with LM probabilities."""

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_BOUNDARY_LABELS),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.layer(probs)


class FrozenQwenBoundaryTagger(nn.Module):
    """Wrap Qwen3 logits with a trainable classifier head."""

    def __init__(
        self,
        base_model_name: str,
        hidden_size: int,
        head_dropout: float,
        torch_dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        self.base.eval()
        for param in self.base.parameters():
            param.requires_grad_(False)
        vocab_size = self.base.lm_head.out_features
        self.head = SentenceBoundaryHead(
            vocab_size, hidden_size, dropout=head_dropout
        ).to(device=device, dtype=torch_dtype)
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            base_logits = outputs.logits.detach()
        probs = torch.softmax(base_logits, dim=-1)
        return self.head(probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen Qwen3 with a binary sentence-boundary head."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to train.bies.jsonl",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        required=True,
        help="Path to dev/test bies jsonl for evaluation.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="/data/workspace/model/Qwen/Qwen3-0.6B",
        help="Path or HF hub id of the Qwen3 base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def make_dataloaders(
    train_path: Path,
    val_path: Path,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = BoundaryJsonlDataset(train_path, tokenizer, max_length)
    val_ds = BoundaryJsonlDataset(val_path, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def eval_model(
    model: FrozenQwenBoundaryTagger,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, NUM_BOUNDARY_LABELS), labels.view(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return avg_loss, accuracy


def dtype_from_str(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        padding_side="right",
    )
    ensure_pad_token(tokenizer)
    train_loader, val_loader = make_dataloaders(
        args.train_file,
        args.val_file,
        tokenizer,
        args.max_length,
        args.batch_size,
    )
    dtype = dtype_from_str(args.dtype)
    model = FrozenQwenBoundaryTagger(
        base_model_name=args.pretrained_model,
        hidden_size=args.hidden_size,
        head_dropout=args.dropout,
        torch_dtype=dtype,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(warmup_steps, 0),
        num_training_steps=max(total_steps, 1),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, NUM_BOUNDARY_LABELS), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                print(
                    f"[Epoch {epoch} | Step {global_step}] "
                    f"loss={avg_loss:.4f}"
                )
                running_loss = 0.0
        val_loss, val_acc = eval_model(model, val_loader, device, loss_fn)
        
        print(
            f"[Epoch {epoch}] val_loss={val_loss:.4f}, val_acc={val_acc:.4%}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "boundary_head": model.head.state_dict(),
                "config": vars(args),
                "tag2id": BOUNDARY_LABEL2ID,
            }
            torch.save(ckpt, args.output_dir / "best_boundary_head.pt")
            tokenizer.save_pretrained(args.output_dir)
            print(
                f"New best accuracy {val_acc:.4%}, checkpoint saved to "
                f"{args.output_dir}"
            )
    print("Training complete.")


if __name__ == "__main__":
    main()
"""
python scripts/train_sentence_boundary.py \
    --train-file data/pku_bies/train.bies.jsonl \
    --val-file data/pku_bies/dev.bies.jsonl \
    --output-dir checkpoints/sb-head \
    --epochs 2000 \
    --max-length 256 \
    --log-interval 100 \
    --batch-size 32 


"""
