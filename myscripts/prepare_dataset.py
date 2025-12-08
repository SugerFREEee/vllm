#!/usr/bin/env python3
"""
Convert PKU Chinese word segmentation data into B/I/E/S label format.

The script accepts either of the following input formats:
1. Tokenized sentences where each line contains whitespace separated words.
2. Character-tag pairs where each non-empty line looks like `<char>\\tB-CWS`.

The output is a JSONL file per split where each row stores `text`, `chars`
and corresponding `tags` using the plain B/I/E/S scheme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

VALID_TAGS = {"B", "I", "E", "S"}


def detect_format(path: Path, sample_size: int = 200) -> str:
    """Detect whether a file contains word-level or char-tag data."""
    char_tag_lines = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            parts = stripped.split()
            if len(parts) == 2 and parts[1].split("-")[0] in VALID_TAGS:
                char_tag_lines += 1
            if total >= sample_size:
                break
    if total == 0:
        raise ValueError(f"{path} appears to be empty.")
    ratio = char_tag_lines / total
    return "char_tag" if ratio >= 0.5 else "word"


def parse_word_line(line: str) -> Tuple[List[str], List[str]]:
    """Convert a whitespace tokenized sentence into chars and BIES tags."""
    words = [token for token in line.strip().split() if token]
    chars: List[str] = []
    tags: List[str] = []
    for word in words:
        characters = list(word)
        chars.extend(characters)
        if len(characters) == 1:
            tags.append("S")
            continue
        tags.extend(["B"] + ["I"] * (len(characters) - 2) + ["E"])
    return chars, tags


def parse_char_tag_file(path: Path) -> Iterable[Tuple[List[str], List[str]]]:
    """Yield sentences from an already labeled file."""
    chars: List[str] = []
    tags: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if chars:
                    yield chars, tags
                    chars, tags = [], []
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Unexpected line `{line}` while parsing {path}. "
                    "Expecting `<char> <tag>` format."
                )
            char, tag = parts
            normalized_tag = tag.split("-")[0]
            if normalized_tag not in VALID_TAGS:
                raise ValueError(
                    f"Tag `{tag}` inside {path} is not a valid BIES tag."
                )
            chars.append(char)
            tags.append(normalized_tag)
    if chars:
        yield chars, tags


def parse_word_file(path: Path) -> Iterable[Tuple[List[str], List[str]]]:
    """Yield sentences by converting word-level annotations to BIES."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            chars, tags = parse_word_line(stripped)
            if not chars:
                continue
            yield chars, tags


def write_jsonl(
    samples: Sequence[Tuple[List[str], List[str]]],
    output_path: Path,
) -> None:
    """Persist processed samples to disk."""
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, (chars, tags) in enumerate(samples):
            if len(chars) != len(tags):
                raise ValueError(
                    f"Sample #{idx} in {output_path} has mismatched lengths: "
                    f"{len(chars)} chars vs {len(tags)} tags."
                )
            payload = {
                "text": "".join(chars),
                "chars": chars,
                "tags": tags,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_split(input_file: Path, output_file: Path) -> int:
    """Read one split and persist its BIES representation."""
    fmt = detect_format(input_file)
    if fmt == "char_tag":
        samples = list(parse_char_tag_file(input_file))
    else:
        samples = list(parse_word_file(input_file))
    if not samples:
        raise ValueError(f"No usable samples extracted from {input_file}.")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples, output_file)
    return len(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PKU word segmentation data to BIES JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing PKU train/dev/test txt files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where converted JSONL files will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=["train.txt", "dev.txt", "test.txt"],
        help="Relative file names under input-dir to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    summary = []
    for name in args.files:
        input_path = input_dir / name
        if not input_path.exists():
            print(f"[WARN] Skip missing {input_path}")
            continue
        output_path = output_dir / f"{Path(name).stem}.bies.jsonl"
        num_samples = process_split(input_path, output_path)
        summary.append(f"{name} -> {output_path.name}: {num_samples} samples")
    if not summary:
        raise SystemExit("No files were processed. Check --files parameter.")
    print("Conversion finished:")
    for line in summary:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
