#!/usr/bin/env python3
"""Streaming example for Qwen3BoundaryForStreaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger("vllm.boundary")

BOUNDARY_LABELS = {
    0: "未完结",
    1: "完整",
}


def stream_tokenize_text(tokenizer, text: str) -> Tuple[List[List[int]], List[int]]:
    """Tokenize the text character by character to mimic streaming input."""
    chunks: List[List[int]] = []
    aggregated: List[int] = []
    for ch in text:
        char_token_ids = tokenizer.encode(ch, add_special_tokens=False)
        if not char_token_ids:
            continue
        aggregated.extend(char_token_ids)
        for token_id in char_token_ids:
            chunks.append([token_id])
    return chunks, aggregated


def decode_boundary_output(
    resp: PoolingRequestOutput[PoolingOutput],
    num_labels: int,
) -> torch.Tensor:
    if resp.outputs.data is None:
        return None
    logits = resp.outputs.data[..., :num_labels].to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    if probs.ndim == 2:
        probs = probs.unsqueeze(1)
    return probs


async def handle_request(
    engine: AsyncLLM,
    engine_args: AsyncEngineArgs,
    request_id: str,
    query_prompt: TokensPrompt,
    message_list: List[List[int]],
    full_token_ids: List[int],
    tokenizer,
    num_labels: int,
) -> None:
    logger.info("🚀 Start %s (%d initial tokens, %d chunks)", request_id,
                len(query_prompt["prompt_token_ids"]), len(message_list))

    response = engine.encode(
        query_prompt,
        pooling_params=PoolingParams(task="encode",
                                     output_kind=RequestOutputKind.DELTA),
        request_id=request_id,
        resumable=bool(message_list),
    )

    processed = 0
    chunk_idx = 0
    async for resp in response:
        if resp.outputs.data is not None:
            probs = decode_boundary_output(resp, num_labels)
            if probs is not None:
                preds = torch.argmax(probs, dim=-1)
                num_tokens = probs.shape[0]
                for i in range(num_tokens):
                    token_id = full_token_ids[processed + i]
                    token_text = tokenizer.decode([token_id],
                                                  clean_up_tokenization_spaces=False)
                    label_id = preds[i, 0].item()
                    label = BOUNDARY_LABELS.get(label_id, str(label_id))
                    confidence = probs[i, 0, label_id].item()
                    logger.info(
                        "   Token #%d: %s -> %s (%.3f)",
                        processed + i,
                        repr(token_text),
                        label,
                        confidence,
                    )
                processed += num_tokens
        if not message_list:
            continue
        next_chunk = message_list.pop(0)
        chunk_idx += 1
        logger.info("🔄 Resume %s chunk #%d (%d tokens)", request_id, chunk_idx,
                    len(next_chunk))
        await engine.resume_request(
            request_id=request_id,
            prompt_token_ids=next_chunk,
            finish_forever=not message_list,
        )


async def run_engine(
    engine: AsyncLLM,
    engine_args: AsyncEngineArgs,
    prompts: List[Tuple[str, TokensPrompt, List[List[int]], List[int]]],
    tokenizer,
    num_labels: int,
) -> None:
    limiter = asyncio.Semaphore(engine_args.max_num_seqs or 32)

    async def _wrapped(args):
        request_id, query_prompt, chunks, token_ids = args
        async with limiter:
            await handle_request(engine, engine_args, request_id, query_prompt,
                                 chunks, token_ids, tokenizer, num_labels)

    await asyncio.gather(*[asyncio.ensure_future(_wrapped(p)) for p in prompts])


def generate_prompts(
    engine_args: AsyncEngineArgs,
) -> Tuple[List[Tuple[str, TokensPrompt, List[List[int]], List[int]]], AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        engine_args.model,
        trust_remote_code=engine_args.trust_remote_code,
    )

    texts = [
        "１２月３１日，中共中央总书记、国家主席江泽民发表新年讲话。新华社记者兰红光摄，这是一九九八年新年讲话的现场记录。",
    ]

    prompts: List[Tuple[str, TokensPrompt, List[List[int]], List[int]]] = []
    for idx, text in enumerate(texts):
        chunks, token_ids = stream_tokenize_text(tokenizer, text)
        if not chunks:
            continue
        initial = chunks.pop(0)
        query_prompt = TokensPrompt(prompt_token_ids=initial)
        prompts.append(
            (f"boundary-{idx}", query_prompt, chunks, token_ids.copy()))
    return prompts, tokenizer


def parse_args():
    parser = FlexibleArgumentParser(
        description="Streaming example for Qwen3BoundaryForStreaming")
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


def init_boundary_engine(engine_args: AsyncEngineArgs) -> AsyncLLM:
    engine_args.runner = "pooling"
    engine_args.disable_log_stats = True
    engine_args.enable_chunked_prefill = False
    engine_args.convert = "none"
    return AsyncLLM.from_engine_args(engine_args,
                                     usage_context=UsageContext.API_SERVER)


async def main():
    args = parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    config = AutoConfig.from_pretrained(
        engine_args.model, trust_remote_code=engine_args.trust_remote_code)
    boundary_cfg = getattr(config, "sentence_boundary", None)
    if boundary_cfg is None:
        raise SystemExit("sentence_boundary config missing; "
                         "ensure the checkpoint was exported correctly.")
    num_labels = int(boundary_cfg.get("num_labels", 2))

    prompts, tokenizer = generate_prompts(engine_args)
    if not prompts:
        logger.error("No prompts generated, exiting.")
        return

    engine = init_boundary_engine(engine_args)
    start = time.perf_counter()
    await run_engine(engine, engine_args, prompts, tokenizer, num_labels)
    logger.info("🏁 Completed in %.2fs", time.perf_counter() - start)
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

"""
python3 examples/offline_inference/qwen_boundary_model.py \
     --model /data/workspace/model/Qwen/Qwen3Boundary-Stream \
     --trust-remote-code \
     --max-num-seqs 1 \
     --max-model-len 2048 \
     --disable-log-stats
"""
