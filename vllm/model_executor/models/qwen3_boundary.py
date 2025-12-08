# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming sentence-boundary model built on top of Qwen3."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from vllm.config import PoolerConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .interfaces_base import VllmModelForPooling, default_pooling_type
from .qwen3 import Qwen3Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class SentenceBoundaryHead(nn.Module):
    """Two-layer MLP that consumes LM probabilities."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.layer(probs)


@default_pooling_type("ALL")
class Qwen3BoundaryForStreaming(nn.Module, SupportsPP, VllmModelForPooling):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        boundary_cfg = getattr(config, "sentence_boundary", None)
        if boundary_cfg is None:
            raise ValueError(
                "sentence_boundary configuration is required for "
                "Qwen3BoundaryForStreaming."
            )
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        self.num_labels = int(boundary_cfg.get("num_labels", 2))
        hidden_size = int(boundary_cfg.get("hidden_size", 512))
        dropout = float(boundary_cfg.get("dropout", 0.1))
        vocab_size = config.vocab_size
        self.boundary_head = SentenceBoundaryHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            dropout=dropout,
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.pooler = DispatchPooler({
            "encode":
            Pooler.for_encode(
                PoolerConfig(
                    pooling_type="ALL",
                    normalize=False,
                    dimensions=None,
                    enable_chunked_processing=True,
                    activation=False,
                    softmax=False,
                )),
        })

        head_file = boundary_cfg.get("head_file")
        if head_file is None:
            logger.warning(
                "sentence_boundary.head_file missing; boundary head "
                "weights must be provided via checkpoint loading."
            )
        self.boundary_head_file = head_file

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        hidden_states = hidden_states[:, None, :]
        lm_logits = self.logits_processor(self.lm_head,
                                          hidden_states,
                                          sampling_metadata=None,
                                          prune_hidden_states=False)
        probs = torch.softmax(lm_logits, dim=-1)
        boundary_logits = self.boundary_head(probs)
        return torch.cat([boundary_logits, hidden_states], dim=-1)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights)
        if self._load_boundary_head():
            loaded |= {
                f"boundary_head.{name}"
                for name in self.boundary_head.state_dict().keys()
            }
        return loaded

    def _load_boundary_head(self) -> bool:
        if self.boundary_head_file is None:
            logger.warning(
                "Boundary head file not specified; using randomly "
                "initialized head weights."
            )
            return False
        model_dir = getattr(self.config, "_name_or_path", None)
        if not model_dir:
            logger.warning(
                "Config does not expose _name_or_path; cannot load "
                "boundary head weights."
            )
            return False
        head_path = Path(model_dir) / self.boundary_head_file
        if not head_path.exists():
            logger.warning("Boundary head file %s not found.", head_path)
            return False
        state_dict = torch.load(head_path,
                                map_location="cpu",
                                weights_only=False)
        self.boundary_head.load_state_dict(state_dict)
        return True
