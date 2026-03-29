# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen2.5-Omni Thinker embedding model for trimodal embeddings.

Extends the Qwen2.5-Omni thinker (generation model) with pooling support
for text/image/audio → embedding tasks. Uses LAST token pooling
with L2 normalization.

Target models:
- LCO-Embedding/LCO-Embedding-Omni-7B
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.seqwise import pooler_for_embed
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces_base import default_pooling_type
from .qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from .utils import AutoWeightsLoader, WeightsMapper


class Qwen2_5OmniEmbeddingProcessingInfo(Qwen2_5OmniThinkerProcessingInfo):
    """Processing info that handles standalone thinker configs.

    LCO-Embedding models ship the thinker config directly as config.json
    (model_type: qwen2_5_omni_thinker) rather than wrapping it inside a
    parent Qwen2_5OmniConfig with a .thinker_config attribute. This
    subclass handles both layouts transparently.
    """

    def get_hf_config(self):
        hf_config = self.ctx.get_hf_config()
        if hasattr(hf_config, "thinker_config"):
            return hf_config.thinker_config
        return hf_config


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniEmbeddingProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class Qwen2_5OmniEmbeddingModel(Qwen2_5OmniThinkerForConditionalGeneration):
    """Qwen2.5-Omni thinker adapted for embedding inference.

    Adds sequence-level pooling (LAST token + L2 normalize) on top of
    the generation model's hidden states. Supports text, image, audio,
    and video inputs — any modality the thinker backbone accepts.

    The forward pass is inherited from the generation model: it returns
    hidden_states from the language model's transformer layers. The
    pooler then extracts the last-token embedding and L2-normalizes it.
    """

    is_pooling_model = True

    # Weight mapper for standalone thinker models (no "thinker." prefix).
    # More-specific prefixes must come first because WeightsMapper applies
    # ALL matching prefix rules sequentially.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Full Omni model weights (with thinker. prefix)
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
            # Standalone thinker weights (no thinker. prefix)
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        hf_config = vllm_config.model_config.hf_config

        # Standalone thinker configs (e.g. LCO-Embedding) lack
        # .thinker_config — add a self-reference so the parent __init__
        # can access it uniformly.
        if not hasattr(hf_config, "thinker_config"):
            hf_config.thinker_config = hf_config

        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = pooler_for_embed(pooler_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["talker.", "token2wav."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
