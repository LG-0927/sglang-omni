# SPDX-License-Identifier: Apache-2.0
"""Run mixed-progress streaming batches through the model's encoder layers."""

import torch
from torch.nn import functional as F

from sglang_omni.models.nemotron3_5_asr.attention_cache import (
    NemotronBatchAttentionCache,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
    NemotronAsrStreamingEncoderModelOutput,
)


def get_mixed_progress_audio_features(
    model: Nemotron3_5AsrForRNNT,
    input_features: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    attention_cache: NemotronBatchAttentionCache,
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache,
    num_lookahead_tokens: int,
) -> NemotronAsrStreamingEncoderModelOutput:
    encoder = model.encoder
    assert not model.training
    hidden_states = encoder.subsampling(
        input_features, attention_mask=None, padding_cache=padding_cache
    )
    hidden_states *= encoder.input_scale
    seq_length = hidden_states.shape[1]
    attention_mask = attention_cache.create_mask(
        seq_length,
        hidden_states.device,
        model.config.encoder_config.sliding_window - 1,
        num_lookahead_tokens,
    )
    position_embeddings = encoder.encode_positions(
        hidden_states, cached_frames=attention_mask.shape[-1] - seq_length
    )
    all_masked_rows = torch.all(~attention_mask, dim=-1)
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            all_masked_rows=all_masked_rows,
            position_embeddings=position_embeddings,
            past_key_values=attention_cache,
            padding_cache=padding_cache,
            use_cache=True,
        )

    prompt_ids = prompt_ids.to(hidden_states.device)
    one_hot = F.one_hot(prompt_ids, num_classes=model.config.num_prompts).to(
        hidden_states.dtype
    )
    one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
    fused = model.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
    return NemotronAsrStreamingEncoderModelOutput(
        last_hidden_state=hidden_states,
        pooler_output=model.encoder_projector(fused),
        past_key_values=attention_cache,
        padding_cache=padding_cache,
    )
