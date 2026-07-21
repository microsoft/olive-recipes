"""Model loaders, wrapper modules, dummy inputs and I/O specs for exporting the
HuggingFace ``MoonshineStreamingForConditionalGeneration`` model into the five
stateful ONNX graphs consumed by onnxruntime-genai's ``streaming_enc_dec_asr``
model type (frontend / encoder / adapter / cross_kv / decoder_kv).

Design principle: reuse the HF submodules verbatim inside thin ``nn.Module``
wrappers so the exported math is numerically identical to the reference model.
The only component that is *reimplemented* is the frontend, whose two causal
convolutions must be made stateful (left padding replaced by carried buffers)
so the model can run chunk-by-chunk.

Runs under the ``moonshine`` conda env (transformers>=5.2 + torch + onnxscript).

Both the standalone exporter (``export_moonshine_streaming.py``) and the Olive
recipe JSONs import from this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MoonshineStreamingForConditionalGeneration
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    apply_rotary_pos_emb,
)

NEG_INF = float("-inf")

# Cache full models by name so the five per-component loaders don't each reload
# the ~200MB checkpoint from disk.
_MODEL_CACHE: dict[str, MoonshineStreamingForConditionalGeneration] = {}


def load_full_model(model_name: str) -> MoonshineStreamingForConditionalGeneration:
    """Load (and cache) the fp32, eval, eager-attention reference model."""
    if model_name not in _MODEL_CACHE:
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            model_name, attn_implementation="eager"
        )
        model = model.to(torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def model_dims(model: MoonshineStreamingForConditionalGeneration) -> dict:
    """Extract the dimensions needed to build dummy inputs / configs, so the
    same code works for both the small and tiny checkpoints."""
    enc = model.model.encoder
    dec = model.model.decoder
    sa0 = dec.layers[0].self_attn
    return {
        "encoder_dim": enc.embedder.conv1.in_channels,          # 620 (small)
        "conv1_channels": enc.embedder.conv1.in_channels,       # 620  (conv1 input buffer)
        "conv2_channels": enc.embedder.conv2.in_channels,       # 1240 (conv2 input buffer)
        "frame_len": int(enc.embedder.frame_len),               # 80
        "left_pad1": int(enc.embedder.conv1.left_pad),          # 4
        "left_pad2": int(enc.embedder.conv2.left_pad),          # 4
        "num_encoder_layers": len(enc.layers),                  # 10
        "num_decoder_layers": len(dec.layers),                  # 10
        "decoder_dim": dec.embed_tokens.embedding_dim,          # 512
        "num_kv_heads": sa0.config.num_key_value_heads,         # 8
        "head_dim": sa0.head_dim,                               # 64
        "vocab_size": model.proj_out.out_features,              # 32768
        "sample_buffer_size": int(enc.embedder.frame_len) - 1,  # 79
    }


# --------------------------------------------------------------------------- #
# 1. Frontend  (stateful streaming feature extractor)                         #
# --------------------------------------------------------------------------- #
class FrontendModule(nn.Module):
    """Streaming re-implementation of ``MoonshineStreamingEncoderEmbedder``.

    Non-streaming HF applies ``F.pad(x, (left_pad, 0))`` before each causal
    conv.  For chunked streaming we instead carry the last ``left_pad`` input
    columns of every conv in a buffer and run a *valid* (unpadded) convolution
    on ``cat(buffer, x)``.  Because a normal chunk is 8000 samples = 100 frames
    (even), the stride-2 phase stays aligned across chunks and the concatenated
    output is bit-for-bit identical to running the embedder on the full signal.

    Sub-frame audio (``total_samples % frame_len``) is carried in
    ``sample_buffer`` / ``sample_len`` and prepended to the next chunk so the
    framing is contiguous.
    """

    def __init__(self, full: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        emb = full.model.encoder.embedder
        self.cmvn = emb.cmvn
        self.comp = emb.comp
        self.linear = emb.linear
        self.conv1 = emb.conv1
        self.conv2 = emb.conv2
        self.frame_len = int(emb.frame_len)
        self.pad1 = int(emb.conv1.left_pad)
        self.pad2 = int(emb.conv2.left_pad)
        self.buf_size = self.frame_len - 1  # sample_buffer width (79)

    def forward(
        self,
        audio_chunk,      # [1, L]     float32
        sample_buffer,    # [1, 79]    float32
        sample_len,       # [1]        int64
        conv1_buffer,     # [1, C1, 4] float32  (last inputs of conv1)
        conv2_buffer,     # [1, C2, 4] float32  (last inputs of conv2)
        frame_count,      # [1]        int64
    ):
        # In the genai streaming pipeline chunk_samples is a multiple of
        # frame_len, so the carried sample_buffer is always empty on input
        # (sample_len == 0).  We therefore frame the chunk directly.  The
        # correct leftover state is still emitted so a final (flush) chunk of
        # non-multiple length threads/records its remainder before reset.
        total = audio_chunk.shape[1]
        n_frames = total // self.frame_len
        used = n_frames * self.frame_len

        frames = audio_chunk.narrow(1, 0, used).reshape(1, -1, self.frame_len)  # [1, nf, 80]
        leftover = audio_chunk.narrow(1, used, total - used)                    # [1, rem]
        rem = leftover.shape[1]
        sample_buffer_out = F.pad(leftover, (0, self.buf_size - rem))           # [1, 79]
        zero_i64 = frame_count * 0
        sample_len_out = zero_i64 + rem                                         # [1]

        # ---- per-frame feature transform (framing-invariant) ----
        h = self.cmvn(frames)                 # [1, nf, 80]
        h = self.comp(h)
        h = F.silu(self.linear(h))            # [1, nf, encoder_dim]
        h = h.transpose(1, 2)                 # [1, encoder_dim, nf]

        # ---- stateful causal conv 1 ----
        c1_in = torch.cat([conv1_buffer, h], dim=2)           # [1, C, pad1 + nf]
        conv1_buffer_out = c1_in[:, :, c1_in.shape[2] - self.pad1:]
        h = F.conv1d(
            c1_in, self.conv1.weight, self.conv1.bias,
            stride=self.conv1.stride, dilation=self.conv1.dilation,
        )
        h = F.silu(h)

        # ---- stateful causal conv 2 ----
        c2_in = torch.cat([conv2_buffer, h], dim=2)           # [1, C2, pad2 + o1]
        conv2_buffer_out = c2_in[:, :, c2_in.shape[2] - self.pad2:]
        h = F.conv1d(
            c2_in, self.conv2.weight, self.conv2.bias,
            stride=self.conv2.stride, dilation=self.conv2.dilation,
        )
        features = h.transpose(1, 2)          # [1, feat_len, encoder_dim]

        frame_count_out = frame_count + n_frames
        return (
            features,
            sample_buffer_out,
            sample_len_out,
            conv1_buffer_out,
            conv2_buffer_out,
            frame_count_out,
        )


# --------------------------------------------------------------------------- #
# 2. Encoder  (sliding-window bidirectional transformer, mask built from T)   #
# --------------------------------------------------------------------------- #
class EncoderModule(nn.Module):
    """Reuses the HF encoder layers + final norm.  The per-layer sliding-window
    attention mask is rebuilt from the dynamic sequence length because the
    genai encoder graph takes no external mask input."""

    def __init__(self, full: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        enc = full.model.encoder
        self.layers = enc.layers
        self.final_norm = enc.final_norm
        self.windows = [tuple(int(v) for v in w) for w in enc.config.sliding_windows]

    @staticmethod
    def _sliding_mask(seq_len, left, right, device, dtype):
        idx = torch.arange(seq_len, device=device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)            # q - k, [T, T]
        allowed = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
        mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, NEG_INF)
        return mask.unsqueeze(0).unsqueeze(0)                 # [1, 1, T, T]

    def forward(self, features):                              # [1, T, encoder_dim]
        hidden = features
        seq_len = hidden.shape[1]
        for layer, (left, right) in zip(self.layers, self.windows):
            mask = self._sliding_mask(seq_len, left, right, hidden.device, hidden.dtype)
            hidden = layer(hidden, attention_mask=mask)
        return self.final_norm(hidden)                        # [1, T, encoder_dim]


# --------------------------------------------------------------------------- #
# 3. Adapter  (positional embedding + projection to decoder dim)              #
# --------------------------------------------------------------------------- #
class AdapterModule(nn.Module):
    """memory = proj(encoded + pos_emb(arange(T) + pos_offset)).  Mirrors the
    top of ``MoonshineStreamingDecoder.forward``."""

    def __init__(self, full: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        dec = full.model.decoder
        self.pos_emb = dec.pos_emb
        self.proj = dec.proj

    def forward(self, encoded, pos_offset):                  # [1,T,enc], [1]
        seq_len = encoded.shape[1]
        offset = pos_offset.reshape(()).to(torch.long)
        positions = torch.arange(seq_len, device=encoded.device) + offset
        hidden = encoded + self.pos_emb(positions)
        return self.proj(hidden)                             # [1, T, decoder_dim]


# --------------------------------------------------------------------------- #
# 4. Cross-KV  (project memory into per-layer cross-attention key/value)      #
# --------------------------------------------------------------------------- #
class CrossKvModule(nn.Module):
    """Stack every decoder layer's ``encoder_attn`` k/v projection of memory."""

    def __init__(self, full: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.layers = full.model.decoder.layers
        a0 = self.layers[0].encoder_attn
        self.num_heads = a0.config.num_key_value_heads
        self.head_dim = a0.head_dim

    def forward(self, memory):                               # [1, T, decoder_dim]
        seq_len = memory.shape[1]
        keys, values = [], []
        for layer in self.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(memory).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(memory).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            keys.append(k)
            values.append(v)
        k_cross = torch.stack(keys, dim=0)                   # [L, 1, H, T, D]
        v_cross = torch.stack(values, dim=0)
        return k_cross, v_cross


# --------------------------------------------------------------------------- #
# 5. Decoder-KV  (autoregressive step with self-KV cache + precomputed cross) #
# --------------------------------------------------------------------------- #
class DecoderKvModule(nn.Module):
    """One decoder step over ``S`` query tokens.  Self-attention appends to the
    incoming self-KV cache (position derived from its length); cross-attention
    reuses the precomputed per-layer cross K/V.  Reuses every HF submodule
    (projections, layernorms, MLP, rotary embedding, lm head)."""

    def __init__(self, full: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        dec = full.model.decoder
        self.embed_tokens = dec.embed_tokens
        self.layers = dec.layers
        self.norm = dec.norm
        self.rotary_emb = dec.rotary_emb
        self.proj_out = full.proj_out
        a0 = dec.layers[0].self_attn
        self.num_heads = a0.config.num_key_value_heads
        self.head_dim = a0.head_dim

    def forward(self, token, k_self, v_self, out_k_cross, out_v_cross):
        # token [1,S] int64 ; *_self [L,1,H,Ts,D] ; out_*_cross [L,1,H,Tc,D]
        seq_len = token.shape[1]
        past_len = k_self.shape[3]
        hidden = self.embed_tokens(token)                    # [1, S, dec_dim]

        position_ids = torch.arange(
            past_len, past_len + seq_len, device=token.device
        ).unsqueeze(0)                                       # [1, S]
        cos, sin = self.rotary_emb(hidden, position_ids)

        # causal mask over [S query positions, past_len + S key positions]
        total = past_len + seq_len
        q_abs = position_ids.reshape(seq_len, 1)             # [S, 1]
        k_abs = torch.arange(total, device=token.device).reshape(1, total)
        causal = torch.zeros(seq_len, total, dtype=hidden.dtype, device=token.device)
        causal = causal.masked_fill(k_abs > q_abs, NEG_INF).unsqueeze(0).unsqueeze(0)

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            # ---- self attention (causal, rotary, cached) ----
            residual = hidden
            hs = layer.input_layernorm(hidden)
            sa = layer.self_attn
            q = sa.q_proj(hs).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = sa.k_proj(hs).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = sa.v_proj(hs).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_full = torch.cat([k_self[i], k], dim=2)        # [1, H, Ts+S, D]
            v_full = torch.cat([v_self[i], v], dim=2)
            new_k.append(k_full)
            new_v.append(v_full)
            scores = torch.matmul(q, k_full.transpose(2, 3)) * sa.scaling + causal
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v_full)
            ctx = ctx.transpose(1, 2).reshape(1, seq_len, -1)
            hidden = residual + sa.o_proj(ctx)

            # ---- cross attention (precomputed k/v, full) ----
            residual = hidden
            hs = layer.post_attention_layernorm(hidden)
            ca = layer.encoder_attn
            qc = ca.q_proj(hs).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(qc, out_k_cross[i].transpose(2, 3)) * ca.scaling
            ctx = torch.matmul(torch.softmax(scores, dim=-1), out_v_cross[i])
            ctx = ctx.transpose(1, 2).reshape(1, seq_len, -1)
            hidden = residual + ca.o_proj(ctx)

            # ---- feed forward ----
            residual = hidden
            hs = layer.final_layernorm(hidden)
            hidden = residual + layer.mlp(hs)

        hidden = self.norm(hidden)
        logits = self.proj_out(hidden)                       # [1, S, vocab]
        out_k_self = torch.stack(new_k, dim=0)               # [L, 1, H, Ts+S, D]
        out_v_self = torch.stack(new_v, dim=0)
        return logits, out_k_self, out_v_self, out_k_cross, out_v_cross


# --------------------------------------------------------------------------- #
# Olive-style per-component loaders                                           #
# --------------------------------------------------------------------------- #
def frontend_model_loader(model_name):
    return FrontendModule(load_full_model(model_name))


def encoder_model_loader(model_name):
    return EncoderModule(load_full_model(model_name))


def adapter_model_loader(model_name):
    return AdapterModule(load_full_model(model_name))


def cross_kv_model_loader(model_name):
    return CrossKvModule(load_full_model(model_name))


def decoder_kv_model_loader(model_name):
    return DecoderKvModule(load_full_model(model_name))


# --------------------------------------------------------------------------- #
# Dummy inputs (used both for tracing/export and Olive dummy_inputs_func)      #
# --------------------------------------------------------------------------- #
def _dims_from_module(module):
    """Recover the dims needed for dummy inputs from a wrapper instance."""
    full = _MODEL_CACHE[next(iter(_MODEL_CACHE))]
    return model_dims(full)


def frontend_dummy_inputs(model):
    d = model_dims(_any_full())
    return {
        "audio_chunk": torch.randn(1, model_chunk_samples(), dtype=torch.float32),
        "sample_buffer": torch.zeros(1, d["sample_buffer_size"], dtype=torch.float32),
        "sample_len": torch.zeros(1, dtype=torch.int64),
        "conv1_buffer": torch.zeros(1, d["conv1_channels"], d["left_pad1"], dtype=torch.float32),
        "conv2_buffer": torch.zeros(1, d["conv2_channels"], d["left_pad2"], dtype=torch.float32),
        "frame_count": torch.zeros(1, dtype=torch.int64),
    }


def encoder_dummy_inputs(model):
    d = model_dims(_any_full())
    return {"features": torch.randn(1, 48, d["encoder_dim"], dtype=torch.float32)}


def adapter_dummy_inputs(model):
    d = model_dims(_any_full())
    return {
        "encoded": torch.randn(1, 24, d["encoder_dim"], dtype=torch.float32),
        "pos_offset": torch.zeros(1, dtype=torch.int64),
    }


def cross_kv_dummy_inputs(model):
    d = model_dims(_any_full())
    return {"memory": torch.randn(1, 24, d["decoder_dim"], dtype=torch.float32)}


def decoder_kv_dummy_inputs(model):
    d = model_dims(_any_full())
    L, H, D = d["num_decoder_layers"], d["num_kv_heads"], d["head_dim"]
    past, cross = 6, 24
    return {
        "token": torch.ones(1, 4, dtype=torch.int64),
        "k_self": torch.randn(L, 1, H, past, D, dtype=torch.float32),
        "v_self": torch.randn(L, 1, H, past, D, dtype=torch.float32),
        "out_k_cross": torch.randn(L, 1, H, cross, D, dtype=torch.float32),
        "out_v_cross": torch.randn(L, 1, H, cross, D, dtype=torch.float32),
    }


# Helpers so dummy funcs work without receiving the model name -------------- #
_CHUNK_SAMPLES = 8000


def model_chunk_samples():
    return _CHUNK_SAMPLES


def set_chunk_samples(value):
    global _CHUNK_SAMPLES
    _CHUNK_SAMPLES = int(value)


def _any_full():
    if not _MODEL_CACHE:
        raise RuntimeError("No model loaded yet; call a *_model_loader first.")
    return next(iter(_MODEL_CACHE.values()))
