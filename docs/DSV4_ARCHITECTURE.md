# DSV4 Architecture

This document records the architecture facts this engine implements, all read
from the fixed ModelScope revision `f981a343464c25f82b901e5882716b3b2fa514de`
and verified against the released `inference/model.py` + `inference/kernel.py`.

## Model dimensions

| field | value |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 43 |
| vocab_size | 129280 |
| num_attention_heads | 64 |
| num_key_value_heads | 1 |
| head_dim | 512 |
| qk_rope_head_dim | 64 |
| q_lora_rank | 1024 |
| o_groups | 8 |
| o_lora_rank | 1024 |
| sliding_window | 128 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 |
| num_hash_layers | 3 |
| moe_intermediate_size | 2048 |
| hc_mult | 4 |
| hc_sinkhorn_iters | 20 |
| index_n_heads | 64 |
| index_head_dim | 128 |
| index_topk | 512 |
| rope_theta | 10000 |
| compress_rope_theta | 160000 |
| rope_scaling | YaRN factor 16, beta_fast 32, beta_slow 1, original 65536 |

`compress_ratios` has 46 entries: the first 43 are the main model
(`0, 0, 4, 128, ...`, alternating ratio-4/ratio-128 after two zeros), and the
trailing 3 zeros belong to the optional experimental `mtp.*` DSpark stages.
There are 21 ratio-4 layers and 20 ratio-128 layers.

## Checkpoint layout

48 shards, 166,886,535,336 bytes on disk; the index total_size is
166,878,536,440 bytes across 67,612 main tensors + 4,705 optional `mtp.*`
tensors. The engine reads the main tensors only.

### Dtypes

- BF16 trunk: embeddings, norms, gates, compressor projections,
  `weights_proj`.
- FP8 E4M3 qmats with per-`[128,128]` E8M0 scales: `wq_a`, `wq_b`, `wkv`,
  `wo_a`, `wo_b`, and shared experts `w1/w3/w2`.
- FP4 E2M1 packed (two nibbles/byte, low nibble first) with per-row per-32
  E8M0 scales: routed experts `w1/w3/w2`.
- F32 small vectors: `hc_*_fn/base/scale`, `attn_sink`, `ape`, `bias`.
- I64 routing tables: `layers.{0,1,2}.ffn.gate.tid2eid`.

### Tensor contract (per layer L)

```
layers.L.attn_norm.weight            BF16 [4096]
layers.L.ffn_norm.weight             BF16 [4096]
layers.L.hc_attn_fn                  F32  [24, 16384]
layers.L.hc_attn_scale               F32  [3]
layers.L.hc_attn_base                F32  [24]
layers.L.hc_ffn_fn                   F32  [24, 16384]
layers.L.hc_ffn_scale                F32  [3]
layers.L.hc_ffn_base                 F32  [24]
layers.L.attn.wq_a.weight            F8_E4M3 [1024, 4096]   scale [8, 32]
layers.L.attn.wq_b.weight            F8_E4M3 [32768, 1024]  scale [256, 8]
layers.L.attn.wkv.weight             F8_E4M3 [512, 4096]    scale [4, 32]
layers.L.attn.wo_a.weight            F8_E4M3 [8192, 4096]   scale [64, 32]
layers.L.attn.wo_b.weight            F8_E4M3 [4096, 8192]   scale [32, 64]
layers.L.attn.q_norm.weight          BF16 [1024]
layers.L.attn.kv_norm.weight         BF16 [512]
layers.L.attn.attn_sink              F32  [64]
layers.L.ffn.gate.weight             BF16 [256, 4096]
layers.L.ffn.gate.tid2eid            I64  [129280, 6]      (first 3 layers)
layers.L.ffn.gate.bias               F32  [256]            (other layers)
layers.L.ffn.shared_experts.w1/w3.weight  F8_E4M3 [2048, 4096] scale [16, 32]
layers.L.ffn.shared_experts.w2.weight     F8_E4M3 [4096, 2048] scale [32, 16]
layers.L.ffn.experts.E.w1/w3.weight  I8  [2048, 2048]  scale [2048, 128]
layers.L.ffn.experts.E.w2.weight     I8  [4096, 1024]  scale [4096, 64]
```

When `ratio != 0` (coff = 2 for ratio 4, else 1):

```
layers.L.attn.compressor.ape         F32  [ratio, coff*512]
layers.L.attn.compressor.wkv.weight  BF16 [coff*512, 4096]
layers.L.attn.compressor.wgate.weight BF16 [coff*512, 4096]
layers.L.attn.compressor.norm.weight BF16 [512]
```

When `ratio == 4` (indexer):

```
layers.L.attn.indexer.wq_b.weight    F8_E4M3 [8192, 1024]  scale [64, 8]
layers.L.attn.indexer.weights_proj.weight BF16 [64, 4096]
layers.L.attn.indexer.compressor.ape F32  [4, 256]
layers.L.attn.indexer.compressor.wkv.weight  BF16 [256, 4096]
layers.L.attn.indexer.compressor.wgate.weight BF16 [256, 4096]
layers.L.attn.indexer.compressor.norm.weight  BF16 [128]
```

Globally: `embed.weight` BF16 [129280, 4096], `head.weight` BF16 [129280, 4096],
`norm.weight` BF16 [4096], `hc_head_fn` F32 [4, 16384], `hc_head_scale` F32 [1],
`hc_head_base` F32 [4].

One routed expert is 13,369,344 raw bytes: three packed FP4 matrices
(3 × 4,194,304) plus three E8M0 scale runs (3 × 262,144). The three scales form
one contiguous run (`w1,w2,w3`) and the three weights another, so a cache miss
needs only two page-aligned coalesced reads.

## Forward pass (per token)

```
embed row -> 4-way Hyper-Connection state
per layer:
  HC pre-reduce (attn) -> RMSNorm -> compressed/sliding MLA -> HC post-expand
  HC pre-reduce (ffn)  -> RMSNorm -> router + 6 FP4 experts + 1 shared FP8
                                     -> HC post-expand
HC head reduce -> RMSNorm -> streamed vocab head -> argmax
```

### Hyper-Connection

State is `[hc_mult=4, hidden=4096]`. `hc_pre` flattens the state, computes
`mixes = (fn @ flat) * rsqrt` with `rsqrt = 1/sqrt(mean(flat²)+eps)`, then runs
the Sinkhorn split: `pre = sigmoid(mixes[0:4]*scale0+base0)+eps`,
`post = 2*sigmoid(mixes[4:8]*scale1+base1)`, and the `4x4` combine matrix from
`mixes[8:24]`, row-softmaxed, one column normalisation, then
`iters-1 = 19` alternating row/column normalisations (final step = column).
`bin = pre . state`; after the branch, `state = post . branch_out + comb @ state`.
The head reduce uses `hc_head_fn` the same way, summing the four weighted
copies into `hidden`.

### Compressed sliding-window MLA

- `q = wq_b(q_norm(wq_a(x)))` reshaped `[64, 512]`, head-wise RMS-normalised,
  rope on the last 64 dims.
- `kv = wkv(x)`, RMS-normalised, rope on the last 64 dims, then the non-rope
  448 dims are FP8 QAT-simulated (group 64) back to BF16 and stored in a
  128-position ring window.
- Ratio-4 layers run an overlap-2 compressor (blocks of 4 positions pooled with
  a per-channel softmax gate into a 512-dim position) plus an indexer that picks
  the top-512 compressed positions; ratio-128 layers use a stride-128 compressor.
- The indexer compresses to 128 dims, Hadamard-rotates, FP4 QAT-simulates, and
  scores `sum_h relu(q_h · key_h) * w_h` to select positions.
- Attention uses a learnable per-head `attn_sink` and an online softmax; the
  output is inverse-roped, then projected through `wo_a` (FP8, grouped) and
  `wo_b` (BF16).

### Router and MoE

`z = gate(x)`, `score = sqrt(softplus(z))`, `choice = score + bias` selects the
top-6 (first 3 layers look the experts up in `tid2eid` instead). Combining
weights come from the unbiased `score`, normalised then scaled by `route_scale =
1.5`, and experts are dispatched in ascending id order. Routed experts use
packed FP4 with clipped SwiGLU (`swiglu_limit = 10`); the shared expert uses FP8
with weight 1.

### Quantisation

Activation FP8 QAT uses per-128 amax (floor 1e-4) with a power-of-two E8M0
scale and clamp to `[-448, 448]`; FP4 QAT uses per-32 amax (floor `6*2^-126`)
with clamp to `[-6, 6]`. E4M3 decode is the standard 3-bit-mantissa form; E2M1
is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with sign.

## Memory planning

`--memory-gib` buys runtime reserve (1.25 GiB), context memory, a persistent
prefix of the decoded `wo_a` projection, and then
`floor(remaining / slot)` expert cache slots. The planner always preserves at
least 1 GiB for experts before adding projection-cache layers. One expert slot
is 13,369,344 + 4×4096 bytes. Context memory is

```
43*128*512*4 + 21*ceil(C/4)*512*4 + 21*ceil(C/4)*128*4 + 20*ceil(C/128)*512*4
```

All other dense weights are resident only for the current layer. Once layer L
has loaded, the engine advises the kernel to read the non-expert tensors for
layer L+1 while L computes; `DSV4_NO_PREFETCH=1` disables this read-ahead.
During incremental decode, when `wo_a` is resident, a background POSIX worker
also loads the remaining layer L+1 bundle while the OpenMP team computes layer
L. `DSV4_NO_LAYER_OVERLAP=1` disables this explicit inter-layer I/O pipeline.
The 1 GiB vocab head streams in 1024-row (8 MiB) chunks.

When the CLI budget is automatic, it takes the smaller of two thirds of
physical RAM and three quarters of current Linux `MemAvailable`, rounded down.
This leaves headroom for the OS and reduces the chance that a larger expert
cache loses performance to reclaim or swap.

## Resident chat state

`--interactive` opens the checkpoint once and retains attention KV,
compressors, decoded `wo_a` layers and the expert LRU across user turns. A later
turn appends the prior response's pending final token, an EOS when the output
cap ended before one, and the next user/assistant role suffix. It then calls
`dsv4_prefill` at the next absolute position instead of replaying the existing
transcript. The tiny suite verifies that split resident prefill is bit-identical
to one contiguous prefill.

`--server PORT` uses a stateless variant of the same resident runtime. It binds
only to `127.0.0.1`, keeps the checkpoint mapping, decoded `wo_a` prefix,
expert cache, tokenizer and worker pools alive, and resets only model context
and sampling state between requests. The request's complete `messages` array is
rebuilt as the official BOS/system/user/assistant/EOS token sequence before one
layer-major prefill, so hidden state cannot leak from an earlier request.

Streaming requests receive standard Chat Completions SSE chunks. A small
boundary buffer joins token bytes that split a multibyte character before JSON
encoding, keeping CJK text and emoji valid UTF-8. The server handles one request
at a time because concurrent generations would need independent multi-gigabyte
context and expert-cache state on a laptop.

When a request declares function tools, their JSON schemas are appended to the
published DeepSeek-V4 DSML instruction block. Generated DSML is buffered until
the complete call is available, checked against the declared tool names, and
converted to ordinary Chat Completions `tool_calls`. Parallel call IDs are
unique and the next request must provide exactly one matching result per ID.
The engine never executes a tool. Tool-capable SSE connections receive periodic
comments during long model work, then a complete call delta and final usage.

## Prompt-lookup speculative verification

Before an ordinary decode step, the CLI searches the committed token history
for a repeated three- or four-token suffix. A historical continuation becomes
a draft; repeated evidence can extend a periodic draft to 16 tokens in the
automatic path. The internal diagnostic ceiling remains 63, so the current
pending token plus those drafts can fit one 64-position layer-major MoE batch.
Each row's target logit vector is then consumed in original token order.

For batches of at least four positions, independent Hyper-Connection work is
parallelised across positions. The vocabulary head decodes each BF16 row once
for up to seven outputs at a time. During a prompt-lookup verification round,
the cache transfers slots from learned-router layers to the three hash-router
layers without copying their expert buffers, then restores the ordinary layout
for scalar decode. These scheduling changes do not alter per-position
floating-point accumulation order.

When a multi-token MoE batch needs more unique experts than one layer's cache
partition can hold, each streamed group is capped at half of that partition.
This leaves a disjoint victim set for the next group's reads while the current
group computes. A pinned victim is excluded from hit lookup until its new cache
tag is published; the synchronous diagnostic path skips the double buffer.
Per-token expert contributions are still accumulated in ascending expert-ID
order.

Greedy decoding compares every draft against target argmax. Temperature/top-p
decoding calls the same sampler in the same delivered-token order as scalar
decode and stops at the first rejection, so a fixed seed produces the same
sequence. The context snapshot commits only `pending + accepted`; the target
token at the rejected row becomes the next pending token. Full acceptance
commits the whole verified block and uses its last logits for a bonus token.
An accepted EOS remains pending for the next interactive turn, preserving the
same `history_len == committed + 1` invariant as scalar decode.

One historical occurrence is capped at four drafts. Longer periodic extension
requires a second agreeing occurrence, and a zero-accept round starts an
eight-token cooldown. `--no-prompt-lookup` bypasses this scheduling layer; it
does not select a different numerical kernel.
