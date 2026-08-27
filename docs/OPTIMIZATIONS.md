# Optimizations, Provenance and Originality

The engineering in this repository falls into three groups: code reused from
kimi-k3-in-c, DeepSeek-V4-Flash-0731 adaptations of its ideas, and original
engineering mechanisms added by this project. Here, "original" means a concrete
mechanism designed in this project relative to the upstream baseline. It does
not claim that general concepts such as SIMD, caching or speculative decoding
were invented here.

## 1. Reused code

| area | reused foundation | current files |
|---|---|---|
| Safetensors I/O | header scanning, FNV-1a tensor index, bounds checks and the `pread` / `O_DIRECT` framework | `src/io/k3_st.c`, `src/io/k3_st.h` |
| Tokenizer | byte-level BPE, Unicode category tables and tokenizer data structures | `third_party/tok.h`, `third_party/tok_unicode*.h` |
| JSON | small header-only parser | `third_party/json.h` |

These files were subsequently extended for DeepSeek dtypes, pre-tokenization,
memory release and defensive checks. [NOTICE](../NOTICE) is authoritative for
copyright and licensing.

## 2. Adaptations of upstream ideas

| upstream idea | adaptation in this project |
|---|---|
| Stream cold MoE experts from disk | Bind the 43-layer DeepSeek routing graph, six routed experts and the shared expert while preserving expert accumulation order |
| Compute directly from compressed expert weights | Support native MXFP4 E2M1 weights and E8M0 scales without expanding the complete expert set in RAM |
| Bounded expert cache | Rework slots for the new expert size, layer count, router types and memory tiers while keeping output independent of the budget |
| Coalesced expert reads | Match the contiguous gate/up/down layout with fewer bounded system calls and a direct-I/O fallback |
| Single-process C99 + OpenMP inference | Implement Hyper-Connection, compressed sliding-window attention, YaRN RoPE, Hadamard, the DeepSeek tokenizer and chat template |
| Incremental context | Retain DeepSeek KV, compressor and Hyper-Connection state so later messages do not replay the transcript |
| Independent tiny-model correctness gate | Build a four-layer DeepSeek graph and compare every float against an independent scalar Python implementation |

## 3. Original engineering mechanisms in this project

These 16 entries are the complete top-level list relative to the
kimi-k3-in-c baseline. Each entry contains several lower-level operators or
scheduling details.

1. **Layer-major batching:** load one layer and advance the full prefill or
   verification batch through it in sequence while retaining recurrent-state
   semantics.
2. **Multi-token operator batching:** make FP8, FP4 and BF16 GEMV, attention
   projections, Hyper-Connection and the vocabulary head process several
   positions at once, sharing weight reads and decoded rows.
3. **Parameter-free speculative decoding (prompt lookup):** use a continuation
   found in committed history as a short draft and verify it with one full-model
   batch. Rejected tokens are removed while scalar RNG, EOS and pending-token
   order remain unchanged.
4. **Double-buffered MoE expert streaming:** compute the current expert group
   while loading the next. Long batches use at most half a layer partition per
   group, preserving independent victim slots and blocking stale cache hits.
5. **MoE I/O-compute overlap:** begin cache hits and the shared expert immediately,
   dispatch each miss when its read completes, and merge outputs in fixed
   expert-ID order.
6. **Read-pool reuse:** retain a bounded set of read workers for the model
   lifetime instead of creating them in every layer or competing unboundedly
   with OpenMP.
7. **Per-layer expert caching:** partition ordinary LRU by layer to prevent
   cross-layer eviction; verification can lend slots to hash-router layers
   temporarily without copying expert buffers.
8. **Hot-weight residency:** keep a decoded BF16 prefix of costly `wo_a`
   projections resident while preserving a minimum expert-cache budget.
9. **Memory-mapped layer weights and an inter-layer I/O pipeline:** map
   non-expert tensors and, while the CPU computes layer L, use kernel read-ahead
   plus a background read worker to load an independent layer L+1 bundle.
10. **In-kernel low-bit dequantization and joint gate/up GEMV execution:** decode
    FP4 with an in-register lookup and use SIMD bit conversion for FP8 on the
    fast path, all inside GEMV without materializing full floating-point
    matrices. On the single-token path, gate and up share the quantized input
    and one OpenMP dispatch; each AVX2 lane maps to a separate output row without
    changing column accumulation order.
11. **Activation-quantization reuse and preallocated workspaces:** quantize each
    activation once and share the result across output rows and related
    projections. Allocate MoE, attention and routing workspaces when the model
    opens, recycle them throughout inference, cache RoPE frequencies for the
    same position, and reuse query/indexer results.
12. **Chunked streaming LM head:** read and compute the roughly 1 GiB vocabulary
    head in 8 MiB blocks, decoding each BF16 weight row once for several outputs
    during batched verification.
13. **Laptop-aware thread scheduling:** use passive waits across long I/O and a
    bounded spin across short kernel gaps; stop automatic thread selection
    before memory bandwidth saturates.
14. **Resource-aware automatic planning:** derive context, resident `wo_a`,
    expert slots, compute threads and I/O workers from physical RAM,
    `MemAvailable` and CPU capacity while retaining reclaim headroom.
15. **Cross-turn reuse:** retain model, KV/compressor, expert cache and hot
    weights across chat turns, process only new input, stream tokens immediately,
    and avoid reloading weights after cancellation or `/reset`.
16. **Native resident chat API:** retain the checkpoint, hot weights, expert
    cache, tokenizer and worker pools across stateless Chat Completions
    requests; reconstruct the official role/EOS token sequence from complete
    message history and emit UTF-8-safe SSE deltas without another runtime.

Primary implementation locations: model scheduling, I/O, caching and
speculative verification are in
[`src/dsv4/dsv4_model.c`](../src/dsv4/dsv4_model.c) and
[`src/dsv4/dsv4_internal.h`](../src/dsv4/dsv4_internal.h); quantized and batched
kernels are in [`src/dsv4/dsv4_ops.c`](../src/dsv4/dsv4_ops.c); resource
planning is in [`src/dsv4/dsv4_config.c`](../src/dsv4/dsv4_config.c) and
[`scripts/try-dsv4.sh`](../scripts/try-dsv4.sh); resident chat, speculation and
streaming output are in [`src/cli/dsv4_run.c`](../src/cli/dsv4_run.c).
Bounded JSON parsing, message reconstruction and UTF-8 boundary handling are in
[`src/cli/dsv4_http.c`](../src/cli/dsv4_http.c). Numerical and transport tests
are in
[`tests/unit/test_dsv4_ops.c`](../tests/unit/test_dsv4_ops.c),
[`tests/unit/test_dsv4_model.c`](../tests/unit/test_dsv4_model.c) and
[`tests/unit/test_dsv4_prompt.c`](../tests/unit/test_dsv4_prompt.c), plus
[`tests/unit/test_dsv4_http.c`](../tests/unit/test_dsv4_http.c).

## Terminology and evidence

Prompt lookup is a parameter-free speculative-decoding pattern. The original
work claimed here is the concrete combination of full-model verification, state
restore, batched operators and expert-cache scheduling, not speculative decoding
as a general concept. Expert caching, prefetch and multi-batch pipelines are
likewise established research areas; this project's boundary is the specific
lossless CPU implementation and the combination designed for laptop memory
pressure. Related terminology can
be compared with the official
[llama.cpp speculative-decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md),
[MoE-Infinity](https://arxiv.org/abs/2401.14361) and
[Klotski](https://arxiv.org/abs/2502.06888).

See the [performance audit](PERFORMANCE_AUDIT.md) for controlled A/B evidence
and rejected experiments, [validation](VALIDATION.md) for the numerical
contract, and [architecture](DSV4_ARCHITECTURE.md) for the implemented graph.
