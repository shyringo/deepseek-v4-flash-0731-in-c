# Changelog

All notable changes to this project.

## [0.2.0]

### Added
- Loopback-only OpenAI-compatible Chat Completions server that keeps the model,
  hot weights, expert cache, tokenizer and worker pools resident.
- Non-streaming JSON responses and live UTF-8-safe SSE token chunks with
  optional final usage data.
- Bounded request parsing, complete message-history reconstruction using the
  official DeepSeek role/EOS sequence, and explicit API limits.
- HTTP/JSON/template tests in GNU make, CMake/CTest, strict, portable and
  sanitizer configurations.

### Changed
- Token output now uses one sink abstraction shared by terminal, buffered HTTP
  and SSE paths without changing sampling or model execution.
- `scripts/try-dsv4.sh --server PORT` applies the same automatic laptop memory,
  context, thread and I/O planning as terminal chat.

## [0.1.0]

Initial port from the upstream kimi-k3-in-c baseline
(`ff11dce`, Release v1.0.0) to DeepSeek-V4-Flash-0731:

### Added
- Pure C99 + OpenMP engine for the 43-layer FP8/FP4 graph
  (`bin/dsv4`), with streamed experts, memory planner, compressed
  sliding-window MLA, Hyper-Connection, router and tokenizer.
- Weightless test suite: config, ops, prompt, safetensors, tokenizer,
  and a 130-position tiny-model oracle crossing ratio-4/ratio-128.
- Downloader with resume workers (`scripts/download-dsv4.sh`), doctor
  (`scripts/doctor.sh`), try (`scripts/try-dsv4.sh`), examples, CMake/CTest,
  kernel benchmarks, checkpoint verifier, tokenizer parity suite.
- English and Chinese READMEs; architecture and performance docs; NOTICE.
- Resident multi-turn chat with streaming output, cancellation, reset, automatic
  memory/context/thread planning and TTFT/TPOT reporting.
- Lossless prompt lookup with full-model verification; the complete repeated-
  output reference task reaches 0.892 s/token on the documented laptop.
- Layer-major prefill, read-only layer mapping, bounded expert caching,
  persistent expert I/O workers and overlapped expert reads/computation.

### Changed
- `src/io/k3_st.*` extended with I8/I64/F8_E4M3/F8_E8M0 dtypes.
- `third_party/tok.h` extended with the DeepSeek pre-tokenizer family;
  generated `tok_unicode_ps.h`.

### Fixed
- Full-checkpoint greedy generation now reproduces the published 16-token
  oracle, while the independent tiny graph remains bit-exact.
- The download resume integration server now returns valid HTTP range responses
  and uses an isolated dynamic port, so repeated CI runs are deterministic.
