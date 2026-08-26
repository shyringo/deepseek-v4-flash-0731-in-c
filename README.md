<h1 align="center">DeepSeek-V4-Flash-0731 CPU Inference in C</h1>

<p align="center">
  <strong>Run the native DeepSeek-V4-Flash-0731 checkpoint locally on a single laptop CPU.</strong><br>
  A pure C local LLM and MoE inference engine, written in portable C99, with no GPU, CUDA, PyTorch or weight conversion.
</p>

<table align="center">
  <tr>
    <td align="center"><strong>284B-A13B</strong><br>parameters</td>
    <td align="center"><strong>8 GB</strong><br>minimum system RAM</td>
    <td align="center"><strong>0.892 s/token</strong><br>best TPOT on a 32 GB x86 laptop<br>1.12 token/s</td>
  </tr>
</table>

<p align="center">
  <a href="https://github.com/shyringo/deepseek-v4-flash-0731-in-c/actions/workflows/dsv4-ci.yml"><img src="https://github.com/shyringo/deepseek-v4-flash-0731-in-c/actions/workflows/dsv4-ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shyringo/deepseek-v4-flash-0731-in-c/releases"><img src="https://img.shields.io/github/v/release/shyringo/deepseek-v4-flash-0731-in-c" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/shyringo/deepseek-v4-flash-0731-in-c" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a><br>
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#measured-performance">Performance</a> ·
  <a href="#inference-optimizations-implemented-in-this-project">Inference optimizations</a>
</p>

## Quick start

You need a mainstream 64-bit CPU such as x86-64 or ARM64, a POSIX environment,
at least 8 GB of system RAM and about 172 GB of disk space. On Windows, use WSL2.

Before the first build, use your system package manager to install a C99
compiler, make, OpenMP, Git, curl and Bash. For example, on Ubuntu/WSL2 and
macOS:

```bash
# Ubuntu / WSL2
sudo apt-get update
sudo apt-get install -y build-essential git curl

# macOS (install Homebrew first)
xcode-select --install
brew install libomp
```

Then run the following commands in the same terminal:

```bash
# Get and build the project
git clone https://github.com/shyringo/deepseek-v4-flash-0731-in-c.git
cd deepseek-v4-flash-0731-in-c
make -j

# Download and verify the fixed ModelScope checkpoint (~166.9 GB; resumable)
scripts/download-dsv4.sh "$HOME/model/DeepSeek-V4-Flash-0731"

# Start a resident chat
scripts/try-dsv4.sh
```

Type a message when the chat starts. The launcher chooses the context, cache,
compute threads and I/O plan from the available RAM and CPU. Use `/reset` for a
new conversation, `Ctrl-C` to stop the current answer without unloading the
model, and `/exit` to quit.

Other common inference modes:

```bash
# Ask one question and exit when the answer finishes
scripts/try-dsv4.sh "Where are the boundaries of technology?"

# On a machine with enough RAM, use an explicit 18 GiB total budget
DSV4_MEMORY_GIB=18 scripts/try-dsv4.sh

# Run the binary directly with an explicit generation length
bin/dsv4 --model "$HOME/model/DeepSeek-V4-Flash-0731" \
  --prompt "Where are the boundaries of technology?" --max-tokens 64

# Start resident chat through the binary
bin/dsv4 --model "$HOME/model/DeepSeek-V4-Flash-0731" --interactive

# Show sampling, thinking, system-prompt and all other options
bin/dsv4 --help
```

## Requirements

- **CPU and system:** a mainstream 64-bit CPU such as x86-64 or ARM64, and an
  environment providing POSIX `mmap`, `pread` and pthreads. On Windows, use WSL2.
- **System RAM:** 8 GB minimum; more RAM can hold a larger expert cache.
- **Disk:** about 172 GB free for the 48 native checkpoint shards and headroom.
- **Tools:** a C99 compiler, make, OpenMP, Git, curl, Bash, and either
  `sha256sum` or `shasum`.

The automatic memory plan is the recommended default. More RAM is faster only
while it does not cause paging.

## Measured Performance

TTFT is the time from submitting a message until the first token is visible.
TPOT is the average interval over the remaining `N-1` output tokens. Here TTFT
starts immediately before model-ready prefill and ends when the first token is
written to the terminal; model opening and tokenizer initialization remain in
wall time but outside TTFT. Tokens stream as soon as they are available.

The table contains the best valid records that completed each workload and
passed the output checks. Values from different runs are never combined:

| workload | input / output tokens | memory plan | TTFT | TPOT | throughput | expert read | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary answer, no reusable text | 5 / 64 | 18 GiB | 20.203 s | 1.705 s/token | 0.59 token/s | 70.67 GiB | 21.95 GiB |
| same ordinary answer, lower memory budget | 5 / 64 | 15 GiB | 20.560 s | 1.791 s/token | 0.56 token/s | 77.84 GiB | 18.95 GiB |
| complete repeated continuation, prompt-lookup match | 10 / 60 | 18 GiB | 26.748 s | **0.892 s/token** | **1.12 token/s** | 53.64 GiB | 22.19 GiB |

The 0.892 s/token result completed all 20 requested lines and 60 generated
tokens. Four target-model verification rounds accepted 50 of 51 draft tokens;
every displayed token was confirmed by the full model. If an ordinary answer
has no reusable continuation, prompt lookup does not launch and adds no verify
rounds merely to improve a benchmark number.

A same-condition A/B shows the source of the gain. With prompt lookup disabled,
the complete repeated task measured 124.2 s wall, 27.932 s TTFT and 1.594
s/token TPOT. Enabling it measured 79.1 s, 22.875 s and 0.939 s/token, while
expert reads fell from 62.83 to 53.64 GiB; all 60 output IDs were identical. A
cold-page A/B for the long-prefill double buffer moved a 17-token prompt from
34.522 to 30.148 s TTFT, MoE time from 26.701 to 23.086 s, and expert wait from
22.034 to 12.796 s without changing the output token.

Reference environment:

| item | environment |
|---|---|
| host OS | Windows 11 with WSL2 |
| Linux | WSL2 Ubuntu 22.04.5 LTS, Linux 5.15.146.1-microsoft-standard-WSL2 |
| CPU / RAM | Intel Core i5-1340P, 12 cores / 16 logical CPUs; 31.65 GiB installed, 23.47 GiB visible to WSL2 |
| disk | Samsung MZVL41T0HBLB-00BH1 NVMe; checkpoint on `/dev/sdc` ext4 |
| build / threads | GCC 11.4.0, OpenMP, 12 threads, `OMP_WAIT_POLICY=PASSIVE` |
| inference configuration | 18 GiB total memory, 65536 context, AC power and active cooling |
| checkpoint | ModelScope revision `f981a343464c25f82b901e5882716b3b2fa514de` |

Laptop temperature, power state, competing processes and the file page cache
affect timing. See the [performance audit](docs/PERFORMANCE_AUDIT.md) for full
conditions, controlled comparisons and rejected experiments.

## Inference Optimizations Implemented in This Project

The engineering in this repository falls into three groups: code reused from
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
DeepSeek-V4-Flash-0731 adaptations of its ideas, and original inference
optimizations added here. The complete classification, provenance boundary and
code locations are in
[Optimizations and Provenance](docs/OPTIMIZATIONS.md). Only the third group is
listed below.

- **Layer-major batching.** Load one layer and advance the full prefill or
  verification batch through it in sequence before loading the next layer.
- **Multi-token operator batching.** FP8, FP4 and BF16 GEMV, attention
  projections, Hyper-Connection and the vocabulary head process several
  positions at once, sharing weight reads and decoded rows across the batch.
- **Parameter-free speculative decoding (prompt lookup).** When the committed
  context contains the same text, use its historical continuation as a short
  draft and verify it with one full-model batch. Rejected tokens are neither
  shown nor retained; identical-output A/B moved from 1.594 to 0.939 s/token.
- **Double-buffered MoE expert streaming.** Compute one expert group while the
  next loads. Long batches use at most half of a layer's cache per group, leaving
  independent slots for the next read instead of falling back to synchronous I/O.
- **MoE I/O-compute overlap.** Begin cached experts immediately and dispatch each
  disk-backed expert when its read completes, rather than waiting for every
  expert in the layer to arrive.
- **Read-pool reuse.** Keep a bounded set of read workers for the model lifetime
  instead of creating them in every layer. The automatic cap is four workers so
  storage I/O does not compete unboundedly with OpenMP compute.
- **Per-layer expert caching.** Give each layer its own LRU share so another
  layer cannot immediately evict its active experts; verification can rebalance
  those shares temporarily without copying expert weights.
- **Hot-weight residency.** Keep costly, repeatedly used `wo_a` projections in
  RAM when the budget permits, while preserving enough space for the expert
  cache so one resident tier does not starve the other.
- **Memory-mapped layer weights and an inter-layer I/O pipeline.** Map non-expert
  weights into the process address space; while the CPU computes layer L, kernel
  read-ahead and a background read worker load layer L+1 concurrently.
- **In-kernel low-bit dequantization and joint gate/up GEMV execution.** Decode
  FP4 with an in-register lookup and use SIMD bit conversion for FP8 on the fast
  path, all inside GEMV without materializing full floating-point matrices. On
  the single-token path, gate and up share the quantized input and one OpenMP
  dispatch; AVX2 processes eight output rows without changing each row's
  accumulation order.
- **Activation-quantization reuse and preallocated workspaces.** Quantize each
  activation once and share the result across output rows and related
  projections. Allocate the MoE, attention and routing workspaces when the model
  opens and recycle them throughout inference, keeping allocation off the hot path.
- **Chunked streaming LM head.** Read and compute the roughly 1 GiB vocabulary
  head in 8 MiB chunks instead of keeping it all resident; batched verification
  decodes each weight row once for several output tokens.
- **Laptop-aware thread scheduling.** Sleep across long storage waits, pause only
  briefly between short kernels, and stop automatic thread selection before
  memory bandwidth saturates.
- **Resource-aware automatic planning.** Use physical RAM, currently available
  memory and CPU capacity to budget context, hot weights, expert cache, compute
  threads and read workers. Manual tuning still needs only one memory value.
- **Cross-turn reuse.** Keep the model, KV/compressor state, expert cache and hot
  weights resident across chat turns, so `/reset`, cancellation and the next
  message never reload the 167 GB checkpoint.

## Inference Correctness

```bash
make test
```

This suite does not require the 167 GB checkpoint. The repository includes a
small four-layer model that runs for 130 positions and must match an independent
Python implementation with `maxdiff=0.000000`. The full checkpoint has a
separate fixed 16-token expected answer. See [Validation](docs/VALIDATION.md) for
the complete commands and evidence.

See [DSV4_ARCHITECTURE.md](docs/DSV4_ARCHITECTURE.md) for the model graph and
memory layout.

## Related Project

For a smaller dense-model option, see
[qwen3.8-27b-in-c](https://github.com/shyringo/qwen3.8-27b-in-c): run
Qwen3.8-27B GGUFs on one laptop CPU, with a tested 8 GB memory path and up to
2.52 token/s on the reference machine.

## License and Acknowledgements

The code is licensed under Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

This project is based on
[FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c).
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) demonstrated that
cold MoE experts can stream from disk so the complete checkpoint does not need
to reside in system RAM. Files reused or adapted by this project are listed
individually in [NOTICE](NOTICE).

The DeepSeek-V4-Flash-0731 checkpoint is published by DeepSeek-AI on
[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731).
This repository does not distribute model weights.
