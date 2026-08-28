# Validation

This file records how this engine is validated against the released checkpoint.
The fixed revision is ModelScope `f981a343464c25f82b901e5882716b3b2fa514de`
(DeepSeek-V4-Flash-0731).

## Acceptance oracle (fixed, must never change)

Single-turn chat template, user text `a`, greedy temperature 0:

- rendered prompt SHA-256:
  `453281a61e27a36aa728a5028d12b50021cfa5d17234a68d81637c73ace37ea3`
- prompt token IDs: `0 128803 67 128804 128822`
- generated 16 token IDs:
  `19923 3 52780 236 271 2107 8777 1277 440 1438 49646 270 7010 582 67 2148`
- decoded text:

```
Hello!
It looks like you just typed the letter "a."
```

Any build, at any memory budget, must reproduce these IDs byte for byte.

## Acceptance checklist

1. `make test`, `make strict`, `make portable`, `make asan` pass.
2. The tiny-model graph test (`tests/unit/test_dsv4_model.c`) runs 130 positions
   against an independent Python scalar oracle and crosses the ratio-4 and
   ratio-128 compression boundaries.
3. Tokenizer parity against `tokenizers==0.22.0`: 22 fixed + 1,000 mixed
   Unicode + full P/S set all match (9,384 samples).
4. All 48 shards present with exact sizes and SHA-256; total 166,886,535,336
   bytes; 67,612 main tensors + 4,705 mtp.* tensors verified by
   `tools/verify_dsv4_checkpoint.py`.
5. The full 43-layer model runs end to end (no mocks, no truncated weights) and
   reproduces the 16-token oracle.
6. `--memory-gib` and `--threads` are honoured; weights are never committed.
7. Two memory budgets produce identical token IDs.

## Full-checkpoint run log

Reference run recorded 2026-08-16:

```
host OS:         Windows 11 with WSL2
guest:           Ubuntu 22.04.5 LTS, Linux 5.15.146.1-microsoft-standard-WSL2
CPU / RAM:       Intel Core i5-1340P (12 cores / 16 logical), 31.65 GiB installed / 23.47 GiB visible
storage:         Samsung MZVL41T0HBLB-00BH1 NVMe; /dev/sdc ext4
compiler:        GCC 11.4.0 + OpenMP
checkpoint:      ModelScope f981a343464c25f82b901e5882716b3b2fa514de
threads:         12 (explicit; automatic choice on this host)
expert I/O:      4 persistent workers (automatic)
OpenMP wait:     PASSIVE (CLI default)
power:           AC connected, HP Optimized, external fan active
context:         65536 (automatic)
memory plan:     14 GiB (explicit reference-machine automatic choice)
prompt tokens:   5
generated:       16
forward steps:   20
wall:            54.029 s (53.545 s profiled forward total)
TTFT:            23.679 s (model-ready prefill start to first token delivered)
TPOT:            1.991 s/token (15 intervals)
peak RSS:        19.29 GiB (20,711,010,304 bytes from rusage)
expert cache:    per-layer LRU, 745 slots
expert hits/misses:  2,446 / 2,363
overlapped misses:   2,363
expert bytes read:   29.42 GiB
page faults:         62,560 major / 604,454 minor
```

AC power and the external fan remained on throughout the run. No other CPU- or
disk-intensive process was active. Starting CPU temperature and page-cache
state were not fixed; those affect wall time, not the acceptance IDs.

## Automatic prompt-lookup checks

Prompt lookup is part of the default CLI path and is always verified by the
full model. The following checks were completed on 2026-08-16 with the same
checkpoint, ext4 storage, 15 GiB plan and 12 threads:

1. The normal default command produced the fixed 16-token oracle exactly. No
   suffix matched, so the optimisation performed zero verify rounds.
2. With the diagnostic minimum match reduced to one token, four rounds drafted
   14 tokens and accepted zero. The corrected output still matched an adjacent
   scalar run for all 32 generated IDs:

   ```
   19923 3 52780 236 271 2107 8777 1277 440 1438 49646 270 7010 582 67 2148
   53769 235 271 19884 440 2716 304 2997 3061 8029 418 396 7010 14 469 477
   ```

3. A periodic raw prompt at `temperature=0.1`, `top_p=1`, `seed=42` generated
   the same 16 IDs with prompt lookup and scalar decode. Prompt lookup accepted
   12/12 drafts and measured 0.889 s/token; scalar measured 1.483 s/token.
4. A greedy repeated-output task generated the identical token triple
   `33310 2058 201` 16 times in both paths. Automatic prompt lookup accepted
   40/40 drafts in two verification rounds, measuring 1.037 s/token versus
   1.770 s/token for `--no-prompt-lookup` in the controlled 15 GiB pair.
5. The primary complete-response check used `Say hello 20 times.`, an 18 GiB
   plan and the normal 16-draft default. Both paths produced the same 60 token
   IDs and all 20 requested lines. Prompt lookup accepted 50/51 drafts in four
   rounds and measured 79.1 seconds wall, 22.875 seconds TTFT and 0.939
   s/token TPOT. Same-binary scalar decode measured 124.2 seconds wall, 27.932
   seconds TTFT and 1.594 s/token TPOT. Expert reads fell from 62.83 to
   53.64 GiB.
   An earlier capped 48-token run reached 0.928 s/token but did not complete the
   requested 20 lines, so it is retained only as historical data rather than
   used as the user-facing result.
6. After the experimental split-read code was removed, an independent
   performance-freeze run repeated the complete 60-token task. It produced all
   20 lines, accepted 50/51 drafts and measured 0.892 s/token TPOT, 26.748 s
   TTFT, 53.64 GiB of expert reads and 22.19 GiB peak RSS. The fixed 16-token
   oracle and the tiny graph (`maxdiff=0.000000`) were also revalidated after
   that cleanup.

The tiny-model suite separately verifies snapshot restore and prefix commit
across the 128-token compressor boundary, plus a populated expert-cache
partition resize from 2 to 6 slots and back to 2. Subsequent logits must remain
bit-exact in both cases.

Record each validation machine with this template:

```
host:            (fill in)
compiler:        (fill in)
threads:         (fill in)
context:         (fill in)
memory plan:     (fill in)
prompt tokens:   (fill in)
generated:       (fill in)
forward steps:   (fill in)
wall:            (fill in)
TTFT:            (fill in; state timing boundary)
TPOT:            (fill in; state interval count)
peak RSS:        (fill in)
expert hits/misses:  (fill in)
overlapped misses:   (fill in)
expert bytes read:   (fill in)
```

When comparing memory budgets, record both runs and require identical token IDs.
The current sustained-performance run uses the five-token chat prompt `a` and
64 generated tokens. With streaming stdout and an explicit 18 GiB / 65536
context plan it measured 130.0 s wall, 20.203 s TTFT-to-delivery and 1.705
s/token TPOT over 63 intervals. It begins with the fixed 16-token oracle above.
A preceding memory-only A/B reproduced the same output IDs at 9, 12, 15 and 18
GiB. The fixed 16-token sequence remains the correctness oracle.

## Function-tool validation

The v0.3 tool path was tested against the same pinned full checkpoint with an
18 GiB plan, 4,096-token context, 12 threads, greedy sampling and non-thinking
mode. These are protocol/correctness checks, not performance records:

1. A `get_weather(city: string)` request returned
   `finish_reason=tool_calls`, a unique call ID, function `get_weather`, and
   arguments `{"city":"Beijing"}`.
2. Replaying that assistant message plus the matching tool result
   `25 C / sunny` produced a final answer using both values and
   `finish_reason=stop`.
3. An `add_numbers(a: number, b: number)` request preserved numeric JSON types:
   `{"a":17,"b":25}`. Whitespace before the DSML block became `content=null`.
4. A two-city request produced two uniquely identified parallel
   `get_weather` calls for Beijing and Shanghai. Replaying both matched results
   produced a comparison containing both supplied temperatures and conditions.
5. A tool-capable SSE request stayed connected for 488 seconds through 48
   keep-alive comments, then returned one completion ID, a standard
   `delta.tool_calls`, `finish_reason=tool_calls`, usage, and `[DONE]`.
6. `openai` JavaScript SDK 6.15.0 consumed the SSE endpoint directly and
   reconstructed `add_numbers` plus `{"a":17,"b":25}` without a custom stream
   parser.
7. DeepSeek Harness rc.2 registered the endpoint through an isolated custom
   `openai-completions` provider and sent its real headless request. The
   captured default code-mode prompt was 8,916 tokens and its first step asked
   for only one output token, so the expensive inference was stopped and no
   end-to-end Harness compatibility claim is made.

The unconstrained checkpoint emitted several deterministic but unambiguous
DSML variations across these prompts: shortened closing tags, one duplicated
`invoke` keyword, and short `invoke` / `parameter` tags. Fixtures preserve the
exact observed strings. Recovery is limited to a standard outer
`<｜DSML｜tool_calls>` block with a declared function, complete parameters and
valid JSON values. Unknown functions, ambiguous text, incomplete parameters,
duplicate names/results, missing results and invalid JSON fail closed.

Weightless tests exercise standard and observed DSML forms at every input
chunk size, string/numeric/boolean/array/object values, parallel calls,
angle-bracket values, UTF-8, tool-history reconstruction, and negative paths.
The complete strict, portable, ASan/UBSan and CMake/CTest suites pass, while the
four-layer model remains `maxdiff=0.000000`.

## What this does NOT claim

Without a compatible official CUDA/TileLang oracle, full-model **logit-level
parity** cannot be claimed. Fluent text is not proof of mathematical
correctness; the 16-token greedy oracle is the evidence we do have.
