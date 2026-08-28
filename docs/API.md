# Local OpenAI-compatible API

The native server keeps DeepSeek-V4-Flash-0731, its hot weights, expert cache,
tokenizer and worker pools loaded so local applications can send more than one
request without reopening the 167 GB checkpoint. It uses no Python service or
external inference runtime.

## Start the server

```bash
scripts/try-dsv4.sh --server 8080
```

The launcher applies the same automatic memory, context, thread and I/O plan as
terminal chat. The server listens only on `127.0.0.1`.

| setting | value |
|---|---|
| base URL | `http://127.0.0.1:8080/v1` |
| API key | not required; use any non-empty placeholder if a client requires one |
| model | `deepseek-v4-flash-0731-in-c` |

Stop the service with `Ctrl+C`.

## Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731-in-c",
    "messages": [
      {"role": "system", "content": "Answer in one concise paragraph."},
      {"role": "user", "content": "Where are the boundaries of technology?"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The non-streaming response is a `chat.completion` object with an assistant
message, finish reason, and prompt/completion/total token counts.

## Streaming

Set `stream` to `true` for standard server-sent Chat Completions chunks:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731-in-c",
    "messages": [{"role": "user", "content": "Where are the boundaries of technology?"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Every event uses the same completion ID and creation time. The stream sends an
assistant-role chunk, UTF-8-safe content deltas, a final `finish_reason`, an
optional usage chunk, and `data: [DONE]`. Tool-capable streams send an SSE
keep-alive comment every 10 seconds during long prefill/generation, then emit a
complete validated `delta.tool_calls` entry.

## Function tools

Function tools work in non-thinking server mode with `tool_choice: "auto"` or
`"none"`. The engine renders the DeepSeek-V4 DSML contract, validates that the
model called a function declared by the request, and returns ordinary OpenAI
`tool_calls` objects. It does not execute the function.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731-in-c",
    "messages": [{"role": "user", "content": "What is 17 plus 25? Use the tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "add_numbers",
        "description": "Add two numbers",
        "parameters": {
          "type": "object",
          "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
          "required": ["a", "b"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0
  }'
```

A successful call has `finish_reason: "tool_calls"` and arguments such as
`{"a":17,"b":25}`. Send the returned assistant message back in `messages`,
then append one `tool` message for every call:

```json
{"role":"tool","tool_call_id":"CALL_ID_FROM_RESPONSE","content":"42"}
```

Parallel calls are supported. Every returned call ID must receive exactly one
result; unknown, duplicate, or missing IDs are rejected before model execution.
The full checkpoint was verified with string and numeric arguments, two
parallel calls, a complete result round trip, SSE, and the official OpenAI
JavaScript SDK.

DeepSeek-V4-Flash-0731 can emit a small set of malformed DSML closing or short
tag variants without grammar-constrained decoding. The parser recovers only
observed unambiguous forms after a declared function name and complete
parameters; ambiguous, truncated, undeclared, duplicate, or invalid-JSON calls
fail closed.

## Supported request fields

- `model`: `deepseek-v4-flash-0731-in-c` or `deepseek-v4-flash-0731`.
- `messages`: up to 128 messages. Initial `system` / `developer` instructions,
  ordinary `user` / `assistant` history, assistant `tool_calls`, and matched
  `tool` results are supported. Consecutive user-side messages are joined with
  a blank line, matching the published DeepSeek-V4 template.
- `tools`: up to 64 function definitions. Function names must be unique.
- `tool_choice`: omitted / `auto`, or `none`. `required` and named forced
  choices are rejected.
- `max_tokens` or `max_completion_tokens`: 1 through 65,536. The server default
  is 1,024 unless `--max-tokens` changes it at startup.
- `temperature`: 0 through 2.
- `top_p`: greater than 0 and at most 1.
- `presence_penalty`: omitted or 0; this sampler does not implement a penalty.
- `seed`: a non-negative integer.
- `stream`: `true` for SSE delivery; otherwise omitted or `false`.
- `stream_options.include_usage`: add the final usage chunk when streaming.
- `n`: omitted or `1`.

Sampling fields omitted from a request inherit the command-line values used to
start the server. Each request is stateless and must send its complete
conversation history. Model weights and runtime caches stay resident, while
context and RNG state are reset before the request is evaluated.

## Current scope

The server is intended for one laptop user and processes one request at a time.
Tool calling currently requires non-thinking server mode; `strict: true`
schemas are rejected because this release does not implement full JSON Schema
constrained decoding. Structured output, multimodal content, authentication,
TLS, and remote-network listening are not implemented. Unsupported features
return a clear error rather than being silently ignored. Request bodies are
limited to 1 MiB and decoded message text to 256 KiB.

The endpoint can be configured as a custom OpenAI Completions provider in
DeepSeek Harness, but default headless code mode is not claimed as a practical
laptop path: the observed Harness request carried an 8,916-token system/tool
prompt and a one-token first-step output cap. The smaller direct Chat
Completions and OpenAI SDK paths above are the validated integrations.
