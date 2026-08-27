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
optional usage chunk, and `data: [DONE]`.

## Supported request fields

- `model`: `deepseek-v4-flash-0731-in-c` or `deepseek-v4-flash-0731`.
- `messages`: up to 128 string-content messages. `system` and `developer`
  messages may appear at the beginning; the remaining messages alternate
  `user` and `assistant`, ending with `user`.
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
Tools, tool choice, structured output, multimodal content, authentication, TLS,
and remote-network listening are not implemented. Unsupported features return
a clear error rather than being silently ignored. Request bodies are limited to
1 MiB and decoded message text to 256 KiB.
