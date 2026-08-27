# 本地 OpenAI 兼容接口

原生服务会让 DeepSeek-V4-Flash-0731、热权重、专家缓存、tokenizer 和读取/计算
线程池保持加载。本地应用可以连续发送请求，不必反复打开 167 GB 权重；整个
服务不依赖 Python 或其他推理框架。

## 启动服务

```bash
scripts/try-dsv4.sh --server 8080
```

脚本会像终端聊天一样自动规划内存、上下文、线程数和读盘方式。服务只监听
`127.0.0.1`。

| 设置 | 填写内容 |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| API Key | 不需要；如果客户端不允许留空，可以随便填一个非空字符串 |
| 模型名 | `deepseek-v4-flash-0731-in-c` |

按 `Ctrl+C` 停止服务。

## 接口

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
      {"role": "system", "content": "请用一段简洁的话回答。"},
      {"role": "user", "content": "科技的边界在哪里？"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

不使用流式模式时，返回标准 `chat.completion` JSON，其中包含 assistant 回复、
结束原因，以及输入、输出和总 token 数。

## 流式输出

设置 `stream: true`，即可在模型生成过程中接收标准 SSE chunk：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-0731-in-c",
    "messages": [{"role": "user", "content": "科技的边界在哪里？"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

所有事件共用同一个 completion ID 和创建时间。服务依次发送 assistant 角色、
UTF-8 安全的内容增量、`finish_reason`、可选 usage，最后发送 `data: [DONE]`。

## 支持的请求参数

- `model`：`deepseek-v4-flash-0731-in-c` 或 `deepseek-v4-flash-0731`。
- `messages`：最多 128 条纯文本消息。开头可以放 `system` 和 `developer`；
  之后按 `user`、`assistant` 交替排列，并以 `user` 结束。
- `max_tokens` 或 `max_completion_tokens`：1 到 65,536。启动服务时没有指定
  `--max-tokens`，默认上限为 1,024。
- `temperature`：0 到 2。
- `top_p`：大于 0、不超过 1。
- `presence_penalty`：省略或设为 0；当前 sampler 没有实现该惩罚。
- `seed`：非负整数。
- `stream`：设为 `true` 时使用 SSE；也可以省略或设为 `false`。
- `stream_options.include_usage`：流式输出结束前发送 usage。
- `n`：省略或设为 1。

请求中没有填写的采样参数，会沿用启动服务时的命令行设置。每个请求都是无状态
的，客户端需要提交完整对话历史；权重和缓存保持加载，但模型上下文与随机数状态
会在处理请求前重置。

## 当前范围

服务面向一台笔记本上的单个使用者，每次处理一个请求。目前不支持工具调用、
结构化输出、多模态消息、身份验证、TLS 或局域网监听；遇到这些参数会明确报错，
不会静默忽略。请求体最大 1 MiB，解码后的消息文本最大 256 KiB。
