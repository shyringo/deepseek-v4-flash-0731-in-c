# 优化、来源与原创边界

本项目的工程工作分为三类：直接复用 kimi-k3-in-c 的代码、基于其思路完成的
DeepSeek-V4-Flash-0731 适配，以及本项目新增的原创工程机制。这里的“原创”指
相对上游基线，由本项目针对这套模型和笔记本 CPU 推理约束设计的具体机制；
不把 SIMD、缓存、投机解码等已有通用概念包装成学术首创。

## 1. 直接复用的代码

| 部分 | 复用内容 | 当前文件 |
|---|---|---|
| Safetensors 读取基础 | 头部扫描、FNV-1a tensor 索引、边界检查、`pread` / `O_DIRECT` 读取框架 | `src/io/k3_st.c`、`src/io/k3_st.h` |
| Tokenizer 基础 | byte-level BPE、Unicode 分类表和 tokenizer 数据结构 | `third_party/tok.h`、`third_party/tok_unicode*.h` |
| JSON 解析 | 轻量级 header-only JSON 解析器 | `third_party/json.h` |

这些文件后续有 DeepSeek 数据类型、预分词规则、内存释放和安全检查等修改；
版权与许可证归属以仓库根目录的 [NOTICE](../NOTICE) 为准。

## 2. 基于上游思路完成的适配

| 上游思路 | 本项目中的适配 |
|---|---|
| 冷 MoE 专家从磁盘流式读取 | 适配 43 层 DeepSeek 路由、6 个 routed experts 和 shared expert，并保持专家贡献的固定累加顺序 |
| 压缩专家权重直接参与计算 | 适配 DeepSeek 原生 MXFP4 E2M1 权重和 E8M0 scale，不在内存里展开全部专家 |
| 有界专家缓存 | 适配新的专家大小、层数、路由类型和内存档位；内存只影响速度，不改变输出 |
| 合并连续权重读取 | 按一个专家的连续 gate/up/down 数据布局减少系统调用，同时保留边界检查和 direct-I/O fallback |
| C99 + OpenMP 单进程推理 | 适配 Hyper-Connection、压缩滑窗注意力、YaRN RoPE、Hadamard、DeepSeek tokenizer 和 chat template |
| 增量上下文 | 适配 DeepSeek 的 KV、compressor 和 Hyper-Connection 状态，使后续输入不必重放整段对话 |
| 独立小模型正确性门禁 | 构建四层 DeepSeek tiny graph，并与独立 Python 标量实现逐 float 对照 |
| DeepSeek-V4 DSML 工具协议 | 把公开的函数 schema、invoke/parameter grammar、assistant 历史与工具结果布局适配为有界 C 请求/输出处理 |

## 3. 本项目原创工程机制

下面 16 项是相对 kimi-k3-in-c 基线新设计并实现的机制，也是 README 中“原创
优化”的完整一级清单。每一项下面仍包含多个算子或调度细节。

1. **分层批处理（Layer-major batching）**：一层权重只加载一次，按原顺序推进
   整段 prefill 或 verification batch，同时保留递归状态的语义。
2. **多 token 算子批处理**：FP8、FP4、BF16 GEMV、attention projection、
   Hyper-Connection 和 vocabulary head 一次处理多个位置，共享权重读取和解码。
3. **无参数投机解码（Prompt lookup）**：从已提交上下文寻找重复片段，用历史
   continuation 生成短 draft，再由完整主模型批量验证；拒绝部分不会输出或留在
   上下文中，采样 RNG、EOS 和 pending-token 顺序与普通解码一致。
4. **MoE 专家双缓冲（Double buffering）**：当前 expert group 计算时读取下一组；
   长 batch 每组最多使用一半层缓存，为下一组留下独立 victim slots，并用
   stale-tag guard 避免读到正在替换的缓存项。
5. **MoE I/O 与计算重叠（I/O-compute overlap）**：cache hit 和 shared expert 立即
   计算；每个 miss 一读完就加入计算，不等待整层 I/O 完成，最终仍按 expert ID
   顺序合并输出。
6. **读取线程池复用**：固定的读取 worker 随模型常驻，不在每层反复创建；线程数
   经过实测设上限，避免磁盘 I/O 与 OpenMP 计算线程过度竞争。
7. **分层专家缓存（Per-layer expert cache）**：正常解码按层划分 LRU，避免层间
   相互驱逐；批量验证时可临时调整 hash-router 层的 slots，不复制专家权重。
8. **热权重常驻（Hot-weight residency）**：把反复使用且解码代价高的 `wo_a`
   前缀以 BF16 留在内存，同时保证专家缓存的最低预算。
9. **层权重内存映射与跨层 I/O 流水线**：非专家 tensor 通过 memory mapping
   使用；CPU 计算第 L 层时，内核 read-ahead 和后台读取线程并行加载第 L+1 层的
   独立 layer bundle。
10. **低比特融合反量化与 Gate/Up 双投影联合计算**：FP4 通过寄存器内查表、
    FP8 快速路径通过 SIMD 位域转换，在 GEMV 内部完成反量化，无需展开完整
    浮点矩阵；单 token 路径的 Gate 和 Up 共用已量化输入与一次 OpenMP 调度。
    AVX2 每个 lane 对应独立输出行，不改变列方向的浮点累加顺序。
11. **激活量化结果复用与工作区预分配**：同一份激活只做一次 FP8 量化，结果
    供所有输出行和多个相关投影共用；MoE、attention、route 的工作区在模型启动
    时统一分配并循环使用，同时缓存同一位置的 RoPE 频率并复用 query/indexer
    结果。
12. **词表头分块流式计算（Chunked streaming LM head）**：约 1 GiB 的
    vocabulary head 按 8 MiB 分块读取和计算；批量验证时，同一 BF16 权重行
    解码一次即可服务多个输出。
13. **适合笔记本的线程调度**：长 I/O 等待使用 passive wait，短 kernel 间隙使用
    有界 spin；自动线程数在内存带宽饱和前止步，而不是占满逻辑处理器。
14. **自适应资源规划**：根据物理内存、`MemAvailable` 和 CPU 自动分配上下文、
    `wo_a` 常驻层、专家 slots、计算线程和读取 workers，并预留系统回收空间。
15. **跨轮次复用**：常驻 chat 保留模型、KV/compressor、专家缓存和热权重，只
    处理新增输入；token 立即写到 stdout，中止回答或 `/reset` 都不重新加载权重。
16. **原生常驻聊天接口**：无状态 Chat Completions 请求复用 checkpoint、热权重、
    专家缓存、tokenizer 和 worker pools；重建角色/EOS/工具历史，校验已声明的
    函数调用与匹配结果，并直接输出 UTF-8 安全的 SSE delta，不依赖其他运行时。

主要实现位置：模型调度、I/O、缓存和投机验证位于
[`src/dsv4/dsv4_model.c`](../src/dsv4/dsv4_model.c) 与
[`src/dsv4/dsv4_internal.h`](../src/dsv4/dsv4_internal.h)；量化和批量内核位于
[`src/dsv4/dsv4_ops.c`](../src/dsv4/dsv4_ops.c)；资源规划位于
[`src/dsv4/dsv4_config.c`](../src/dsv4/dsv4_config.c) 和
[`scripts/try-dsv4.sh`](../scripts/try-dsv4.sh)；常驻对话、投机调度与流式输出位于
[`src/cli/dsv4_run.c`](../src/cli/dsv4_run.c)；有界 JSON 解析、消息序列重建和
UTF-8 边界处理位于 [`src/cli/dsv4_http.c`](../src/cli/dsv4_http.c)，DSML 校验
位于 [`src/cli/dsv4_dsml.c`](../src/cli/dsv4_dsml.c)。相关测试位于
[`tests/unit/test_dsv4_ops.c`](../tests/unit/test_dsv4_ops.c)、
[`tests/unit/test_dsv4_model.c`](../tests/unit/test_dsv4_model.c) 和
[`tests/unit/test_dsv4_prompt.c`](../tests/unit/test_dsv4_prompt.c)，以及
[`tests/unit/test_dsv4_http.c`](../tests/unit/test_dsv4_http.c) 和
[`tests/unit/test_dsv4_dsml.c`](../tests/unit/test_dsv4_dsml.c)。

## 术语与证据

Prompt lookup 属于无参数投机解码范式；本项目的原创点是完整主模型验证、状态
恢复、批量算子和专家缓存调度的具体组合，而不是“投机解码”这个概念本身。
类似地，专家缓存、预取、multi-batch pipeline 都是已有研究方向，本项目的原创
边界在具体的 lossless CPU 实现和面向笔记本内存压力的组合策略。可对照
[llama.cpp speculative decoding 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)、
[MoE-Infinity](https://arxiv.org/abs/2401.14361) 和
[Klotski](https://arxiv.org/abs/2502.06888)。

每项性能结论、A/B 条件和未采用实验见 [性能审计](PERFORMANCE_AUDIT.md)，数值
正确性契约见 [推理验证](VALIDATION.md)，模型图见
[DeepSeek-V4 架构说明](DSV4_ARCHITECTURE.md)。
