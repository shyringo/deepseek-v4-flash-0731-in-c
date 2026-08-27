/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_run.c - the dsv4 command line interface.
 *
 *   bin/dsv4 --model DIR --prompt TEXT [options]
 *
 * See README.md for the full option table. The CLI renders the single-turn
 * chat template, tokenises with the vendored BPE (from the official
 * tokenizer.json), prefills every prompt token in order (only the last one
 * asks for logits), then decodes greedily. Generated text goes to stdout; the
 * complete token ID audit line goes to stderr.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <strings.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_http.h"
#include "dsv4_internal.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

static void usage(const char *prog)
{
    fprintf(stderr,
"usage: %s --model DIR [--prompt TEXT | --interactive] [options]\n"
"\n"
"  --model DIR           checkpoint directory (config.json + 48 safetensors)\n"
"  --prompt TEXT         user text (initial turn with --interactive)\n"
"  --interactive         keep model, caches and chat context resident\n"
"  --server PORT         loopback OpenAI-compatible HTTP server\n"
"  --validate-only       open and verify metadata, then exit\n"
"  --max-tokens N        optional hard cap per response (default: until EOS)\n"
"  --context N           context window (default: automatic from memory budget)\n"
"  --memory-gib N        total memory budget in GiB (default: automatic)\n"
"  --threads N           OpenMP threads (default: automatic, up to 12)\n"
"  --cache-gib N         advanced: expert cache GiB, exclusive with --memory-gib\n"
"  --temperature N       sampling temperature, 0 = greedy (default 1)\n"
"  --top-p N             nucleus sampling (default 1)\n"
"  --thinking            enable thinking mode (ends the prompt with <think>)\n"
"  --reasoning-effort E  low | high | max (thinking instruction prefix)\n"
"  --system TEXT         system prompt\n"
"  --raw-prompt          bypass the chat template (encode TEXT as-is)\n"
"  --no-prompt-lookup    disable lossless repeated-text acceleration\n"
"  --seed N              RNG seed for temperature > 0\n"
"  --help                this message\n",
            prog);
}

static int parse_int(const char *s, long *v)
{
    char *end = NULL;
    long x = strtol(s, &end, 10);
    if (!end || *end || end == s) return 0;
    *v = x;
    return 1;
}

static int parse_float(const char *s, double *v)
{
    char *end = NULL;
    double x = strtod(s, &end);
    if (!end || *end || end == s) return 0;
    *v = x;
    return 1;
}

typedef struct {
    double weight;
    int id;
} SampleItem;

typedef struct {
    int (*write)(void *context, const char *data, size_t length);
    void *context;
    int bytes;
    unsigned char last_byte;
    int failed;
    int append_final_newline;
} DSV4Output;

static int output_file_write(void *context, const char *data, size_t length)
{
    FILE *file = (FILE *)context;
    return fwrite(data, 1, length, file) == length && fflush(file) == 0;
}

static int output_write(DSV4Output *output, const char *data, size_t length)
{
    if (!output || output->failed || !output->write ||
        !output->write(output->context, data, length)) {
        if (output) output->failed = 1;
        return 0;
    }
    if (length) {
        output->bytes += (int)length;
        output->last_byte = (unsigned char)data[length - 1u];
    }
    return 1;
}

static int sample_item_desc(const void *ap, const void *bp)
{
    const SampleItem *a = (const SampleItem *)ap;
    const SampleItem *b = (const SampleItem *)bp;
    if (a->weight > b->weight) return -1;
    if (a->weight < b->weight) return 1;
    return (a->id > b->id) - (a->id < b->id);
}

static double linux_available_gib(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    }
    fclose(f);
    return (double)kib / (1024.0 * 1024.0);
}

static double automatic_memory_gib(double *visible_gib, double *available_gib)
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        *visible_gib = 0.0;
        *available_gib = 0.0;
        return 2.0;
    }

    const long double gib = 1024.0L * 1024.0L * 1024.0L;
    long double visible = (long double)pages * (long double)page_size / gib;
    long double budget = floorl(visible * 2.0L / 3.0L);
    *available_gib = linux_available_gib();
    if (*available_gib > 0.0) {
        long double available_budget = floorl((long double)*available_gib * 3.0L / 4.0L);
        if (available_budget < budget) budget = available_budget;
    }
    if (budget < 2.0L) budget = 2.0L;
    if (budget > 1024.0L) budget = 1024.0L;
    *visible_gib = (double)visible;
    return (double)budget;
}

/* deterministic xorshift64 for temperature > 0 */
static uint64_t rng_state;
static volatile sig_atomic_t stop_generation;

static void handle_sigint(int sig)
{
    (void)sig;
    stop_generation = 1;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    double started, load, attn, moe, expert_wait, expert_io, hc, head;
    long hits, misses, read_ops;
    uint64_t bytes;
} DSV4PhaseMark;

typedef struct {
    long calls, hits, misses, read_ops;
    uint64_t bytes;
    double wall, load, attn, moe, expert_wait, expert_io, hc, head;
} DSV4PhaseStats;

static DSV4PhaseMark dspark_phase_mark(const DSV4Model *m)
{
    DSV4PhaseMark mark;
    mark.started = now_s();
    mark.load = m->time_load;
    mark.attn = m->time_attn;
    mark.moe = m->time_moe;
    mark.expert_wait = m->time_expert_read;
    mark.expert_io = m->time_expert_io;
    mark.hc = m->time_hc;
    mark.head = m->time_head;
    mark.hits = m->cache_hits;
    mark.misses = m->cache_misses;
    mark.read_ops = m->expert_read_ops;
    mark.bytes = m->expert_bytes_read;
    return mark;
}

static void dspark_phase_add(DSV4PhaseStats *stats, DSV4PhaseMark mark,
                             const DSV4Model *m)
{
    stats->calls++;
    stats->wall += now_s() - mark.started;
    stats->load += m->time_load - mark.load;
    stats->attn += m->time_attn - mark.attn;
    stats->moe += m->time_moe - mark.moe;
    stats->expert_wait += m->time_expert_read - mark.expert_wait;
    stats->expert_io += m->time_expert_io - mark.expert_io;
    stats->hc += m->time_hc - mark.hc;
    stats->head += m->time_head - mark.head;
    stats->hits += m->cache_hits - mark.hits;
    stats->misses += m->cache_misses - mark.misses;
    stats->read_ops += m->expert_read_ops - mark.read_ops;
    stats->bytes += m->expert_bytes_read - mark.bytes;
}

static int dspark_propose_profiled(DSV4Model *m, int anchor_token,
                                   int anchor_position, int n_drafts,
                                   int *draft_tokens,
                                   DSV4PhaseStats *stats)
{
    DSV4PhaseMark mark = dspark_phase_mark(m);
    int rc = dsv4_dspark_propose(m, anchor_token, anchor_position,
                                 n_drafts, draft_tokens);
    dspark_phase_add(stats, mark, m);
    return rc;
}

static int dspark_prefill_capture_profiled(
    DSV4Model *m, const int *token_ids, int n_tokens, int start_position,
    float *last_logits, float *all_logits, const int *capture_layers,
    int n_capture_layers, float *capture_hidden, DSV4PhaseStats *stats)
{
    DSV4PhaseMark mark = dspark_phase_mark(m);
    int rc = dsv4_prefill_capture(m, token_ids, n_tokens, start_position,
                                  last_logits, all_logits, capture_layers,
                                  n_capture_layers, capture_hidden);
    dspark_phase_add(stats, mark, m);
    return rc;
}

static int dspark_commit_verified_prefix(
    DSV4Model *m, const DSV4Config *cfg, DSV4ContextSnapshot *snapshot,
    const int *verify_ids, int n_positions, int start_position, float *logits,
    float *capture_hidden, DSV4PhaseStats *replay_stats)
{
    if (getenv("DSV4_DSPARK_FORCE_REPLAY")) {
        if (dsv4_context_snapshot_restore(snapshot) != 0 ||
            dspark_prefill_capture_profiled(
                m, verify_ids, n_positions, start_position, logits, NULL,
                cfg->dspark_target_layer, cfg->dspark_stages, capture_hidden,
                replay_stats) != 0)
            return -1;
    } else if (dsv4_context_snapshot_commit_prefix(snapshot, n_positions) != 0) {
        return -1;
    }
    return dsv4_dspark_commit_target_hidden(m, capture_hidden, n_positions,
                                             start_position);
}

static void report_dspark_phase(const char *name, const DSV4PhaseStats *stats)
{
    fprintf(stderr,
            "spec profile: %-8s calls=%ld wall=%.3f load=%.3f "
            "attn=%.3f moe=%.3f expert_wait=%.3f expert_io=%.3f "
            "hc=%.3f head=%.3f hits=%ld misses=%ld reads=%ld "
            "bytes=%.2f GiB\n",
            name, stats->calls, stats->wall, stats->load, stats->attn,
            stats->moe, stats->expert_wait, stats->expert_io, stats->hc,
            stats->head, stats->hits, stats->misses, stats->read_ops,
            (double)stats->bytes / (1024.0 * 1024.0 * 1024.0));
}

static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}

#define DSV4_NGRAM_MAX 63
#define DSV4_NGRAM_DEFAULT 16

/* Draft only when a repeated suffix has an unambiguous historical
 * continuation. Verification remains authoritative, so this can affect
 * latency but never the generated token sequence. */
static int ngram_draft(const int *history, int n_history, int cap,
                       int min_match, int allow_overlap, int periodic_extend,
                       int *out)
{
    if (cap > DSV4_NGRAM_MAX) cap = DSV4_NGRAM_MAX;
    for (int n = 4; n >= min_match; n--) {
        if (n_history < n + 1) continue;
        int newest = -1, previous = -1;
        for (int start = n_history - n - 1; start >= 0; start--) {
            int match = 1;
            for (int i = 0; i < n; i++) {
                if (history[start + i] != history[n_history - n + i]) {
                    match = 0;
                    break;
                }
            }
            if (!match) continue;
            if (newest < 0) newest = start;
            else {
                previous = start;
                break;
            }
        }
        if (newest < 0) continue;

        const int available = n_history - newest - n;
        int draft_cap = cap;
        if (previous < 0 && draft_cap > 4) draft_cap = 4;
        int drafted = 0;
        for (int i = 0; drafted < draft_cap; i++) {
            if (i >= available && (!periodic_extend || previous < 0)) break;
            const int source_offset = i % available;
            const int candidate = history[newest + n + source_offset];
            if (previous >= 0) {
                const int previous_index = previous + n + source_offset;
                if ((!allow_overlap && previous_index >= newest) ||
                    history[previous_index] != candidate)
                    break;
            }
            out[drafted++] = candidate;
        }
        if (drafted > 0) return drafted;
    }
    return 0;
}

static int sample_top_p(const float *logits, int vocab, double temperature,
                        double top_p, SampleItem *items)
{
    double max_logit = logits[0];
    for (int v = 1; v < vocab; v++)
        if (logits[v] > max_logit) max_logit = logits[v];

    double total = 0.0;
    double inv_temperature = temperature > 1e-5 ? 1.0 / temperature : 1e5;
    for (int v = 0; v < vocab; v++) {
        items[v].weight = exp(((double)logits[v] - max_logit) * inv_temperature);
        items[v].id = v;
        total += items[v].weight;
    }
    /* Use 53 random bits so the uniform variate has double's full precision. */
    double uniform = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
    if (top_p == 1.0) {
        double target = uniform * total;
        double cumulative = 0.0;
        for (int i = 0; i < vocab; i++) {
            cumulative += items[i].weight;
            if (target < cumulative) return items[i].id;
        }
        return items[vocab - 1].id;
    }

    qsort(items, (size_t)vocab, sizeof(*items), sample_item_desc);
    double kept_sum = 0.0;
    int keep = vocab;
    double threshold = total * top_p;
    for (int i = 0; i < vocab; i++) {
        kept_sum += items[i].weight;
        if (kept_sum >= threshold) {
            keep = i + 1;
            break;
        }
    }

    double target = uniform * kept_sum;
    double cumulative = 0.0;
    for (int i = 0; i < keep; i++) {
        cumulative += items[i].weight;
        if (target < cumulative) return items[i].id;
    }
    return items[keep - 1].id;
}

static int argmax_logits(const float *logits, int vocab)
{
    int best = 0;
    for (int v = 1; v < vocab; v++)
        if (logits[v] > logits[best]) best = v;
    return best;
}

static void emit_token(Tok *tok, int token, char *piece, size_t piece_cap,
                       DSV4Output *output)
{
    if (token != 1) {
        int plen = tok_decode(tok, &token, 1, piece, (int)piece_cap - 1);
        if (plen > 0) {
            (void)output_write(output, piece, (size_t)plen);
        }
    }
}

static int prefill_with_dspark_state(DSV4Model *m, const DSV4Config *cfg,
                                     const int *tokens, int n_tokens,
                                     int start_position, float *logits)
{
    if (!dsv4_dspark_ready(m))
        return dsv4_prefill(m, tokens, n_tokens, start_position, logits);
    const size_t one = (size_t)cfg->dspark_stages * cfg->hidden;
    if ((size_t)n_tokens > SIZE_MAX / one / sizeof(float)) return -1;
    float *capture = (float *)malloc((size_t)n_tokens * one * sizeof(float));
    if (!capture) return -1;
    int rc = dsv4_prefill_capture(m, tokens, n_tokens, start_position, logits,
                                  NULL, cfg->dspark_target_layer,
                                  cfg->dspark_stages, capture);
    if (rc == 0)
        rc = dsv4_dspark_commit_target_hidden(m, capture, n_tokens,
                                               start_position);
    free(capture);
    return rc;
}

static int read_chat_line(char **line, size_t *cap)
{
    for (;;) {
        if (isatty(STDIN_FILENO)) {
            fputs("you> ", stdout);
            fflush(stdout);
        }
        ssize_t got = getline(line, cap, stdin);
        if (got < 0) return 0;
        while (got > 0 && ((*line)[got - 1] == '\n' || (*line)[got - 1] == '\r'))
            (*line)[--got] = '\0';
        if (!strcmp(*line, "/exit") || !strcmp(*line, "/quit")) return 0;
        if (!strcmp(*line, "/help")) {
            fprintf(stderr, "interactive commands: /help, /reset, /exit, /quit\n");
            continue;
        }
        if (got > 0) return 1;
    }
}

typedef struct {
    char method[8];
    char path[128];
    char *body;
    size_t body_length;
} DSV4IncomingRequest;

typedef struct DSV4ServerCall DSV4ServerCall;

typedef struct {
    DSV4ServerCall *call;
    DSV4Utf8Stream utf8;
} DSV4ServerStream;

struct DSV4ServerCall {
    int client;
    int streaming;
    int include_usage;
    int output_started;
    int has_max_tokens;
    int has_seed;
    int has_temperature;
    int has_top_p;
    uint32_t max_tokens;
    uint64_t seed;
    double temperature;
    double top_p;
    long long created;
    char id[96];
    char *rendered_prompt;
    DSV4Buffer answer;
    DSV4ServerStream stream;
    DSV4Output output;
};

static int server_buffer_write(void *context, const char *data, size_t length)
{
    return dsv4_buffer_append((DSV4Buffer *)context, data, length);
}

static int server_socket_send_all(int client, const char *data, size_t length)
{
    while (length) {
        const ssize_t sent = send(client, data, length, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return 0;
        data += (size_t)sent;
        length -= (size_t)sent;
    }
    return 1;
}

static int server_response(int client, int status, const char *reason,
                           const char *content_type,
                           const char *body, size_t body_length)
{
    char header[1024];
    const int length = snprintf(
        header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n",
        status, reason, content_type, body_length);
    return length > 0 && (size_t)length < sizeof(header) &&
           server_socket_send_all(client, header, (size_t)length) &&
           server_socket_send_all(client, body ? body : "", body_length);
}

static int server_error_response(int client, int status,
                                 const char *reason, const char *message)
{
    DSV4Buffer body;
    dsv4_buffer_init(&body);
    int ok = dsv4_buffer_append_string(&body, "{\"error\":{\"message\":") &&
             dsv4_buffer_append_json_string(&body, message, strlen(message)) &&
             dsv4_buffer_append_string(
                 &body, ",\"type\":\"invalid_request_error\"}}");
    if (ok) ok = server_response(client, status, reason,
                                  "application/json; charset=utf-8",
                                  body.data, body.length);
    dsv4_buffer_free(&body);
    return ok;
}

static int server_parse_content_length(const char *text, size_t *length)
{
    if (!text || !*text || *text == '-') return 0;
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || value > DSV4_HTTP_MAX_BODY) return 0;
    *length = (size_t)value;
    return 1;
}

static int server_read_request(int client, DSV4IncomingRequest *request,
                               char *error, size_t error_capacity)
{
    enum { MAX_HEADER = 32 * 1024 };
    char header[MAX_HEADER + 1];
    size_t used = 0;
    char *separator = NULL;
    memset(request, 0, sizeof(*request));
    while (!separator) {
        if (used == MAX_HEADER) {
            snprintf(error, error_capacity, "request headers are too large");
            return 0;
        }
        const ssize_t received = recv(client, header + used,
                                      MAX_HEADER - used, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            snprintf(error, error_capacity,
                     "request ended before headers completed");
            return 0;
        }
        used += (size_t)received;
        header[used] = '\0';
        separator = strstr(header, "\r\n\r\n");
    }
    const size_t body_offset = (size_t)(separator - header) + 4u;
    *separator = '\0';
    char *line_end = strstr(header, "\r\n");
    if (!line_end) {
        snprintf(error, error_capacity, "invalid HTTP request line");
        return 0;
    }
    *line_end = '\0';
    char version[16];
    if (sscanf(header, "%7s %127s HTTP/%15s",
               request->method, request->path, version) != 3 ||
        (strcmp(version, "1.1") != 0 && strcmp(version, "1.0") != 0)) {
        snprintf(error, error_capacity, "invalid HTTP request line");
        return 0;
    }
    size_t content_length = 0;
    int saw_content_length = 0;
    char *line = line_end + 2;
    while (*line) {
        char *next = strstr(line, "\r\n");
        if (next) *next = '\0';
        if (strncasecmp(line, "Content-Length:", 15u) == 0) {
            const char *value = line + 15;
            while (*value == ' ' || *value == '\t') ++value;
            if (saw_content_length ||
                !server_parse_content_length(value, &content_length)) {
                snprintf(error, error_capacity, "invalid Content-Length");
                return 0;
            }
            saw_content_length = 1;
        } else if (strncasecmp(line, "Transfer-Encoding:", 18u) == 0) {
            snprintf(error, error_capacity,
                     "chunked requests are not supported");
            return 0;
        }
        if (!next) break;
        line = next + 2;
    }
    if (strcmp(request->method, "POST") == 0 && !saw_content_length) {
        snprintf(error, error_capacity, "POST requires Content-Length");
        return 0;
    }
    request->body = (char *)malloc(content_length + 1u);
    if (!request->body) {
        snprintf(error, error_capacity, "out of memory reading request");
        return 0;
    }
    size_t copied = used > body_offset ? used - body_offset : 0u;
    if (copied > content_length) copied = content_length;
    if (copied) memcpy(request->body, header + body_offset, copied);
    while (copied < content_length) {
        const ssize_t received = recv(client, request->body + copied,
                                      content_length - copied, 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) {
            free(request->body);
            request->body = NULL;
            snprintf(error, error_capacity, "request body ended early");
            return 0;
        }
        copied += (size_t)received;
    }
    request->body[content_length] = '\0';
    request->body_length = content_length;
    return 1;
}

static int server_open(uint32_t port)
{
    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
        perror("dsv4: socket");
        return -1;
    }
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    if (inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) != 1) {
        fprintf(stderr, "dsv4: unable to create loopback address\n");
        close(server);
        return -1;
    }
    address.sin_port = htons((uint16_t)port);
    if (bind(server, (const struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(server, 16) != 0) {
        perror("dsv4: bind/listen");
        close(server);
        return -1;
    }
    signal(SIGPIPE, SIG_IGN);
    return server;
}

static int server_buffer_printf(DSV4Buffer *buffer, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    const int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(args);
        return 0;
    }
    char *text = (char *)malloc((size_t)needed + 1u);
    if (!text) {
        va_end(args);
        return 0;
    }
    vsnprintf(text, (size_t)needed + 1u, format, args);
    va_end(args);
    const int ok = dsv4_buffer_append(buffer, text, (size_t)needed);
    free(text);
    return ok;
}

static int server_stream_headers(int client)
{
    static const char headers[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "X-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n";
    return server_socket_send_all(client, headers, sizeof(headers) - 1u);
}

static int server_stream_send(DSV4ServerCall *call, DSV4Buffer *event)
{
    const int ok = dsv4_buffer_append_string(event, "\n\n") &&
                   server_socket_send_all(call->client,
                                          event->data, event->length);
    dsv4_buffer_free(event);
    return ok;
}

static int server_stream_role(DSV4ServerCall *call)
{
    DSV4Buffer event;
    dsv4_buffer_init(&event);
    const int ok = server_buffer_printf(
        &event,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"deepseek-v4-flash-0731-in-c\","
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\","
        "\"content\":\"\"},\"finish_reason\":null}]%s}",
        call->id, call->created,
        call->include_usage ? ",\"usage\":null" : "");
    if (!ok) {
        dsv4_buffer_free(&event);
        return 0;
    }
    return server_stream_send(call, &event);
}

static int server_stream_content(void *context,
                                 const char *data, size_t length)
{
    DSV4ServerCall *call = (DSV4ServerCall *)context;
    DSV4Buffer event;
    dsv4_buffer_init(&event);
    int ok = server_buffer_printf(
        &event,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"deepseek-v4-flash-0731-in-c\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":",
        call->id, call->created);
    if (ok) ok = dsv4_buffer_append_json_string(&event, data, length);
    if (ok) ok = server_buffer_printf(
        &event, "},\"finish_reason\":null}]%s}",
        call->include_usage ? ",\"usage\":null" : "");
    if (!ok) {
        dsv4_buffer_free(&event);
        return 0;
    }
    return server_stream_send(call, &event);
}

static int server_call_begin(DSV4ServerCall *call)
{
    call->output_started = 1;
    call->output.bytes = 0;
    call->output.last_byte = 0;
    call->output.failed = 0;
    call->output.append_final_newline = 0;
    if (!call->streaming) {
        call->output.write = server_buffer_write;
        call->output.context = &call->answer;
        return 1;
    }
    call->stream.call = call;
    dsv4_utf8_stream_init(&call->stream.utf8,
                          server_stream_content, call);
    call->output.write = dsv4_utf8_stream_write;
    call->output.context = &call->stream.utf8;
    return server_stream_headers(call->client) && server_stream_role(call);
}

static int server_stream_finish(DSV4ServerCall *call, int prompt_tokens,
                                int generated_tokens, int truncated)
{
    if (!dsv4_utf8_stream_flush(&call->stream.utf8)) return 0;
    DSV4Buffer event;
    dsv4_buffer_init(&event);
    int ok = server_buffer_printf(
        &event,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"created\":%lld,\"model\":\"deepseek-v4-flash-0731-in-c\","
        "\"choices\":[{\"index\":0,\"delta\":{},"
        "\"finish_reason\":\"%s\"}]%s}",
        call->id, call->created, truncated ? "length" : "stop",
        call->include_usage ? ",\"usage\":null" : "");
    if (!ok) {
        dsv4_buffer_free(&event);
        return 0;
    }
    if (!server_stream_send(call, &event)) return 0;
    if (call->include_usage) {
        dsv4_buffer_init(&event);
        ok = server_buffer_printf(
            &event,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"created\":%lld,\"model\":\"deepseek-v4-flash-0731-in-c\","
            "\"choices\":[],\"usage\":{\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,\"total_tokens\":%d}}",
            call->id, call->created, prompt_tokens, generated_tokens,
            prompt_tokens + generated_tokens);
        if (!ok) {
            dsv4_buffer_free(&event);
            return 0;
        }
        if (!server_stream_send(call, &event)) return 0;
    }
    static const char done[] = "data: [DONE]\n\n";
    return server_socket_send_all(call->client, done, sizeof(done) - 1u);
}

static int server_call_finish(DSV4ServerCall *call, int prompt_tokens,
                              int generated_tokens, int truncated)
{
    int ok = 0;
    if (call->streaming) {
        ok = server_stream_finish(call, prompt_tokens,
                                  generated_tokens, truncated);
    } else {
        DSV4Buffer response;
        dsv4_buffer_init(&response);
        ok = server_buffer_printf(
            &response,
            "{\"id\":\"%s\",\"object\":\"chat.completion\","
            "\"created\":%lld,\"model\":\"deepseek-v4-flash-0731-in-c\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":",
            call->id, call->created);
        if (ok) ok = dsv4_buffer_append_json_string(
            &response, call->answer.data ? call->answer.data : "",
            call->answer.length);
        if (ok) ok = server_buffer_printf(
            &response,
            "},\"finish_reason\":\"%s\"}],\"usage\":{"
            "\"prompt_tokens\":%d,\"completion_tokens\":%d,"
            "\"total_tokens\":%d}}",
            truncated ? "length" : "stop", prompt_tokens,
            generated_tokens, prompt_tokens + generated_tokens);
        if (ok) ok = server_response(call->client, 200, "OK",
                                      "application/json; charset=utf-8",
                                      response.data, response.length);
        dsv4_buffer_free(&response);
    }
    return ok;
}

static void server_call_dispose(DSV4ServerCall *call)
{
    if (!call) return;
    if (call->streaming && call->output_started)
        dsv4_utf8_stream_free(&call->stream.utf8);
    if (call->client >= 0) close(call->client);
    free(call->rendered_prompt);
    dsv4_buffer_free(&call->answer);
    memset(call, 0, sizeof(*call));
    call->client = -1;
}

static int server_next_call(int server, const char *base_system,
                            const char *reasoning_effort, const char *eos_text,
                            DSV4ServerCall *call)
{
    static unsigned long long serial = 0;
    for (;;) {
        memset(call, 0, sizeof(*call));
        call->client = -1;
        const int client = accept(server, NULL, NULL);
        if (client < 0 && errno == EINTR) continue;
        if (client < 0) {
            perror("dsv4: accept");
            return 0;
        }
        call->client = client;
        struct timeval timeout = {30, 0};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        DSV4IncomingRequest incoming;
        char error[256] = {0};
        if (!server_read_request(client, &incoming, error, sizeof(error))) {
            (void)server_error_response(client, 400, "Bad Request", error);
            server_call_dispose(call);
            continue;
        }
        int handled = 1;
        if (strcmp(incoming.method, "OPTIONS") == 0) {
            (void)server_response(client, 204, "No Content",
                                  "application/json; charset=utf-8", "", 0u);
        } else if (strcmp(incoming.method, "GET") == 0 &&
                   strcmp(incoming.path, "/health") == 0) {
            static const char body[] = "{\"status\":\"ok\"}";
            (void)server_response(client, 200, "OK",
                                  "application/json; charset=utf-8",
                                  body, sizeof(body) - 1u);
        } else if (strcmp(incoming.method, "GET") == 0 &&
                   strcmp(incoming.path, "/v1/models") == 0) {
            static const char body[] =
                "{\"object\":\"list\",\"data\":[{"
                "\"id\":\"deepseek-v4-flash-0731-in-c\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}";
            (void)server_response(client, 200, "OK",
                                  "application/json; charset=utf-8",
                                  body, sizeof(body) - 1u);
        } else if (strcmp(incoming.method, "POST") != 0 ||
                   strcmp(incoming.path, "/v1/chat/completions") != 0) {
            (void)server_error_response(client, 404, "Not Found",
                                        "unknown endpoint");
        } else {
            handled = 0;
        }
        if (handled) {
            free(incoming.body);
            server_call_dispose(call);
            continue;
        }
        DSV4HttpChatRequest request;
        if (!dsv4_http_parse_chat_request(incoming.body, incoming.body_length,
                                          &request, error, sizeof(error))) {
            free(incoming.body);
            (void)server_error_response(client, 400, "Bad Request", error);
            server_call_dispose(call);
            continue;
        }
        free(incoming.body);
        if (strcmp(request.model, "deepseek-v4-flash-0731-in-c") != 0 &&
            strcmp(request.model, "deepseek-v4-flash-0731") != 0) {
            dsv4_http_chat_request_free(&request);
            (void)server_error_response(client, 404, "Not Found",
                                        "requested model is not available");
            server_call_dispose(call);
            continue;
        }
        if (request.has_presence_penalty &&
            request.presence_penalty != 0.0f) {
            dsv4_http_chat_request_free(&request);
            (void)server_error_response(
                client, 400, "Bad Request",
                "presence_penalty is not supported by this sampler");
            server_call_dispose(call);
            continue;
        }
        call->rendered_prompt = dsv4_http_render_messages(
            &request, base_system, reasoning_effort, eos_text,
            error, sizeof(error));
        if (!call->rendered_prompt) {
            dsv4_http_chat_request_free(&request);
            (void)server_error_response(client, 400, "Bad Request", error);
            server_call_dispose(call);
            continue;
        }
        call->streaming = request.stream;
        call->include_usage = request.include_usage;
        call->has_max_tokens = request.has_max_tokens;
        call->max_tokens = request.max_tokens;
        call->has_seed = request.has_seed;
        call->seed = request.seed;
        call->has_temperature = request.has_temperature;
        call->temperature = request.temperature;
        call->has_top_p = request.has_top_p;
        call->top_p = request.top_p;
        dsv4_http_chat_request_free(&request);
        call->created = (long long)time(NULL);
        const int length = snprintf(
            call->id, sizeof(call->id), "chatcmpl-local-%lld-%llu",
            call->created, ++serial);
        if (length <= 0 || (size_t)length >= sizeof(call->id)) {
            (void)server_error_response(client, 500, "Internal Server Error",
                                        "unable to create completion id");
            server_call_dispose(call);
            continue;
        }
        dsv4_buffer_init(&call->answer);
        return 1;
    }
}

static int server_call_tokenize(Tok *tok, DSV4ServerCall *call, int context,
                                int **ids_out, int *count_out,
                                char *error, size_t error_capacity)
{
    const size_t length = strlen(call->rendered_prompt);
    if (length > (size_t)INT_MAX - 1u) {
        snprintf(error, error_capacity, "rendered prompt is too large");
        return 0;
    }
    int *ids = (int *)malloc((length + 1u) * sizeof(int));
    if (!ids) {
        snprintf(error, error_capacity, "out of memory tokenizing messages");
        return 0;
    }
    const int count = tok_encode(tok, call->rendered_prompt, (int)length,
                                 ids, (int)length + 1);
    if (count <= 0) {
        free(ids);
        snprintf(error, error_capacity, "messages produced no tokens");
        return 0;
    }
    if (count >= context) {
        free(ids);
        snprintf(error, error_capacity,
                 "messages need %d tokens but context is %d", count, context);
        return 0;
    }
    *ids_out = ids;
    *count_out = count;
    return 1;
}

static void server_apply_call(const DSV4ServerCall *call,
                              int default_max_tokens,
                              double default_temperature,
                              double default_top_p, uint64_t default_seed,
                              int *max_tokens, int *max_tokens_set,
                              double *temperature, double *top_p)
{
    *max_tokens = call->has_max_tokens
                ? (int)call->max_tokens : default_max_tokens;
    *max_tokens_set = 1;
    *temperature = call->has_temperature
                 ? call->temperature : default_temperature;
    *top_p = call->has_top_p ? call->top_p : default_top_p;
    const uint64_t request_seed = call->has_seed ? call->seed : default_seed;
    rng_state = request_seed ? request_seed : 0x9E3779B97F4A7C15ull;
}

int main(int argc, char **argv)
{
    const char *model_dir = NULL, *prompt = NULL, *system = NULL, *effort = NULL;
    int max_tokens = 0, max_tokens_set = 0, context = 0, context_set = 0;
    int validate_only = 0;
    int thinking = 0, raw_prompt = 0, interactive = 0, threads = 0;
    uint32_t server_port = 0;
    int no_prompt_lookup = 0;
    int memory_gib_set = 0, cache_gib_set = 0;
    double memory_gib = 0.0, cache_gib = 0.0, temperature = 1.0, top_p = 1.0;
    uint64_t seed = 0;
    int seed_set = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        long lv; double dv;
        if (!strcmp(a, "--model")) { if (++i >= argc) goto needarg; model_dir = argv[i]; }
        else if (!strcmp(a, "--prompt")) { if (++i >= argc) goto needarg; prompt = argv[i]; }
        else if (!strcmp(a, "--system")) { if (++i >= argc) goto needarg; system = argv[i]; }
        else if (!strcmp(a, "--reasoning-effort")) { if (++i >= argc) goto needarg; effort = argv[i]; }
        else if (!strcmp(a, "--max-tokens")) { if (++i >= argc || !parse_int(argv[i], &lv)) goto badnum; max_tokens = (int)lv; max_tokens_set = 1; }
        else if (!strcmp(a, "--context")) { if (++i >= argc || !parse_int(argv[i], &lv)) goto badnum; context = (int)lv; context_set = 1; }
        else if (!strcmp(a, "--memory-gib")) { if (++i >= argc || !parse_float(argv[i], &dv)) goto badnum; memory_gib = dv; memory_gib_set = 1; }
        else if (!strcmp(a, "--cache-gib")) { if (++i >= argc || !parse_float(argv[i], &dv)) goto badnum; cache_gib = dv; cache_gib_set = 1; }
        else if (!strcmp(a, "--threads")) { if (++i >= argc || !parse_int(argv[i], &lv)) goto badnum; threads = (int)lv; }
        else if (!strcmp(a, "--temperature")) { if (++i >= argc || !parse_float(argv[i], &dv)) goto badnum; temperature = dv; }
        else if (!strcmp(a, "--top-p")) { if (++i >= argc || !parse_float(argv[i], &dv)) goto badnum; top_p = dv; }
        else if (!strcmp(a, "--seed")) { if (++i >= argc || !parse_int(argv[i], &lv)) goto badnum; seed = (uint64_t)lv; seed_set = 1; }
        else if (!strcmp(a, "--validate-only")) validate_only = 1;
        else if (!strcmp(a, "--thinking")) thinking = 1;
        else if (!strcmp(a, "--raw-prompt")) raw_prompt = 1;
        else if (!strcmp(a, "--no-prompt-lookup")) no_prompt_lookup = 1;
        else if (!strcmp(a, "--interactive")) interactive = 1;
        else if (!strcmp(a, "--server")) {
            if (++i >= argc || !parse_int(argv[i], &lv) ||
                lv < 1 || lv > 65535) goto badnum;
            server_port = (uint32_t)lv;
        }
        else if (!strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "%s: unknown option %s\n", argv[0], a); usage(argv[0]); return 2; }
        continue;
    needarg:
        fprintf(stderr, "%s: option %s needs an argument\n", argv[0], a);
        return 2;
    badnum:
        fprintf(stderr, "%s: bad number for %s\n", argv[0], a);
        return 2;
    }

    if (cache_gib_set && memory_gib_set) {
        fprintf(stderr, "dsv4: --memory-gib and --cache-gib are mutually exclusive\n");
        return 2;
    }
    if (cache_gib_set && (cache_gib <= 0 || cache_gib > 1024.0)) {
        fprintf(stderr, "dsv4: implausible --cache-gib %.1f\n", cache_gib);
        return 2;
    }
    if (!isfinite(temperature) || temperature < 0.0) {
        fprintf(stderr, "dsv4: --temperature must be finite and non-negative\n");
        return 2;
    }
    if (!isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) {
        fprintf(stderr, "dsv4: --top-p must be in (0, 1]\n");
        return 2;
    }
    if (interactive && raw_prompt) {
        fprintf(stderr, "dsv4: --interactive requires the chat template; remove --raw-prompt\n");
        return 2;
    }
    if (server_port && (interactive || prompt || raw_prompt || validate_only)) {
        fprintf(stderr,
                "dsv4: --server cannot be combined with --prompt, "
                "--interactive, --raw-prompt or --validate-only\n");
        return 2;
    }
    if (max_tokens_set && max_tokens < 1) {
        fprintf(stderr, "dsv4: --max-tokens must be at least 1\n");
        return 2;
    }
    if (effort) {
        if (strcmp(effort, "low") && strcmp(effort, "high") && strcmp(effort, "max")) {
            fprintf(stderr, "dsv4: --reasoning-effort must be low, high or max\n");
            return 2;
        }
        thinking = 1;
    }
    if (!seed_set) {
        struct timespec realtime;
        clock_gettime(CLOCK_REALTIME, &realtime);
        seed = ((uint64_t)realtime.tv_sec << 32) ^ (uint64_t)realtime.tv_nsec
             ^ (uint64_t)getpid();
    }
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ull;
    if (server_port && !max_tokens_set) {
        max_tokens = 1024;
        max_tokens_set = 1;
    }
    const int server_default_max_tokens = max_tokens;
    const double server_default_temperature = temperature;
    const double server_default_top_p = top_p;
    const uint64_t server_default_seed = seed;
    if (!model_dir) { fprintf(stderr, "dsv4: --model DIR is required\n"); usage(argv[0]); return 2; }

#ifdef _OPENMP
    /* OpenMP runtimes read their wait policy during process startup. Keep the
     * long expert-I/O waits passive, but let libgomp spin briefly across the
     * much shorter gaps between adjacent kernels. Re-exec only the CLI;
     * embedders retain full control, and explicit user settings always win. */
    if (!validate_only && !getenv("OMP_WAIT_POLICY")) {
        if (setenv("OMP_WAIT_POLICY", "PASSIVE", 1) != 0) {
            perror("dsv4: cannot set OMP_WAIT_POLICY");
            return 1;
        }
        if (!getenv("GOMP_SPINCOUNT") &&
            setenv("GOMP_SPINCOUNT", "10000", 1) != 0) {
            perror("dsv4: cannot set GOMP_SPINCOUNT");
            return 1;
        }
        execvp(argv[0], argv);
        perror("dsv4: cannot restart with bounded OpenMP waiting");
        return 1;
    }
#endif

    struct sigaction sigint_action;
    memset(&sigint_action, 0, sizeof sigint_action);
    sigint_action.sa_handler = handle_sigint;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = SA_RESTART;
    if (!server_port && sigaction(SIGINT, &sigint_action, NULL) != 0) {
        perror("dsv4: cannot install SIGINT handler");
        return 1;
    }

    if (!cache_gib_set && !memory_gib_set) {
        double visible_gib, available_gib;
        memory_gib = automatic_memory_gib(&visible_gib, &available_gib);
        if (visible_gib > 0.0 && available_gib > 0.0)
            fprintf(stderr, "dsv4: automatic memory budget %.0f GiB from %.2f GiB "
                    "visible / %.2f GiB currently available\n",
                    memory_gib, visible_gib, available_gib);
        else if (visible_gib > 0.0)
            fprintf(stderr, "dsv4: automatic memory budget %.0f GiB from %.2f GiB "
                    "visible RAM\n", memory_gib, visible_gib);
        else
            fprintf(stderr, "dsv4: could not detect RAM; using a conservative 2 GiB budget\n");
    }
    if (!cache_gib_set && (memory_gib <= 0 || memory_gib > 1024.0)) {
        fprintf(stderr, "dsv4: implausible --memory-gib %.1f\n", memory_gib);
        return 2;
    }

    /* ---- config + memory plan ---- */
    DSV4Config cfg;
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", model_dir);
    if (!dsv4_config_load_file(&cfg, path)) return 1;

    const uint64_t GIB = 1024ull * 1024 * 1024;
    if (!context_set) {
        if (cache_gib_set) {
            context = cfg.original_position < 4096 ? cfg.original_position : 4096;
            fprintf(stderr, "dsv4: advanced --cache-gib mode uses context %d; "
                    "pass --context to change it\n", context);
        } else {
            uint64_t auto_budget = (uint64_t)(memory_gib * (double)GIB);
            context = dsv4_auto_context(&cfg, auto_budget);
            if (context < 1) {
                fprintf(stderr, "dsv4: memory budget cannot hold a usable context\n");
                return 2;
            }
            fprintf(stderr, "dsv4: automatic context %d tokens "
                    "(checkpoint native limit %d)\n",
                    context, cfg.original_position);
        }
    }

    if (max_tokens < 0 || max_tokens > 1048576 ||
        context < 1 || context > cfg.max_position) {
        fprintf(stderr, "dsv4: implausible context/max-tokens\n");
        return 2;
    }

    DSV4MemoryPlan plan;
    uint64_t budget;
    if (cache_gib_set) {
        /* --cache-gib N sizes the expert LRU; runtime, context and all decoded
         * wo_a layers are additional. --memory-gib is the total-budget path. */
        uint64_t wo_a_values = (uint64_t)cfg.o_lora * cfg.n_heads *
                               cfg.head_dim;
        uint64_t wo_a_layer = wo_a_values * 2u;
        if (getenv("DSV4_PACKED_WO_A")) {
            uint64_t rows = (uint64_t)cfg.o_groups * cfg.o_lora;
            uint64_t cols = (uint64_t)cfg.n_heads * cfg.head_dim /
                            cfg.o_groups;
            wo_a_layer = wo_a_values + (rows + 127) / 128 *
                         ((cols + 127) / 128);
        }
        uint64_t wo_a_all = (uint64_t)cfg.n_layers * wo_a_layer;
        uint64_t dspark_persistent = 0;
        if (getenv("DSV4_EXPERIMENTAL_DSPARK"))
            dspark_persistent = (uint64_t)cfg.dspark_stages * wo_a_layer
                + (uint64_t)cfg.dspark_stages * (uint64_t)cfg.window *
                  (uint64_t)cfg.head_dim * sizeof(float);
        budget = 1258291200ull + dsv4_context_bytes(&cfg, context)
               + wo_a_all + dspark_persistent
               + (uint64_t)(cache_gib * (double)GIB);
    } else {
        budget = (uint64_t)(memory_gib * (double)GIB);
    }
    if (!dsv4_memory_plan(&cfg, context, budget, &plan)) return 1;
    if (validate_only) {
        DSV4Options validate_opt;
        memset(&validate_opt, 0, sizeof validate_opt);
        validate_opt.max_context = context;
        validate_opt.threads = threads;
        validate_opt.expert_cache_bytes = plan.expert_cache_bytes;
        validate_opt.wo_a_cache_layers = plan.wo_a_cache_layers;
        DSV4Model *validate_model = dsv4_model_open(model_dir, &validate_opt);
        if (!validate_model) return 1;
        dsv4_model_close(validate_model);
        fprintf(stderr, "dsv4: metadata OK (config, 48 shard headers, tensor bindings, "
                "memory plan)\n");
        return 0;
    }

    /* ---- tokenizer ---- */
    Tok tok;
    snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
    tok_load(&tok, path);

    char *input_line = NULL;
    size_t input_cap = 0;
    const char *eff = thinking ? (effort ? effort : "high") : NULL;
    DSV4Options opt;
    memset(&opt, 0, sizeof opt);
    opt.max_context = context;
    opt.threads = threads;
    opt.expert_cache_bytes = plan.expert_cache_bytes;
    opt.wo_a_cache_layers = plan.wo_a_cache_layers;
    DSV4Model *m = NULL;
    int server_fd = -1;
    DSV4ServerCall server_call;
    memset(&server_call, 0, sizeof(server_call));
    server_call.client = -1;
    if (server_port) {
        m = dsv4_model_open(model_dir, &opt);
        if (!m) return 1;
        server_fd = server_open(server_port);
        if (server_fd < 0) return 1;
        fprintf(stderr,
                "dsv4: OpenAI-compatible server listening on "
                "http://127.0.0.1:%u\n", server_port);
        fprintf(stderr,
                "dsv4: model id deepseek-v4-flash-0731-in-c; "
                "one request at a time\n");
    }

    char *ptext = NULL;
    int ptext_len = 0;
    int *ids = NULL;
    int ntoks = 0;
    if (server_port) {
        char request_error[256];
        for (;;) {
            if (!server_next_call(server_fd, system, eff, tok.id2str[1],
                                  &server_call)) return 1;
            if (server_call_tokenize(&tok, &server_call, context,
                                     &ids, &ntoks,
                                     request_error, sizeof(request_error)))
                break;
            (void)server_error_response(server_call.client, 400, "Bad Request",
                                        request_error);
            server_call_dispose(&server_call);
        }
        server_apply_call(&server_call, server_default_max_tokens,
                          server_default_temperature, server_default_top_p,
                          server_default_seed, &max_tokens, &max_tokens_set,
                          &temperature, &top_p);
    } else {
        if (!prompt && interactive) {
            if (!read_chat_line(&input_line, &input_cap)) return 0;
            prompt = input_line;
        }
        if (!prompt || !prompt[0]) {
            fprintf(stderr,
                    "dsv4: --prompt TEXT is required unless --interactive reads it\n");
            return 2;
        }
        if (raw_prompt) {
            ptext_len = (int)strlen(prompt);
            ptext = (char *)malloc((size_t)ptext_len + 1);
            if (!ptext) { fprintf(stderr, "dsv4: OOM prompt buffer\n"); return 1; }
            memcpy(ptext, prompt, (size_t)ptext_len + 1);
        } else {
            int need = dsv4_format_prompt(NULL, 0, prompt, system, eff);
            if (need < 0) { fprintf(stderr, "dsv4: empty prompt\n"); return 2; }
            ptext = (char *)malloc((size_t)need + 1);
            if (!ptext) { fprintf(stderr, "dsv4: OOM prompt buffer\n"); return 1; }
            dsv4_format_prompt(ptext, (size_t)need + 1, prompt, system, eff);
            ptext_len = need;
        }
        ids = (int *)malloc((size_t)(ptext_len + 1) * sizeof(int));
        if (!ids) { fprintf(stderr, "dsv4: OOM token buffer\n"); return 1; }
        ntoks = tok_encode(&tok, ptext, ptext_len, ids, ptext_len + 1);
        if (ntoks <= 0) {
            fprintf(stderr, "dsv4: prompt produced no tokens\n");
            return 2;
        }
        if (ntoks >= context) {
            fprintf(stderr,
                    "dsv4: prompt (%d tokens) leaves no generation room in context %d\n",
                    ntoks, context);
            return 2;
        }
    }
    fprintf(stderr, "prompt: %d tokens\n", ntoks);
    if (!interactive && !server_port) {
        fprintf(stderr, "prompt ids:");
        for (int i = 0; i < ntoks; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
    }

    /* ---- open model ---- */
    if (!m) {
        m = dsv4_model_open(model_dir, &opt);
        if (!m) return 1;
    }

    /* Keep the speculative path opt-in until its acceptance rate and batched
     * verifier are fast enough to improve end-user latency on the full model. */
    int use_dspark = !server_port && temperature == 0.0 && dsv4_dspark_ready(m) &&
                     getenv("DSV4_EXPERIMENTAL_DSPARK") != NULL;
    int use_ngram = !no_prompt_lookup;
    if (!no_prompt_lookup && getenv("DSV4_EXPERIMENTAL_NGRAM") != NULL)
        use_ngram = 1;
    const int ngram_allow_overlap = use_ngram &&
        (getenv("DSV4_NGRAM_ALLOW_OVERLAP") != NULL ||
         getenv("DSV4_NGRAM_STRICT") == NULL);
    const int ngram_periodic_extend = ngram_allow_overlap &&
        (getenv("DSV4_NGRAM_PERIODIC_EXTEND") != NULL ||
         getenv("DSV4_NGRAM_NO_PERIODIC_EXTEND") == NULL);
    if (use_ngram && use_dspark) {
        use_ngram = 0;
        fprintf(stderr, "dsv4: n-gram and DSpark speculation cannot be combined; "
                        "using DSpark\n");
    }
    int ngram_block = DSV4_NGRAM_DEFAULT;
    const char *ngram_block_env = getenv("DSV4_NGRAM_DRAFTS");
    if (use_ngram && ngram_block_env) {
        char *end = NULL;
        long requested = strtol(ngram_block_env, &end, 10);
        if (end != ngram_block_env && *end == '\0' && requested >= 1 &&
            requested <= DSV4_NGRAM_MAX) {
            ngram_block = (int)requested;
        } else {
            fprintf(stderr, "dsv4: ignoring invalid DSV4_NGRAM_DRAFTS=%s "
                            "(expected 1..%d)\n",
                    ngram_block_env, DSV4_NGRAM_MAX);
        }
    }
    int ngram_min_match = 3;
    const char *ngram_match_env = getenv("DSV4_NGRAM_MIN_MATCH");
    if (use_ngram && ngram_match_env) {
        char *end = NULL;
        long requested = strtol(ngram_match_env, &end, 10);
        if (end != ngram_match_env && *end == '\0' && requested >= 1 &&
            requested <= 4) {
            ngram_min_match = (int)requested;
        } else {
            fprintf(stderr, "dsv4: ignoring invalid DSV4_NGRAM_MIN_MATCH=%s "
                            "(expected 1..4)\n", ngram_match_env);
        }
    }
    int ngram_reject_cooldown = 8;
    const char *ngram_cooldown_env = getenv("DSV4_NGRAM_REJECT_COOLDOWN");
    if (use_ngram && ngram_cooldown_env) {
        char *end = NULL;
        long requested = strtol(ngram_cooldown_env, &end, 10);
        if (end != ngram_cooldown_env && *end == '\0' && requested >= 0 &&
            requested <= INT_MAX) {
            ngram_reject_cooldown = (int)requested;
        } else {
            fprintf(stderr,
                    "dsv4: ignoring invalid DSV4_NGRAM_REJECT_COOLDOWN=%s "
                    "(expected 0..%d)\n", ngram_cooldown_env, INT_MAX);
        }
    }
    int spec_block = cfg.dspark_block_size;
    const char *spec_block_env = getenv("DSV4_DSPARK_VERIFY_DRAFTS");
    if (use_dspark && spec_block_env) {
        char *end = NULL;
        long requested = strtol(spec_block_env, &end, 10);
        if (end != spec_block_env && *end == '\0' && requested >= 1 &&
            requested <= cfg.dspark_block_size) {
            spec_block = (int)requested;
        } else {
            fprintf(stderr,
                    "dsv4: ignoring invalid DSV4_DSPARK_VERIFY_DRAFTS=%s "
                    "(expected 1..%d)\n",
                    spec_block_env, cfg.dspark_block_size);
        }
    }
    int spec_reject_cooldown = 0;
    const char *spec_cooldown_env = getenv("DSV4_DSPARK_REJECT_COOLDOWN");
    if (use_dspark && spec_cooldown_env) {
        char *end = NULL;
        long requested = strtol(spec_cooldown_env, &end, 10);
        if (end != spec_cooldown_env && *end == '\0' && requested >= 0 &&
            requested <= INT_MAX) {
            spec_reject_cooldown = (int)requested;
        } else {
            fprintf(stderr,
                    "dsv4: ignoring invalid DSV4_DSPARK_REJECT_COOLDOWN=%s "
                    "(expected 0..%d)\n",
                    spec_cooldown_env, INT_MAX);
        }
    }
    int spec_verify_hash_slots = cfg.topk;
    const char *spec_hash_slots_env =
        getenv("DSV4_DSPARK_VERIFY_HASH_SLOTS");
    if (use_dspark && spec_hash_slots_env) {
        char *end = NULL;
        long requested = strtol(spec_hash_slots_env, &end, 10);
        if (end != spec_hash_slots_env && *end == '\0' &&
            requested >= cfg.topk && requested <= cfg.n_experts) {
            spec_verify_hash_slots = (int)requested;
        } else {
            fprintf(stderr,
                    "dsv4: ignoring invalid DSV4_DSPARK_VERIFY_HASH_SLOTS=%s "
                    "(expected %d..%d)\n",
                    spec_hash_slots_env, cfg.topk, cfg.n_experts);
        }
    }
    const int verify_tokens = cfg.dspark_block_size + 1;
    DSV4ContextSnapshot *spec_snapshot = NULL;
    float *spec_logits = NULL;
    float *spec_capture = NULL;
    if (use_dspark) {
        const size_t capture_one = (size_t)cfg.dspark_stages * cfg.hidden;
        spec_snapshot = dsv4_context_snapshot_create(m, verify_tokens);
        spec_logits = (float *)malloc((size_t)verify_tokens * cfg.vocab *
                                      sizeof(float));
        spec_capture = (float *)malloc((size_t)verify_tokens * capture_one *
                                       sizeof(float));
        if (!spec_snapshot || !spec_logits || !spec_capture) {
            dsv4_context_snapshot_free(spec_snapshot);
            free(spec_logits);
            free(spec_capture);
            spec_snapshot = NULL;
            spec_logits = NULL;
            spec_capture = NULL;
            use_dspark = 0;
            fprintf(stderr, "dsv4: DSpark workspace unavailable; using exact "
                    "single-token decode\n");
        }
    }
    long spec_rounds = 0, spec_accepted = 0, spec_outputs = 0;
    long spec_cooldown_events = 0;
    long spec_accept_hist[33] = {0};
    DSV4PhaseStats spec_draft_stats = {0};
    DSV4PhaseStats spec_verify_stats = {0};
    DSV4PhaseStats spec_replay_stats = {0};
    DSV4PhaseStats spec_fallback_stats = {0};
    DSV4ContextSnapshot *ngram_snapshot = NULL;
    float *ngram_logits = NULL;
    long ngram_rounds = 0, ngram_drafted = 0, ngram_accepted = 0;
    long ngram_outputs = 0, ngram_cooldown_events = 0;
    long ngram_accept_hist[DSV4_NGRAM_MAX + 1] = {0};
    DSV4PhaseStats ngram_verify_stats = {0};
    if (use_ngram) {
        ngram_snapshot = dsv4_context_snapshot_create(m, ngram_block + 1);
        ngram_logits = (float *)malloc((size_t)(ngram_block + 1) * cfg.vocab *
                                       sizeof(float));
        if (!ngram_snapshot || !ngram_logits) {
            dsv4_context_snapshot_free(ngram_snapshot);
            free(ngram_logits);
            ngram_snapshot = NULL;
            ngram_logits = NULL;
            use_ngram = 0;
            fprintf(stderr, "dsv4: n-gram verifier workspace unavailable; "
                            "using exact single-token decode\n");
        }
    }

    /* ---- prefill (layer-major; each layer's weights are streamed once) ---- */
    float *logits = (float *)malloc((size_t)cfg.vocab * sizeof(float));
    if (!logits) { fprintf(stderr, "dsv4: OOM logits\n"); return 1; }
    SampleItem *sample_items = NULL;
    if (temperature > 0.0 || server_port) {
        sample_items = (SampleItem *)malloc((size_t)cfg.vocab * sizeof(*sample_items));
        if (!sample_items) { fprintf(stderr, "dsv4: OOM sampling buffer\n"); return 1; }
    }
    int *generated = (int *)malloc((size_t)context * sizeof(int));
    if (!generated) { fprintf(stderr, "dsv4: OOM token buffer\n"); return 1; }
    int *history = (int *)malloc((size_t)(context + 1) * sizeof(int));
    if (!history) { fprintf(stderr, "dsv4: OOM history buffer\n"); return 1; }
    memcpy(history, ids, (size_t)ntoks * sizeof(int));
    int history_len = ntoks;
    size_t piece_cap = 1;
    for (int id = 0; id < tok.n_ids; id++) {
        if (tok.id2str[id]) {
            size_t need = strlen(tok.id2str[id]) + 1;
            if (need > piece_cap) piece_cap = need;
        }
    }
    char *piece = (char *)malloc(piece_cap);
    if (!piece) { fprintf(stderr, "dsv4: OOM decode buffer\n"); return 1; }
    int committed = ntoks;
    int prefill_tokens = ntoks;
    stop_generation = 0;
    double inference_start = now_s();
    DSV4Output output = {output_file_write, stdout, 0, 0, 0, 1};
    if (server_port) {
        if (!server_call_begin(&server_call)) return 1;
        output = server_call.output;
    }
    if (interactive && isatty(STDIN_FILENO)) {
        fputs("assistant> ", stdout);
        fflush(stdout);
    }
    if ((use_dspark
             ? prefill_with_dspark_state(m, &cfg, ids, ntoks, 0, logits)
             : dsv4_prefill(m, ids, ntoks, 0, logits)) != 0) return 1;
    free(ptext);
    ptext = NULL;
    free(ids);
    ids = NULL;

    for (;;) {
        int gen = 0;
        int pending = -1;
        int spec_cooldown = 0;
        int spec_cache_expanded = 0;
        int ngram_cooldown = 0;
        int ngram_cache_expanded = 0;
        output.bytes = 0;
        output.last_byte = 0;
        output.failed = 0;
        double first_token_at = 0.0, last_token_at = 0.0;
        int response_limit = context - committed;
        if (max_tokens_set && max_tokens < response_limit) response_limit = max_tokens;
        if (response_limit > 0) {
            int nid = temperature == 0.0
                    ? argmax_logits(logits, cfg.vocab)
                    : sample_top_p(logits, cfg.vocab, temperature, top_p,
                                   sample_items);
            generated[gen++] = nid;
            pending = nid;
            history[history_len++] = nid;
            emit_token(&tok, nid, piece, piece_cap, &output);
            last_token_at = now_s();
            first_token_at = last_token_at;
        }

        while (gen < response_limit && pending != 1 && !stop_generation) {
            int handled = 0;
            const int block = spec_block;
            if (history_len != committed + 1) {
                fprintf(stderr, "dsv4: internal token history is out of sync\n");
                return 1;
            }
            if (use_ngram && ngram_cooldown == 0 &&
                response_limit - gen >= 2 &&
                committed < context - 1) {
                int proposal[DSV4_NGRAM_MAX];
                int verify_ids[DSV4_NGRAM_MAX + 1];
                int cap = response_limit - gen - 1;
                if (cap > context - committed - 1)
                    cap = context - committed - 1;
                if (cap > ngram_block) cap = ngram_block;
                int proposed = ngram_draft(history, history_len, cap,
                                            ngram_min_match,
                                            ngram_allow_overlap,
                                            ngram_periodic_extend, proposal);
                if (proposed > 0 && !ngram_cache_expanded &&
                    dsv4_expert_cache_set_hash_slots(m, 3 * cfg.topk) == 0)
                    ngram_cache_expanded = 1;
                if (proposed > 0 &&
                    dsv4_context_snapshot_take(ngram_snapshot, committed,
                                                proposed + 1) == 0) {
                    verify_ids[0] = pending;
                    for (int i = 0; i < proposed; i++)
                        verify_ids[i + 1] = proposal[i];
                    if (dspark_prefill_capture_profiled(
                            m, verify_ids, proposed + 1, committed, NULL,
                            ngram_logits, NULL, 0, NULL,
                            &ngram_verify_stats) != 0)
                        return 1;

                    int accepted = 0;
                    int correction = -1;
                    while (accepted < proposed) {
                        const float *target_logits = ngram_logits +
                            (size_t)accepted * cfg.vocab;
                        const int target = temperature == 0.0
                            ? argmax_logits(target_logits, cfg.vocab)
                            : sample_top_p(target_logits, cfg.vocab,
                                           temperature, top_p, sample_items);
                        if (proposal[accepted] != target) {
                            correction = target;
                            break;
                        }
                        accepted++;
                        if (target == 1) break;
                    }
                    ngram_rounds++;
                    ngram_drafted += proposed;
                    ngram_accepted += accepted;
                    ngram_accept_hist[accepted]++;
                    if (accepted == 0 && ngram_reject_cooldown > 0) {
                        ngram_cooldown = ngram_reject_cooldown;
                        ngram_cooldown_events++;
                    }

                    int eos_at = -1;
                    for (int i = 0; i < accepted; i++) {
                        if (proposal[i] == 1) {
                            eos_at = i;
                            break;
                        }
                    }
                    if (eos_at >= 0) {
                        const int replay = eos_at + 1;
                        if (dsv4_context_snapshot_commit_prefix(
                                ngram_snapshot, replay) != 0)
                            return 1;
                        committed += replay;
                        for (int i = 0; i <= eos_at; i++) {
                            const int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            ngram_outputs++;
                        }
                        handled = 1;
                    } else if (accepted < proposed) {
                        const int replay = accepted + 1;
                        if (dsv4_context_snapshot_commit_prefix(
                                ngram_snapshot, replay) != 0)
                            return 1;
                        committed += replay;
                        for (int i = 0; i < accepted; i++) {
                            const int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            ngram_outputs++;
                        }
                        generated[gen++] = correction;
                        pending = correction;
                        history[history_len++] = correction;
                        emit_token(&tok, correction, piece, piece_cap, &output);
                        last_token_at = now_s();
                        ngram_outputs++;
                        handled = 1;
                    } else {
                        const float *bonus_logits = ngram_logits +
                            (size_t)proposed * cfg.vocab;
                        const int bonus = temperature == 0.0
                            ? argmax_logits(bonus_logits, cfg.vocab)
                            : sample_top_p(bonus_logits, cfg.vocab,
                                           temperature, top_p, sample_items);
                        if (dsv4_context_snapshot_commit_prefix(
                                ngram_snapshot, proposed + 1) != 0)
                            return 1;
                        committed += proposed + 1;
                        for (int i = 0; i < proposed; i++) {
                            const int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            ngram_outputs++;
                        }
                        generated[gen++] = bonus;
                        pending = bonus;
                        history[history_len++] = bonus;
                        emit_token(&tok, bonus, piece, piece_cap, &output);
                        last_token_at = now_s();
                        ngram_outputs++;
                        handled = 1;
                    }
                }
            }
            if (!handled && use_dspark && spec_cooldown == 0 &&
                response_limit - gen >= block + 1 &&
                committed <= context - (block + 1)) {
                int proposal[32];
                int verify_ids[33];
                int proposed = dspark_propose_profiled(
                    m, pending, committed, block, proposal, &spec_draft_stats);
                if (proposed >= block && !spec_cache_expanded &&
                    spec_verify_hash_slots > cfg.topk &&
                    dsv4_expert_cache_set_hash_slots(
                        m, spec_verify_hash_slots) == 0)
                    spec_cache_expanded = 1;
                if (proposed >= block &&
                    dsv4_context_snapshot_take(spec_snapshot, committed,
                                                block + 1) == 0) {
                    verify_ids[0] = pending;
                    for (int i = 0; i < block; i++)
                        verify_ids[i + 1] = proposal[i];
                    if (dspark_prefill_capture_profiled(
                            m, verify_ids, block + 1, committed, NULL,
                            spec_logits, cfg.dspark_target_layer,
                            cfg.dspark_stages, spec_capture,
                            &spec_verify_stats) != 0)
                        return 1;

                    int accepted = 0;
                    while (accepted < block &&
                           proposal[accepted] ==
                               argmax_logits(spec_logits +
                                   (size_t)accepted * cfg.vocab, cfg.vocab))
                        accepted++;
                    if (getenv("DSV4_DEBUG_DSPARK")) {
                        fprintf(stderr, "dspark debug: proposal");
                        for (int i = 0; i < block; i++)
                            fprintf(stderr, " %d", proposal[i]);
                        fprintf(stderr, " | target");
                        for (int i = 0; i < block; i++)
                            fprintf(stderr, " %d", argmax_logits(
                                spec_logits + (size_t)i * cfg.vocab,
                                cfg.vocab));
                        fprintf(stderr, " | accepted=%d/%d\n", accepted,
                                block);
                    }
                    int eos_at = -1;
                    for (int i = 0; i < accepted; i++) {
                        if (proposal[i] == 1) { eos_at = i; break; }
                    }
                    spec_rounds++;
                    spec_accepted += accepted;
                    spec_accept_hist[accepted]++;
                    if (spec_reject_cooldown > 0 && accepted * 2 <= block) {
                        spec_cooldown = spec_reject_cooldown;
                        spec_cooldown_events++;
                    }

                    if (eos_at >= 0) {
                        const int replay = eos_at + 1;
                        if (dspark_commit_verified_prefix(
                                m, &cfg, spec_snapshot, verify_ids, replay,
                                committed, logits, spec_capture,
                                &spec_replay_stats) != 0)
                            return 1;
                        committed += replay;
                        for (int i = 0; i <= eos_at; i++) {
                            int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            spec_outputs++;
                        }
                        handled = 1;
                    } else if (accepted < block) {
                        int correction = argmax_logits(
                            spec_logits + (size_t)accepted * cfg.vocab,
                            cfg.vocab);
                        const int replay = accepted + 1;
                        if (dspark_commit_verified_prefix(
                                m, &cfg, spec_snapshot, verify_ids, replay,
                                committed, logits, spec_capture,
                                &spec_replay_stats) != 0)
                            return 1;
                        committed += replay;
                        for (int i = 0; i < accepted; i++) {
                            int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            spec_outputs++;
                        }
                        generated[gen++] = correction;
                        pending = correction;
                        history[history_len++] = correction;
                        emit_token(&tok, correction, piece, piece_cap, &output);
                        last_token_at = now_s();
                        spec_outputs++;
                        handled = 1;
                    } else {
                        int bonus = argmax_logits(
                            spec_logits + (size_t)block * cfg.vocab,
                            cfg.vocab);
                        if (dsv4_context_snapshot_commit_prefix(
                                spec_snapshot, block + 1) != 0 ||
                            dsv4_dspark_commit_target_hidden(
                                m, spec_capture, block + 1, committed) != 0)
                            return 1;
                        committed += block + 1;
                        for (int i = 0; i < block; i++) {
                            int nid = proposal[i];
                            generated[gen++] = nid;
                            pending = nid;
                            history[history_len++] = nid;
                            emit_token(&tok, nid, piece, piece_cap, &output);
                            last_token_at = now_s();
                            spec_outputs++;
                        }
                        generated[gen++] = bonus;
                        pending = bonus;
                        history[history_len++] = bonus;
                        emit_token(&tok, bonus, piece, piece_cap, &output);
                        last_token_at = now_s();
                        spec_outputs++;
                        handled = 1;
                    }
                }
            }
            if (handled) continue;

            /* Exact fallback and short tail. The pending token is committed
             * before sampling its successor, preserving the same state
             * convention used by speculative verification. */
            DSV4PhaseMark fallback_mark = dspark_phase_mark(m);
            int rc = use_dspark
                   ? prefill_with_dspark_state(m, &cfg, &pending, 1,
                                               committed, logits)
                   : dsv4_forward_token(m, pending, committed, logits);
            if (use_dspark)
                dspark_phase_add(&spec_fallback_stats, fallback_mark, m);
            if (rc != 0) return 1;
            committed++;
            int nid = temperature == 0.0
                    ? argmax_logits(logits, cfg.vocab)
                    : sample_top_p(logits, cfg.vocab, temperature, top_p,
                                   sample_items);
            generated[gen++] = nid;
            pending = nid;
            history[history_len++] = nid;
            emit_token(&tok, nid, piece, piece_cap, &output);
            last_token_at = now_s();
            if (spec_cooldown > 0) spec_cooldown--;
            if (ngram_cooldown > 0) ngram_cooldown--;
        }

        if (output.append_final_newline &&
            (output.bytes == 0 || output.last_byte != '\n'))
            (void)output_write(&output, "\n", 1u);
        if (output.failed) {
            fprintf(stderr, "dsv4: unable to write generated output\n");
            if (!server_port) return 1;
        }
        if (!interactive && !server_port) {
            fprintf(stderr, "generated ids:");
            for (int i = 0; i < gen; i++) fprintf(stderr, " %d", generated[i]);
            fprintf(stderr, "\n");
        }

        if (gen > 0) {
            fprintf(stderr, "timing: TTFT=%.3f s", first_token_at - inference_start);
            if (gen > 1)
                fprintf(stderr, " TPOT=%.3f s/token",
                        (last_token_at - first_token_at) / (gen - 1));
            else
                fprintf(stderr, " TPOT=n/a");
            fprintf(stderr, " (model-ready prefill start -> token delivered; "
                    "prefill=%d generated=%d)\n", prefill_tokens, gen);
        } else {
            fprintf(stderr, "timing: TTFT=n/a TPOT=n/a (no tokens generated)\n");
        }

        if (ngram_cache_expanded)
            (void)dsv4_expert_cache_set_hash_slots(m, cfg.topk);
        if (spec_cache_expanded)
            (void)dsv4_expert_cache_set_hash_slots(m, cfg.topk);

        if (server_port) {
            const int truncated = pending != 1 && gen >= response_limit;
            if (!output.failed &&
                !server_call_finish(&server_call, prefill_tokens,
                                    gen, truncated))
                fprintf(stderr,
                        "dsv4: request ended before a complete response\n");
            server_call_dispose(&server_call);
            dsv4_model_reset_context(m);
            committed = 0;
            history_len = 0;

            int *next_ids = NULL;
            int next_count = 0;
            char request_error[256];
            for (;;) {
                if (!server_next_call(server_fd, system, eff, tok.id2str[1],
                                      &server_call))
                    goto chat_done;
                if (server_call_tokenize(&tok, &server_call, context,
                                         &next_ids, &next_count,
                                         request_error,
                                         sizeof(request_error)))
                    break;
                (void)server_error_response(server_call.client, 400,
                                            "Bad Request", request_error);
                server_call_dispose(&server_call);
            }
            server_apply_call(&server_call, server_default_max_tokens,
                              server_default_temperature,
                              server_default_top_p, server_default_seed,
                              &max_tokens, &max_tokens_set,
                              &temperature, &top_p);
            if (!server_call_begin(&server_call)) {
                free(next_ids);
                goto chat_done;
            }
            output = server_call.output;
            stop_generation = 0;
            inference_start = now_s();
            if (dsv4_prefill(m, next_ids, next_count, 0, logits) != 0) {
                free(next_ids);
                goto chat_done;
            }
            memcpy(history, next_ids, (size_t)next_count * sizeof(int));
            history_len = next_count;
            committed = next_count;
            prefill_tokens = next_count;
            fprintf(stderr, "prompt: %d tokens\n", next_count);
            free(next_ids);
            continue;
        }

        if (!interactive) break;
        int *turn_ids = NULL;
        int nturn = 0, carry = 0;
        for (;;) {
            if (!read_chat_line(&input_line, &input_cap)) goto chat_done;
            int fresh = committed == 0;
            if (!strcmp(input_line, "/reset")) {
                dsv4_model_reset_context(m);
                committed = 0;
                pending = -1;
                history_len = 0;
                fprintf(stderr, "dsv4: conversation reset; model and caches remain resident\n");
                continue;
            }

            int need = fresh
                     ? dsv4_format_prompt(NULL, 0, input_line, system, eff)
                     : dsv4_format_turn(NULL, 0, input_line, eff);
            if (need < 0) continue;
            char *turn_text = (char *)malloc((size_t)need + 1);
            if (!turn_text) { fprintf(stderr, "dsv4: OOM turn buffer\n"); return 1; }
            if (fresh)
                dsv4_format_prompt(turn_text, (size_t)need + 1,
                                   input_line, system, eff);
            else
                dsv4_format_turn(turn_text, (size_t)need + 1, input_line, eff);

            carry = fresh ? 0 : (pending == 1 ? 1 : 2);
            int turn_cap = need + 1;
            turn_ids = (int *)malloc((size_t)(carry + turn_cap) * sizeof(int));
            if (!turn_ids) { free(turn_text); fprintf(stderr, "dsv4: OOM turn token buffer\n"); return 1; }
            if (!fresh) {
                turn_ids[0] = pending;
                if (pending != 1) turn_ids[1] = 1;
            }
            int encoded = tok_encode(&tok, turn_text, need,
                                     turn_ids + carry, turn_cap);
            free(turn_text);
            if (encoded <= 0) {
                fprintf(stderr, "dsv4: later turn produced no tokens\n");
                free(turn_ids);
                turn_ids = NULL;
                continue;
            }
            nturn = carry + encoded;
            if ((int64_t)committed + nturn >= context) {
                fprintf(stderr, "dsv4: chat context full (%d committed + %d turn >= %d); "
                        "use /reset for a fresh conversation\n",
                        committed, nturn, context);
                free(turn_ids);
                turn_ids = NULL;
                continue;
            }
            break;
        }

        fprintf(stderr, "turn: %d prefill tokens (%d carried from prior response)\n",
                nturn, carry);
        if (isatty(STDIN_FILENO)) {
            fputs("assistant> ", stdout);
            fflush(stdout);
        }
        stop_generation = 0;
        inference_start = now_s();
        if ((use_dspark
                 ? prefill_with_dspark_state(m, &cfg, turn_ids, nturn,
                                             committed, logits)
                 : dsv4_prefill(m, turn_ids, nturn, committed, logits)) != 0)
            return 1;
        memcpy(history + committed, turn_ids, (size_t)nturn * sizeof(int));
        history_len = committed + nturn;
        committed += nturn;
        prefill_tokens = nturn;
        free(turn_ids);
    }

chat_done:

    if (use_ngram) {
        fprintf(stderr,
                "ngram: mode=%s min_match=%d rounds=%ld drafted=%ld "
                "accepted=%ld outputs=%ld",
                ngram_periodic_extend ? "periodic" :
                ngram_allow_overlap ? "overlap" : "strict", ngram_min_match,
                ngram_rounds, ngram_drafted, ngram_accepted, ngram_outputs);
        if (ngram_rounds > 0)
            fprintf(stderr, " mean_accepted=%.2f\n",
                    (double)ngram_accepted / ngram_rounds);
        else
            fputc('\n', stderr);
        fprintf(stderr, "ngram: accepted histogram");
        int reported_acceptance = 0;
        for (int i = 0; i <= ngram_block; i++) {
            if (ngram_accept_hist[i] == 0) continue;
            fprintf(stderr, " %d:%ld", i, ngram_accept_hist[i]);
            reported_acceptance = 1;
        }
        if (!reported_acceptance) fputs(" none", stderr);
        fputc('\n', stderr);
        fprintf(stderr, "ngram: reject cooldown=%d events=%ld\n",
                ngram_reject_cooldown, ngram_cooldown_events);
        report_dspark_phase("ngram", &ngram_verify_stats);
    }
    if (use_dspark) {
        fprintf(stderr, "dspark: rounds=%ld accepted=%ld outputs=%ld",
                spec_rounds, spec_accepted, spec_outputs);
        if (spec_rounds > 0)
            fprintf(stderr, " mean_accepted=%.2f/%d\n",
                    (double)spec_accepted / spec_rounds,
                    spec_block);
        else
            fputc('\n', stderr);
        fprintf(stderr, "dspark: accepted histogram");
        for (int i = 0; i <= spec_block; i++)
            fprintf(stderr, " %d:%ld", i, spec_accept_hist[i]);
        fputc('\n', stderr);
        if (spec_reject_cooldown > 0)
            fprintf(stderr, "dspark: reject cooldown=%d events=%ld\n",
                    spec_reject_cooldown, spec_cooldown_events);
        report_dspark_phase("draft", &spec_draft_stats);
        report_dspark_phase("verify", &spec_verify_stats);
        report_dspark_phase("replay", &spec_replay_stats);
        report_dspark_phase("fallback", &spec_fallback_stats);
    }
    dsv4_model_report(m);

    dsv4_context_snapshot_free(spec_snapshot);
    free(spec_logits);
    free(spec_capture);
    dsv4_context_snapshot_free(ngram_snapshot);
    free(ngram_logits);
    server_call_dispose(&server_call);
    if (server_fd >= 0) close(server_fd);
    dsv4_model_close(m);
    free(ptext);
    free(ids);
    free(generated);
    free(history);
    free(logits);
    free(piece);
    free(sample_items);
    free(input_line);
    return 0;
}
