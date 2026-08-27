/* SPDX-License-Identifier: Apache-2.0 */
#ifndef DSV4_HTTP_H
#define DSV4_HTTP_H

#include <stddef.h>
#include <stdint.h>

#define DSV4_HTTP_MAX_MESSAGES 128u
#define DSV4_HTTP_MAX_BODY (1024u * 1024u)
#define DSV4_HTTP_MAX_TEXT (256u * 1024u)

typedef struct {
    char *role;
    char *content;
} DSV4HttpMessage;

typedef struct {
    char *model;
    DSV4HttpMessage messages[DSV4_HTTP_MAX_MESSAGES];
    size_t message_count;
    uint32_t max_tokens;
    uint64_t seed;
    float temperature;
    float top_p;
    float presence_penalty;
    int stream;
    int include_usage;
    int has_max_tokens;
    int has_seed;
    int has_temperature;
    int has_top_p;
    int has_presence_penalty;
} DSV4HttpChatRequest;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} DSV4Buffer;

typedef struct {
    DSV4Buffer pending;
    int (*emit)(void *context, const char *data, size_t length);
    void *context;
} DSV4Utf8Stream;

void dsv4_http_chat_request_init(DSV4HttpChatRequest *request);
void dsv4_http_chat_request_free(DSV4HttpChatRequest *request);

int dsv4_http_parse_chat_request(const char *json, size_t length,
                                DSV4HttpChatRequest *request,
                                char *error, size_t error_capacity);
char *dsv4_http_render_messages(const DSV4HttpChatRequest *request,
                                const char *base_system,
                                const char *reasoning_effort,
                                const char *eos_text,
                                char *error, size_t error_capacity);

void dsv4_buffer_init(DSV4Buffer *buffer);
void dsv4_buffer_free(DSV4Buffer *buffer);
int dsv4_buffer_append(DSV4Buffer *buffer, const char *data, size_t length);
int dsv4_buffer_append_string(DSV4Buffer *buffer, const char *text);
int dsv4_buffer_append_json_string(DSV4Buffer *buffer,
                                  const char *text, size_t length);

void dsv4_utf8_stream_init(DSV4Utf8Stream *stream,
                          int (*emit)(void *context,
                                      const char *data, size_t length),
                          void *context);
void dsv4_utf8_stream_free(DSV4Utf8Stream *stream);
int dsv4_utf8_stream_write(void *context, const char *data, size_t length);
int dsv4_utf8_stream_flush(DSV4Utf8Stream *stream);

#endif
