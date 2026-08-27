/* SPDX-License-Identifier: Apache-2.0 */
#include "dsv4_http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int expect_error(const char *json, const char *needle)
{
    DSV4HttpChatRequest request;
    char error[256] = {0};
    const int ok = dsv4_http_parse_chat_request(
        json, strlen(json), &request, error, sizeof(error));
    if (ok) dsv4_http_chat_request_free(&request);
    return !ok && strstr(error, needle) != NULL;
}

static int collect(void *context, const char *data, size_t length)
{
    return dsv4_buffer_append((DSV4Buffer *)context, data, length);
}

int main(void)
{
    static const char valid[] =
        "{\"model\":\"deepseek-v4-flash-0731-in-c\",\"messages\":["
        "{\"role\":\"system\",\"content\":\"line 1\\nline 2\"},"
        "{\"role\":\"user\",\"content\":\"\\u79d1\\u6280 \\ud83d\\ude80\"}],"
        "\"max_completion_tokens\":42,\"seed\":7,\"temperature\":0.5,"
        "\"top_p\":0.8,\"presence_penalty\":1.25,\"stream\":false,\"n\":1}";
    DSV4HttpChatRequest request;
    char error[256] = {0};
    CHECK(dsv4_http_parse_chat_request(valid, strlen(valid), &request,
                                       error, sizeof(error)));
    CHECK(strcmp(request.model, "deepseek-v4-flash-0731-in-c") == 0);
    CHECK(request.message_count == 2);
    CHECK(strcmp(request.messages[0].role, "system") == 0);
    CHECK(strcmp(request.messages[0].content, "line 1\nline 2") == 0);
    CHECK(strcmp(request.messages[1].content, "科技 🚀") == 0);
    CHECK(request.has_max_tokens && request.max_tokens == 42);
    CHECK(request.has_seed && request.seed == 7);
    CHECK(request.has_temperature && request.temperature == 0.5f);
    CHECK(request.has_top_p && request.top_p == 0.8f);
    CHECK(request.has_presence_penalty && request.presence_penalty == 1.25f);
    CHECK(!request.stream && !request.include_usage);
    dsv4_http_chat_request_free(&request);

    static const char streaming[] =
        "{\"model\":\"deepseek-v4-flash-0731-in-c\",\"messages\":["
        "{\"role\":\"developer\",\"content\":\"brief\"},"
        "{\"role\":\"user\",\"content\":\"hello\"}],"
        "\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
    CHECK(dsv4_http_parse_chat_request(streaming, strlen(streaming), &request,
                                       error, sizeof(error)));
    CHECK(request.stream && request.include_usage);
    dsv4_http_chat_request_free(&request);

    static const char history_json[] =
        "{\"model\":\"deepseek-v4-flash-0731-in-c\",\"messages\":["
        "{\"role\":\"developer\",\"content\":\"be brief\"},"
        "{\"role\":\"user\",\"content\":\"one\"},"
        "{\"role\":\"assistant\",\"content\":\"first\"},"
        "{\"role\":\"user\",\"content\":\"two\"}]}";
    CHECK(dsv4_http_parse_chat_request(history_json, strlen(history_json),
                                       &request, error, sizeof(error)));
    char *rendered = dsv4_http_render_messages(
        &request, "base", NULL, "<EOS>", error, sizeof(error));
    CHECK(rendered != NULL);
    CHECK(strcmp(rendered,
        "<｜begin▁of▁sentence｜>base\n\nbe brief<｜User｜>one"
        "<｜Assistant｜></think>first<EOS><｜User｜>two"
        "<｜Assistant｜></think>") == 0);
    free(rendered);
    dsv4_http_chat_request_free(&request);

    static const char bad_history[] =
        "{\"model\":\"deepseek-v4-flash-0731-in-c\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"one\"},"
        "{\"role\":\"user\",\"content\":\"two\"}]}";
    CHECK(dsv4_http_parse_chat_request(bad_history, strlen(bad_history),
                                       &request, error, sizeof(error)));
    rendered = dsv4_http_render_messages(
        &request, NULL, NULL, "<EOS>", error, sizeof(error));
    CHECK(rendered == NULL && strstr(error, "alternate") != NULL);
    dsv4_http_chat_request_free(&request);

    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":1}",
        "stream must"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream_options\":{}}",
        "requires stream"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\"}]}]}",
        "string role and content"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}",
        "unsupported message role"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"\\ud83dX\"}]}",
        "string role and content"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[],\"tools\":[]}",
        "tools"));
    CHECK(expect_error("{\"model\":\"x\",\"messages\":[]}",
                       "must not be empty"));
    CHECK(expect_error("[]", "one valid JSON object"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":65537}",
        "max_tokens"));
    CHECK(expect_error(
        "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]} true",
        "trailing JSON"));

    DSV4Buffer buffer;
    dsv4_buffer_init(&buffer);
    static const char raw[] = "quote=\" slash=\\ line=\n tab=\t";
    CHECK(dsv4_buffer_append_json_string(&buffer, raw, sizeof(raw) - 1u));
    CHECK(strcmp(buffer.data, "\"quote=\\\" slash=\\\\ line=\\n tab=\\t\"") == 0);
    dsv4_buffer_free(&buffer);

    DSV4Buffer streamed;
    dsv4_buffer_init(&streamed);
    DSV4Utf8Stream utf8;
    dsv4_utf8_stream_init(&utf8, collect, &streamed);
    static const char split_one[] = "A\xf0\x9f";
    static const char split_two[] = "\x9a\x80\xe7";
    static const char split_three[] = "\xa7\x91";
    CHECK(dsv4_utf8_stream_write(&utf8, split_one, sizeof(split_one) - 1u));
    CHECK(streamed.length == 1u && strcmp(streamed.data, "A") == 0);
    CHECK(dsv4_utf8_stream_write(&utf8, split_two, sizeof(split_two) - 1u));
    CHECK(strcmp(streamed.data, "A\xf0\x9f\x9a\x80") == 0);
    CHECK(dsv4_utf8_stream_write(&utf8, split_three,
                                 sizeof(split_three) - 1u));
    CHECK(strcmp(streamed.data, "A\xf0\x9f\x9a\x80\xe7\xa7\x91") == 0);
    static const char invalid[] = "\xff" "B";
    CHECK(dsv4_utf8_stream_write(&utf8, invalid, sizeof(invalid) - 1u));
    CHECK(strcmp(streamed.data,
                 "A\xf0\x9f\x9a\x80\xe7\xa7\x91\xef\xbf\xbd" "B") == 0);
    static const char incomplete[] = "\xe4\xb8";
    CHECK(dsv4_utf8_stream_write(&utf8, incomplete,
                                 sizeof(incomplete) - 1u));
    CHECK(dsv4_utf8_stream_flush(&utf8));
    CHECK(strcmp(streamed.data,
                 "A\xf0\x9f\x9a\x80\xe7\xa7\x91\xef\xbf\xbd" "B"
                 "\xef\xbf\xbd") == 0);
    dsv4_utf8_stream_free(&utf8);
    dsv4_buffer_free(&streamed);

    char *large = (char *)calloc(DSV4_HTTP_MAX_BODY + 2u, 1u);
    CHECK(large != NULL);
    memset(large, ' ', DSV4_HTTP_MAX_BODY + 1u);
    CHECK(!dsv4_http_parse_chat_request(large, DSV4_HTTP_MAX_BODY + 1u,
                                       &request, error, sizeof(error)));
    free(large);

    puts("dsv4 HTTP/JSON tests passed");
    return 0;
}
