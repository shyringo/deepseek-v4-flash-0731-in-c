/* SPDX-License-Identifier: Apache-2.0 */
#include "dsv4_http.h"
#include "dsv4.h"

#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSMN_STATIC
#include "jsmn.h"

static void dsv4_http_error(char *error, size_t capacity, const char *message)
{
    if (!error || capacity == 0) return;
    snprintf(error, capacity, "%s", message);
}

void dsv4_http_chat_request_init(DSV4HttpChatRequest *request)
{
    if (request) memset(request, 0, sizeof(*request));
}

void dsv4_http_chat_request_free(DSV4HttpChatRequest *request)
{
    if (!request) return;
    free(request->model);
    for (size_t i = 0; i < request->message_count; ++i) {
        free(request->messages[i].role);
        free(request->messages[i].content);
        free(request->messages[i].tool_call_id);
        for (size_t j = 0; j < request->messages[i].tool_call_count; ++j) {
            free(request->messages[i].tool_calls[j].id);
            free(request->messages[i].tool_calls[j].name);
            free(request->messages[i].tool_calls[j].arguments);
        }
    }
    for (size_t i = 0; i < request->tool_count; ++i) {
        free(request->tools[i].name);
        free(request->tools[i].function_json);
    }
    memset(request, 0, sizeof(*request));
}

void dsv4_buffer_init(DSV4Buffer *buffer)
{
    if (buffer) memset(buffer, 0, sizeof(*buffer));
}

void dsv4_buffer_free(DSV4Buffer *buffer)
{
    if (!buffer) return;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

int dsv4_buffer_append(DSV4Buffer *buffer, const char *data, size_t length)
{
    if (!buffer || (!data && length)) return 0;
    if (length > SIZE_MAX - buffer->length - 1u) return 0;
    const size_t needed = buffer->length + length + 1u;
    if (needed > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 256u;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = needed;
                break;
            }
            capacity *= 2u;
        }
        char *larger = (char *)realloc(buffer->data, capacity);
        if (!larger) return 0;
        buffer->data = larger;
        buffer->capacity = capacity;
    }
    if (length) memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return 1;
}

int dsv4_buffer_append_string(DSV4Buffer *buffer, const char *text)
{
    return text && dsv4_buffer_append(buffer, text, strlen(text));
}

int dsv4_buffer_append_json_string(DSV4Buffer *buffer,
                                  const char *text, size_t length)
{
    if (!dsv4_buffer_append(buffer, "\"", 1u)) return 0;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = (unsigned char)text[i];
        const char *escape = NULL;
        switch (c) {
        case '"': escape = "\\\""; break;
        case '\\': escape = "\\\\"; break;
        case '\b': escape = "\\b"; break;
        case '\f': escape = "\\f"; break;
        case '\n': escape = "\\n"; break;
        case '\r': escape = "\\r"; break;
        case '\t': escape = "\\t"; break;
        default: break;
        }
        if (escape) {
            if (!dsv4_buffer_append_string(buffer, escape)) return 0;
        } else if (c < 0x20u) {
            char encoded[7];
            snprintf(encoded, sizeof(encoded), "\\u%04x", c);
            if (!dsv4_buffer_append(buffer, encoded, 6u)) return 0;
        } else if (!dsv4_buffer_append(buffer, (const char *)&text[i], 1u)) {
            return 0;
        }
    }
    return dsv4_buffer_append(buffer, "\"", 1u);
}

static int dsv4_utf8_sequence(const unsigned char *text, size_t available,
                             size_t *length)
{
    const unsigned char first = text[0];
    if (first <= 0x7fu) {
        *length = 1u;
        return 1;
    }
    size_t needed = 0;
    if (first >= 0xc2u && first <= 0xdfu) needed = 2u;
    else if (first >= 0xe0u && first <= 0xefu) needed = 3u;
    else if (first >= 0xf0u && first <= 0xf4u) needed = 4u;
    else return -1;
    if (available < 2u) return 0;
    const unsigned char second = text[1];
    if ((second & 0xc0u) != 0x80u ||
        (first == 0xe0u && second < 0xa0u) ||
        (first == 0xedu && second >= 0xa0u) ||
        (first == 0xf0u && second < 0x90u) ||
        (first == 0xf4u && second >= 0x90u))
        return -1;
    if (available < needed) return 0;
    for (size_t i = 2u; i < needed; ++i)
        if ((text[i] & 0xc0u) != 0x80u) return -1;
    *length = needed;
    return 1;
}

void dsv4_utf8_stream_init(DSV4Utf8Stream *stream,
                          int (*emit)(void *context,
                                      const char *data, size_t length),
                          void *context)
{
    if (!stream) return;
    dsv4_buffer_init(&stream->pending);
    stream->emit = emit;
    stream->context = context;
}

void dsv4_utf8_stream_free(DSV4Utf8Stream *stream)
{
    if (!stream) return;
    dsv4_buffer_free(&stream->pending);
    stream->emit = NULL;
    stream->context = NULL;
}

int dsv4_utf8_stream_write(void *context, const char *data, size_t length)
{
    DSV4Utf8Stream *stream = (DSV4Utf8Stream *)context;
    if (!stream || !stream->emit || !dsv4_buffer_append(&stream->pending,
                                                        data, length))
        return 0;
    static const char replacement[] = "\xef\xbf\xbd";
    size_t cursor = 0;
    size_t run = 0;
    while (cursor < stream->pending.length) {
        size_t sequence = 0;
        const int status = dsv4_utf8_sequence(
            (const unsigned char *)stream->pending.data + cursor,
            stream->pending.length - cursor, &sequence);
        if (status == 0) break;
        if (status > 0) {
            cursor += sequence;
            continue;
        }
        if (cursor > run && !stream->emit(stream->context,
                                          stream->pending.data + run,
                                          cursor - run))
            return 0;
        if (!stream->emit(stream->context, replacement,
                          sizeof(replacement) - 1u))
            return 0;
        ++cursor;
        run = cursor;
    }
    if (cursor > run && !stream->emit(stream->context,
                                      stream->pending.data + run,
                                      cursor - run))
        return 0;
    const size_t remaining = stream->pending.length - cursor;
    if (remaining)
        memmove(stream->pending.data, stream->pending.data + cursor, remaining);
    stream->pending.length = remaining;
    if (stream->pending.data) stream->pending.data[remaining] = '\0';
    return 1;
}

int dsv4_utf8_stream_flush(DSV4Utf8Stream *stream)
{
    static const char replacement[] = "\xef\xbf\xbd";
    if (!stream || !stream->emit) return 0;
    if (stream->pending.length &&
        !stream->emit(stream->context, replacement, sizeof(replacement) - 1u))
        return 0;
    stream->pending.length = 0;
    if (stream->pending.data) stream->pending.data[0] = '\0';
    return 1;
}

static int dsv4_append_system(DSV4Buffer *system, const char *text)
{
    return (!system->length || dsv4_buffer_append_string(system, "\n\n")) &&
           dsv4_buffer_append_string(system, text);
}

static int dsv4_tool_name_valid(const char *name)
{
    const size_t length = name ? strlen(name) : 0u;
    if (!length || length > 128u) return 0;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = (unsigned char)name[i];
        if (!(isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':'))
            return 0;
    }
    return 1;
}

static int dsv4_json_token_next(const jsmntok_t *tokens, int count, int index)
{
    if (index < 0 || index >= count) return count;
    const int end = tokens[index].end;
    ++index;
    while (index < count && tokens[index].start < end) ++index;
    return index;
}

static char *dsv4_decode_string(const char *json, const jsmntok_t *token);

static int dsv4_append_history_calls(DSV4Buffer *prompt,
                                     const DSV4HttpMessage *message,
                                     char *error, size_t error_capacity)
{
    if (!dsv4_buffer_append_string(prompt, "\n\n<｜DSML｜tool_calls>\n"))
        return 0;
    for (size_t i = 0; i < message->tool_call_count; ++i) {
        const DSV4HttpToolCall *call = &message->tool_calls[i];
        if (!dsv4_tool_name_valid(call->name)) {
            dsv4_http_error(error, error_capacity,
                            "invalid tool name in assistant history");
            return 0;
        }
        if (!dsv4_buffer_append_string(prompt, "<｜DSML｜invoke name=\"") ||
            !dsv4_buffer_append_string(prompt, call->name) ||
            !dsv4_buffer_append_string(prompt, "\">\n"))
            return 0;
        jsmn_parser parser;
        jsmntok_t tokens[512];
        jsmn_init(&parser);
        const size_t arguments_length = strlen(call->arguments);
        const int count = jsmn_parse(&parser, call->arguments,
                                     arguments_length, tokens,
                                     sizeof(tokens) / sizeof(tokens[0]));
        if (count < 1 || tokens[0].type != JSMN_OBJECT ||
            tokens[0].start != 0 || tokens[0].end != (int)arguments_length) {
            dsv4_http_error(error, error_capacity,
                            "tool_call arguments must be one JSON object");
            return 0;
        }
        int index = 1;
        while (index < count && tokens[index].start < tokens[0].end) {
            const int value = index + 1;
            if (value >= count || tokens[index].type != JSMN_STRING ||
                tokens[value].start >= tokens[0].end) {
                dsv4_http_error(error, error_capacity,
                                "invalid tool_call arguments object");
                return 0;
            }
            char *name = dsv4_decode_string(call->arguments, &tokens[index]);
            if (!dsv4_tool_name_valid(name)) {
                free(name);
                dsv4_http_error(error, error_capacity,
                                "invalid tool parameter name");
                return 0;
            }
            const int is_string = tokens[value].type == JSMN_STRING;
            char *string_value = is_string
                               ? dsv4_decode_string(call->arguments,
                                                    &tokens[value]) : NULL;
            int ok = dsv4_buffer_append_string(
                         prompt, "<｜DSML｜parameter name=\"") &&
                     dsv4_buffer_append_string(prompt, name) &&
                     dsv4_buffer_append_string(
                         prompt, is_string
                             ? "\" string=\"true\">"
                             : "\" string=\"false\">");
            free(name);
            if (ok && is_string) ok = string_value &&
                dsv4_buffer_append_string(prompt, string_value);
            else if (ok) ok = dsv4_buffer_append(
                prompt, call->arguments + tokens[value].start,
                (size_t)(tokens[value].end - tokens[value].start));
            free(string_value);
            if (!ok || !dsv4_buffer_append_string(
                    prompt, "</｜DSML｜parameter>\n"))
                return 0;
            index = dsv4_json_token_next(tokens, count, value);
        }
        if (!dsv4_buffer_append_string(prompt, "</｜DSML｜invoke>\n"))
            return 0;
    }
    return dsv4_buffer_append_string(prompt, "</｜DSML｜tool_calls>");
}

static int dsv4_append_tools_system(DSV4Buffer *system,
                                    const DSV4HttpChatRequest *request)
{
    static const char header[] =
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user's question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like "
        "the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">"
        "$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. "
        "For all other types (numbers, booleans, arrays, objects), pass the "
        "value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output "
        "your complete reasoning inside <think>...</think> BEFORE any tool "
        "calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final "
        "response.\n\n"
        "### Available Tool Schemas\n\n";
    static const char footer[] =
        "\nYou MUST strictly follow the above defined tool name and parameter "
        "schemas to invoke tool calls.\n";
    if (!request->tool_count || request->tool_choice_none) return 1;
    if (system->length && !dsv4_buffer_append_string(system, "\n\n")) return 0;
    if (!dsv4_buffer_append_string(system, header)) return 0;
    for (size_t i = 0; i < request->tool_count; ++i) {
        if (!dsv4_tool_name_valid(request->tools[i].name) ||
            !dsv4_buffer_append_string(system,
                                       request->tools[i].function_json) ||
            !dsv4_buffer_append_string(system, "\n"))
            return 0;
    }
    return dsv4_buffer_append_string(system, footer);
}

char *dsv4_http_render_messages(const DSV4HttpChatRequest *request,
                                const char *base_system,
                                const char *reasoning_effort,
                                const char *eos_text,
                                char *error, size_t error_capacity)
{
    if (!request || request->message_count == 0) {
        dsv4_http_error(error, error_capacity, "messages must not be empty");
        return NULL;
    }
    DSV4Buffer system;
    DSV4Buffer prompt;
    dsv4_buffer_init(&system);
    dsv4_buffer_init(&prompt);
    int ok = 1;
    if (base_system && *base_system)
        ok = dsv4_append_system(&system, base_system);
    size_t instruction_end = 0;
    while (instruction_end < request->message_count &&
           (strcmp(request->messages[instruction_end].role, "system") == 0 ||
            strcmp(request->messages[instruction_end].role, "developer") == 0)) {
        if (ok) ok = dsv4_append_system(
            &system, request->messages[instruction_end].content);
        ++instruction_end;
    }
    if (ok) ok = dsv4_append_tools_system(&system, request);
    int first = 1;
    int assistant_open = 0;
    int tool_results = 0;
    const DSV4HttpMessage *tool_source = NULL;
    int tool_result_seen[DSV4_HTTP_MAX_TOOL_CALLS] = {0};
    for (size_t index = 0; ok && index < request->message_count; ++index) {
        const DSV4HttpMessage *message = &request->messages[index];
        if (index < instruction_end) continue;
        if (strcmp(message->role, "system") == 0) {
            dsv4_http_error(error, error_capacity,
                            "system messages must appear first");
            ok = 0;
            break;
        }
        if (strcmp(message->role, "user") == 0 ||
            strcmp(message->role, "developer") == 0) {
            if (assistant_open || tool_results) {
                dsv4_http_error(error, error_capacity,
                                "invalid user/assistant/tool message order");
                ok = 0;
                break;
            }
            DSV4Buffer user_group;
            dsv4_buffer_init(&user_group);
            size_t group_end = index;
            while (group_end < request->message_count &&
                   (strcmp(request->messages[group_end].role, "user") == 0 ||
                    strcmp(request->messages[group_end].role,
                           "developer") == 0)) {
                if (!request->messages[group_end].content[0] ||
                    (user_group.length &&
                     !dsv4_buffer_append_string(&user_group, "\n\n")) ||
                    !dsv4_buffer_append_string(
                        &user_group, request->messages[group_end].content)) {
                    ok = 0;
                    break;
                }
                ++group_end;
            }
            if (!ok) {
                dsv4_buffer_free(&user_group);
                dsv4_http_error(error, error_capacity,
                                "invalid or oversized user message group");
                break;
            }
            const int needed = first
                ? dsv4_format_prompt(NULL, 0, user_group.data,
                                     system.data ? system.data : "",
                                     reasoning_effort)
                : dsv4_format_turn(NULL, 0, user_group.data,
                                   reasoning_effort);
            char *turn = needed >= 0
                       ? (char *)malloc((size_t)needed + 1u) : NULL;
            if (!turn) {
                dsv4_http_error(error, error_capacity,
                                "out of memory rendering messages");
                ok = 0;
                dsv4_buffer_free(&user_group);
                break;
            }
            if (first)
                dsv4_format_prompt(turn, (size_t)needed + 1u,
                                   user_group.data,
                                   system.data ? system.data : "",
                                   reasoning_effort);
            else
                dsv4_format_turn(turn, (size_t)needed + 1u,
                                 user_group.data, reasoning_effort);
            ok = dsv4_buffer_append(&prompt, turn, (size_t)needed);
            free(turn);
            dsv4_buffer_free(&user_group);
            first = 0;
            assistant_open = 1;
            index = group_end - 1u;
        } else if (strcmp(message->role, "assistant") == 0) {
            if (!assistant_open ||
                (!message->content[0] && !message->tool_call_count) ||
                !eos_text || !*eos_text) {
                dsv4_http_error(error, error_capacity,
                                "invalid assistant message position");
                ok = 0;
                break;
            }
            if (reasoning_effort)
                ok = dsv4_buffer_append_string(&prompt, "</think>");
            if (ok && message->content[0])
                ok = dsv4_buffer_append_string(&prompt, message->content);
            if (ok && message->tool_call_count)
                ok = dsv4_append_history_calls(&prompt, message,
                                                error, error_capacity);
            if (ok) ok = dsv4_buffer_append_string(&prompt, eos_text);
            assistant_open = 0;
            tool_results = message->tool_call_count > 0;
            tool_source = message->tool_call_count ? message : NULL;
            memset(tool_result_seen, 0, sizeof(tool_result_seen));
        } else if (strcmp(message->role, "tool") == 0) {
            if (assistant_open || !tool_results || !tool_source ||
                !message->content[0] || !message->tool_call_id) {
                dsv4_http_error(error, error_capacity,
                                "tool result does not follow tool_calls");
                ok = 0;
                break;
            }
            size_t matched = tool_source->tool_call_count;
            for (size_t i = 0; i < tool_source->tool_call_count; ++i) {
                if (tool_source->tool_calls[i].id &&
                    strcmp(tool_source->tool_calls[i].id,
                           message->tool_call_id) == 0) {
                    matched = i;
                    break;
                }
            }
            if (matched == tool_source->tool_call_count ||
                tool_result_seen[matched]) {
                dsv4_http_error(error, error_capacity,
                                "tool_call_id is unknown or duplicated");
                ok = 0;
                break;
            }
            tool_result_seen[matched] = 1;
            if (!dsv4_buffer_append_string(
                    &prompt, tool_results == 1
                        ? "<｜User｜><tool_result>" : "\n\n<tool_result>") ||
                !dsv4_buffer_append_string(&prompt, message->content) ||
                !dsv4_buffer_append_string(&prompt, "</tool_result>"))
                ok = 0;
            ++tool_results;
        } else {
            dsv4_http_error(error, error_capacity, "unsupported message role");
            ok = 0;
        }
    }
    if (ok && tool_results) {
        if ((size_t)(tool_results - 1) != tool_source->tool_call_count) {
            dsv4_http_error(error, error_capacity,
                            "every tool_call needs exactly one result");
            ok = 0;
        }
    }
    if (ok && tool_results) {
        ok = dsv4_buffer_append_string(&prompt, "<｜Assistant｜>") &&
             dsv4_buffer_append_string(
                 &prompt, reasoning_effort ? "<think>" : "</think>");
        assistant_open = 1;
    }
    if (ok && !assistant_open) {
        dsv4_http_error(error, error_capacity,
                        "the final message must request an assistant response");
        ok = 0;
    }
    dsv4_buffer_free(&system);
    if (!ok) {
        dsv4_buffer_free(&prompt);
        if (error && error_capacity && !error[0])
            dsv4_http_error(error, error_capacity,
                            "out of memory rendering messages");
        return NULL;
    }
    return prompt.data;
}

static int dsv4_token_equal(const char *json, const jsmntok_t *token,
                           const char *text)
{
    const size_t length = strlen(text);
    return token->type == JSMN_STRING &&
           token->end >= token->start &&
           (size_t)(token->end - token->start) == length &&
           memcmp(json + token->start, text, length) == 0;
}

static int dsv4_token_next(const jsmntok_t *tokens, int count, int index)
{
    if (index < 0 || index >= count) return count;
    const int end = tokens[index].end;
    ++index;
    while (index < count && tokens[index].start < end) ++index;
    return index;
}

static int dsv4_object_get(const char *json, const jsmntok_t *tokens,
                          int count, int object, const char *key)
{
    if (object < 0 || object >= count || tokens[object].type != JSMN_OBJECT)
        return -1;
    int index = object + 1;
    while (index < count && tokens[index].start < tokens[object].end) {
        const int value = index + 1;
        if (value >= count || tokens[value].start >= tokens[object].end)
            return -1;
        if (dsv4_token_equal(json, &tokens[index], key)) return value;
        index = dsv4_token_next(tokens, count, value);
    }
    return -1;
}

static int dsv4_hex(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int dsv4_utf8(DSV4Buffer *output, uint32_t codepoint)
{
    char bytes[4];
    size_t count = 0;
    if (codepoint <= 0x7fu) {
        bytes[count++] = (char)codepoint;
    } else if (codepoint <= 0x7ffu) {
        bytes[count++] = (char)(0xc0u | (codepoint >> 6));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0xffffu) {
        if (codepoint >= 0xd800u && codepoint <= 0xdfffu) return 0;
        bytes[count++] = (char)(0xe0u | (codepoint >> 12));
        bytes[count++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0x10ffffu) {
        bytes[count++] = (char)(0xf0u | (codepoint >> 18));
        bytes[count++] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
        bytes[count++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        bytes[count++] = (char)(0x80u | (codepoint & 0x3fu));
    } else {
        return 0;
    }
    return dsv4_buffer_append(output, bytes, count);
}

static char *dsv4_decode_string(const char *json, const jsmntok_t *token)
{
    if (!json || !token || token->type != JSMN_STRING ||
        token->start < 0 || token->end < token->start)
        return NULL;
    DSV4Buffer output;
    dsv4_buffer_init(&output);
    for (int i = token->start; i < token->end; ++i) {
        const unsigned char c = (unsigned char)json[i];
        if (c != '\\') {
            if (c < 0x20u || !dsv4_buffer_append(&output, (const char *)&json[i], 1u))
                goto fail;
            continue;
        }
        if (++i >= token->end) goto fail;
        const char escaped = json[i];
        char decoded = 0;
        switch (escaped) {
        case '"': decoded = '"'; break;
        case '\\': decoded = '\\'; break;
        case '/': decoded = '/'; break;
        case 'b': decoded = '\b'; break;
        case 'f': decoded = '\f'; break;
        case 'n': decoded = '\n'; break;
        case 'r': decoded = '\r'; break;
        case 't': decoded = '\t'; break;
        case 'u': {
            if (i + 4 >= token->end) goto fail;
            uint32_t codepoint = 0;
            for (int j = 0; j < 4; ++j) {
                const int digit = dsv4_hex(json[++i]);
                if (digit < 0) goto fail;
                codepoint = (codepoint << 4) | (uint32_t)digit;
            }
            if (codepoint >= 0xd800u && codepoint <= 0xdbffu) {
                if (i + 6 >= token->end || json[i + 1] != '\\' ||
                    json[i + 2] != 'u') goto fail;
                i += 2;
                uint32_t low = 0;
                for (int j = 0; j < 4; ++j) {
                    const int digit = dsv4_hex(json[++i]);
                    if (digit < 0) goto fail;
                    low = (low << 4) | (uint32_t)digit;
                }
                if (low < 0xdc00u || low > 0xdfffu) goto fail;
                codepoint = 0x10000u + ((codepoint - 0xd800u) << 10) +
                            (low - 0xdc00u);
            }
            if (!dsv4_utf8(&output, codepoint)) goto fail;
            continue;
        }
        default: goto fail;
        }
        if (!dsv4_buffer_append(&output, &decoded, 1u)) goto fail;
    }
    if (!output.data) {
        output.data = (char *)calloc(1u, 1u);
        if (!output.data) return NULL;
    }
    return output.data;
fail:
    dsv4_buffer_free(&output);
    return NULL;
}

static char *dsv4_copy_token_json(const char *json, const jsmntok_t *token)
{
    if (!json || !token || token->start < 0 || token->end < token->start)
        return NULL;
    const size_t length = (size_t)(token->end - token->start);
    char *copy = (char *)malloc(length + 1u);
    if (!copy) return NULL;
    memcpy(copy, json + token->start, length);
    copy[length] = '\0';
    return copy;
}

static int dsv4_parse_u64(const char *json, const jsmntok_t *token,
                         uint64_t *value)
{
    if (!token || token->type != JSMN_PRIMITIVE || token->end <= token->start)
        return 0;
    const size_t length = (size_t)(token->end - token->start);
    if (length >= 64u) return 0;
    char text[64];
    memcpy(text, json + token->start, length);
    text[length] = '\0';
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end || text[0] == '-') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int dsv4_parse_float(const char *json, const jsmntok_t *token,
                           float *value)
{
    if (!token || token->type != JSMN_PRIMITIVE || token->end <= token->start)
        return 0;
    const size_t length = (size_t)(token->end - token->start);
    if (length >= 64u) return 0;
    char text[64];
    memcpy(text, json + token->start, length);
    text[length] = '\0';
    char *end = NULL;
    errno = 0;
    const float parsed = strtof(text, &end);
    if (errno || end == text || *end || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static int dsv4_primitive_is(const char *json, const jsmntok_t *token,
                            const char *value)
{
    if (!token || token->type != JSMN_PRIMITIVE) return 0;
    const size_t length = strlen(value);
    return token->end >= token->start &&
           (size_t)(token->end - token->start) == length &&
           memcmp(json + token->start, value, length) == 0;
}

static int dsv4_supported_role(const char *role)
{
    return strcmp(role, "system") == 0 || strcmp(role, "developer") == 0 ||
           strcmp(role, "user") == 0 || strcmp(role, "assistant") == 0 ||
           strcmp(role, "tool") == 0;
}

int dsv4_http_parse_chat_request(const char *json, size_t length,
                                DSV4HttpChatRequest *request,
                                char *error, size_t error_capacity)
{
    if (!json || !request || length == 0 || length > DSV4_HTTP_MAX_BODY) {
        dsv4_http_error(error, error_capacity, "request body is empty or too large");
        return 0;
    }
    dsv4_http_chat_request_init(request);
    unsigned int capacity = 256u;
    jsmntok_t *tokens = NULL;
    int count = JSMN_ERROR_NOMEM;
    while (count == JSMN_ERROR_NOMEM && capacity <= 16384u) {
        jsmntok_t *larger = (jsmntok_t *)realloc(tokens,
                                                  capacity * sizeof(*tokens));
        if (!larger) break;
        tokens = larger;
        jsmn_parser parser;
        jsmn_init(&parser);
        count = jsmn_parse(&parser, json, length, tokens, capacity);
        capacity *= 2u;
    }
    if (count < 1 || tokens[0].type != JSMN_OBJECT) {
        dsv4_http_error(error, error_capacity, "body must be one valid JSON object");
        free(tokens);
        return 0;
    }
    for (size_t i = (size_t)tokens[0].end; i < length; ++i) {
        if (json[i] != ' ' && json[i] != '\t' &&
            json[i] != '\r' && json[i] != '\n') {
            dsv4_http_error(error, error_capacity,
                           "body contains trailing JSON data");
            free(tokens);
            return 0;
        }
    }

    int token = dsv4_object_get(json, tokens, count, 0, "model");
    if (token < 0 || !(request->model = dsv4_decode_string(json, &tokens[token]))) {
        dsv4_http_error(error, error_capacity, "model must be a JSON string");
        goto fail;
    }
    if (strlen(request->model) > 128u) {
        dsv4_http_error(error, error_capacity, "model is too long");
        goto fail;
    }

    token = dsv4_object_get(json, tokens, count, 0, "stream");
    if (token >= 0) {
        if (dsv4_primitive_is(json, &tokens[token], "true")) request->stream = 1;
        else if (!dsv4_primitive_is(json, &tokens[token], "false")) {
            dsv4_http_error(error, error_capacity, "stream must be true or false");
            goto fail;
        }
    }
    token = dsv4_object_get(json, tokens, count, 0, "stream_options");
    if (token >= 0) {
        if (!request->stream || tokens[token].type != JSMN_OBJECT) {
            dsv4_http_error(error, error_capacity,
                           "stream_options requires stream true");
            goto fail;
        }
        const int include = dsv4_object_get(json, tokens, count, token,
                                           "include_usage");
        if (include >= 0) {
            if (dsv4_primitive_is(json, &tokens[include], "true"))
                request->include_usage = 1;
            else if (!dsv4_primitive_is(json, &tokens[include], "false")) {
                dsv4_http_error(error, error_capacity,
                               "include_usage must be true or false");
                goto fail;
            }
        }
    }
    token = dsv4_object_get(json, tokens, count, 0, "n");
    if (token >= 0) {
        uint64_t n = 0;
        if (!dsv4_parse_u64(json, &tokens[token], &n) || n != 1u) {
            dsv4_http_error(error, error_capacity, "n must be 1");
            goto fail;
        }
    }
    if (dsv4_object_get(json, tokens, count, 0, "response_format") >= 0) {
        dsv4_http_error(error, error_capacity,
                        "structured output is not supported");
        goto fail;
    }
    token = dsv4_object_get(json, tokens, count, 0, "tool_choice");
    if (token >= 0) {
        if (dsv4_token_equal(json, &tokens[token], "none")) {
            request->tool_choice_none = 1;
        } else if (!dsv4_token_equal(json, &tokens[token], "auto")) {
            dsv4_http_error(error, error_capacity,
                "tool_choice currently supports only auto or none");
            goto fail;
        }
    }
    const int tools_token = dsv4_object_get(json, tokens, count, 0, "tools");
    if (tools_token >= 0) {
        if (tokens[tools_token].type != JSMN_ARRAY) {
            dsv4_http_error(error, error_capacity, "tools must be a JSON array");
            goto fail;
        }
        int tool_index = tools_token + 1;
        while (tool_index < count &&
               tokens[tool_index].start < tokens[tools_token].end) {
            if (tokens[tool_index].type != JSMN_OBJECT ||
                request->tool_count >= DSV4_HTTP_MAX_TOOLS) {
                dsv4_http_error(error, error_capacity,
                                "tools contains too many or invalid items");
                goto fail;
            }
            const int type = dsv4_object_get(json, tokens, count,
                                             tool_index, "type");
            const int function = dsv4_object_get(json, tokens, count,
                                                 tool_index, "function");
            if (type < 0 || function < 0 ||
                !dsv4_token_equal(json, &tokens[type], "function") ||
                tokens[function].type != JSMN_OBJECT) {
                dsv4_http_error(error, error_capacity,
                                "only function tools are supported");
                goto fail;
            }
            const int name = dsv4_object_get(json, tokens, count,
                                             function, "name");
            const int strict = dsv4_object_get(json, tokens, count,
                                               function, "strict");
            if (strict >= 0 &&
                !dsv4_primitive_is(json, &tokens[strict], "false")) {
                dsv4_http_error(error, error_capacity,
                                "strict function tools are not supported");
                goto fail;
            }
            DSV4HttpTool *tool = &request->tools[request->tool_count];
            if (name < 0 ||
                !(tool->name = dsv4_decode_string(json, &tokens[name])) ||
                !(tool->function_json = dsv4_copy_token_json(
                    json, &tokens[function]))) {
                free(tool->name);
                free(tool->function_json);
                tool->name = NULL;
                tool->function_json = NULL;
                dsv4_http_error(error, error_capacity,
                                "each function tool needs a name and schema");
                goto fail;
            }
            ++request->tool_count;
            for (size_t i = 0; i + 1u < request->tool_count; ++i) {
                if (strcmp(request->tools[i].name, tool->name) == 0) {
                    dsv4_http_error(error, error_capacity,
                                    "tool names must be unique");
                    goto fail;
                }
            }
            tool_index = dsv4_token_next(tokens, count, tool_index);
        }
    }

    token = dsv4_object_get(json, tokens, count, 0, "max_completion_tokens");
    if (token < 0) token = dsv4_object_get(json, tokens, count, 0, "max_tokens");
    if (token >= 0) {
        uint64_t value = 0;
        if (!dsv4_parse_u64(json, &tokens[token], &value) ||
            value == 0 || value > 65536u) {
            dsv4_http_error(error, error_capacity, "max_tokens is out of range");
            goto fail;
        }
        request->max_tokens = (uint32_t)value;
        request->has_max_tokens = 1;
    }
    token = dsv4_object_get(json, tokens, count, 0, "seed");
    if (token >= 0) {
        if (!dsv4_parse_u64(json, &tokens[token], &request->seed)) {
            dsv4_http_error(error, error_capacity, "seed must be a non-negative integer");
            goto fail;
        }
        request->has_seed = 1;
    }
    token = dsv4_object_get(json, tokens, count, 0, "temperature");
    if (token >= 0) {
        if (!dsv4_parse_float(json, &tokens[token], &request->temperature) ||
            request->temperature < 0.0f || request->temperature > 2.0f) {
            dsv4_http_error(error, error_capacity, "temperature must be between 0 and 2");
            goto fail;
        }
        request->has_temperature = 1;
    }
    token = dsv4_object_get(json, tokens, count, 0, "top_p");
    if (token >= 0) {
        if (!dsv4_parse_float(json, &tokens[token], &request->top_p) ||
            request->top_p <= 0.0f || request->top_p > 1.0f) {
            dsv4_http_error(error, error_capacity, "top_p must be in (0, 1]");
            goto fail;
        }
        request->has_top_p = 1;
    }
    token = dsv4_object_get(json, tokens, count, 0, "presence_penalty");
    if (token >= 0) {
        if (!dsv4_parse_float(json, &tokens[token], &request->presence_penalty) ||
            request->presence_penalty < 0.0f || request->presence_penalty > 2.0f) {
            dsv4_http_error(error, error_capacity, "presence_penalty must be between 0 and 2");
            goto fail;
        }
        request->has_presence_penalty = 1;
    }

    const int messages = dsv4_object_get(json, tokens, count, 0, "messages");
    if (messages < 0 || tokens[messages].type != JSMN_ARRAY) {
        dsv4_http_error(error, error_capacity, "messages must be a JSON array");
        goto fail;
    }
    size_t total_text = 0;
    int index = messages + 1;
    while (index < count && tokens[index].start < tokens[messages].end) {
        if (tokens[index].type != JSMN_OBJECT ||
            request->message_count >= DSV4_HTTP_MAX_MESSAGES) {
            dsv4_http_error(error, error_capacity, "messages contains too many or invalid items");
            goto fail;
        }
        const int role = dsv4_object_get(json, tokens, count, index, "role");
        const int content = dsv4_object_get(json, tokens, count, index,
                                            "content");
        DSV4HttpMessage *message = &request->messages[request->message_count];
        if (role < 0 || content < 0) {
            dsv4_http_error(error, error_capacity,
                            "each message needs role and content");
            goto fail;
        }
        message->role = dsv4_decode_string(json, &tokens[role]);
        if (tokens[content].type == JSMN_STRING)
            message->content = dsv4_decode_string(json, &tokens[content]);
        else if (dsv4_primitive_is(json, &tokens[content], "null"))
            message->content = (char *)calloc(1u, 1u);
        if (!message->role || !message->content) {
            free(message->role);
            free(message->content);
            message->role = NULL;
            message->content = NULL;
            dsv4_http_error(error, error_capacity,
                            "message content must be a string or null");
            goto fail;
        }
        ++request->message_count;
        if (!dsv4_supported_role(message->role)) {
            dsv4_http_error(error, error_capacity, "unsupported message role");
            goto fail;
        }
        const int tool_call_id = dsv4_object_get(
            json, tokens, count, index, "tool_call_id");
        if (tool_call_id >= 0 &&
            !(message->tool_call_id = dsv4_decode_string(
                json, &tokens[tool_call_id]))) {
            dsv4_http_error(error, error_capacity,
                            "tool_call_id must be a string");
            goto fail;
        }
        const int calls = dsv4_object_get(json, tokens, count,
                                          index, "tool_calls");
        if (calls >= 0) {
            if (tokens[calls].type != JSMN_ARRAY ||
                strcmp(message->role, "assistant") != 0) {
                dsv4_http_error(error, error_capacity,
                                "tool_calls belongs on an assistant message");
                goto fail;
            }
            int call_index = calls + 1;
            while (call_index < count &&
                   tokens[call_index].start < tokens[calls].end) {
                if (tokens[call_index].type != JSMN_OBJECT ||
                    message->tool_call_count >= DSV4_HTTP_MAX_TOOL_CALLS) {
                    dsv4_http_error(error, error_capacity,
                                    "too many or invalid tool_calls");
                    goto fail;
                }
                const int call_type = dsv4_object_get(
                    json, tokens, count, call_index, "type");
                const int call_id = dsv4_object_get(
                    json, tokens, count, call_index, "id");
                const int function = dsv4_object_get(
                    json, tokens, count, call_index, "function");
                if (call_type < 0 || function < 0 ||
                    !dsv4_token_equal(json, &tokens[call_type], "function") ||
                    tokens[function].type != JSMN_OBJECT) {
                    dsv4_http_error(error, error_capacity,
                                    "only function tool_calls are supported");
                    goto fail;
                }
                const int call_name = dsv4_object_get(
                    json, tokens, count, function, "name");
                const int arguments = dsv4_object_get(
                    json, tokens, count, function, "arguments");
                DSV4HttpToolCall *call =
                    &message->tool_calls[message->tool_call_count];
                if (call_id >= 0)
                    call->id = dsv4_decode_string(json, &tokens[call_id]);
                call->name = call_name >= 0
                           ? dsv4_decode_string(json, &tokens[call_name]) : NULL;
                call->arguments = arguments >= 0
                                ? dsv4_decode_string(json, &tokens[arguments])
                                : NULL;
                if ((call_id >= 0 && !call->id) ||
                    !call->name || !call->arguments) {
                    free(call->id);
                    free(call->name);
                    free(call->arguments);
                    call->id = NULL;
                    call->name = NULL;
                    call->arguments = NULL;
                    dsv4_http_error(error, error_capacity,
                                    "invalid function tool_call");
                    goto fail;
                }
                ++message->tool_call_count;
                call_index = dsv4_token_next(tokens, count, call_index);
            }
        }
        if (strcmp(message->role, "tool") == 0 &&
            (!message->tool_call_id || !message->content[0])) {
            dsv4_http_error(error, error_capacity,
                            "tool messages need tool_call_id and content");
            goto fail;
        }
        if (message->tool_call_count && !request->tool_count) {
            dsv4_http_error(error, error_capacity,
                            "tool_calls history requires tools");
            goto fail;
        }
        total_text += strlen(message->role) + strlen(message->content) +
                      (message->tool_call_id
                           ? strlen(message->tool_call_id) : 0u);
        for (size_t i = 0; i < message->tool_call_count; ++i)
            total_text += strlen(message->tool_calls[i].name) +
                          strlen(message->tool_calls[i].arguments);
        if (total_text > DSV4_HTTP_MAX_TEXT) {
            dsv4_http_error(error, error_capacity, "decoded message text is too large");
            goto fail;
        }
        index = dsv4_token_next(tokens, count, index);
    }
    if (request->message_count == 0) {
        dsv4_http_error(error, error_capacity, "messages must not be empty");
        goto fail;
    }
    free(tokens);
    return 1;

fail:
    free(tokens);
    dsv4_http_chat_request_free(request);
    return 0;
}
