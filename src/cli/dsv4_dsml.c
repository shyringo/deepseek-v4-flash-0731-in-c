/* SPDX-License-Identifier: Apache-2.0 */
#include "dsv4_dsml.h"
#include "dsv4_http.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSMN_STATIC
#include "jsmn.h"

static const char tool_start[] = "<｜DSML｜tool_calls>";
static const char tool_end[] = "</｜DSML｜tool_calls>";
static const char invoke_start[] = "<｜DSML｜invoke";
static const char invoke_end[] = "</｜DSML｜invoke>";
static const char invoke_short_start[] = "<invoke";
static const char invoke_short_end[] = "</inv>";
static const char invoke_dsml_short_end[] = "</｜DSML｜inv>";
static const char parameter_start[] = "<｜DSML｜parameter";
static const char parameter_end[] = "</｜DSML｜parameter>";
static const char parameter_short_start[] = "<parameter";
static const char parameter_short_end[] = "</parameter>";

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error && capacity) snprintf(error, capacity, "%s", message);
}

static char *copy_text(const char *text, size_t length)
{
    char *copy = (char *)malloc(length + 1u);
    if (!copy) return NULL;
    if (length) memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

static const char *find_bytes(const char *text, size_t length,
                              const char *needle, size_t needle_length)
{
    if (needle_length > length) return NULL;
    for (size_t i = 0; i <= length - needle_length; ++i)
        if (memcmp(text + i, needle, needle_length) == 0) return text + i;
    return NULL;
}

static const char *skip_space(const char *cursor, const char *end)
{
    while (cursor < end && isspace((unsigned char)*cursor)) ++cursor;
    return cursor;
}

static int append_json_key(DSV4Buffer *output, const char *name, size_t length)
{
    return dsv4_buffer_append_json_string(output, name, length) &&
           dsv4_buffer_append_string(output, ":");
}

static int valid_json_primitive(const char *text, size_t length)
{
    if ((length == 4u && memcmp(text, "true", 4u) == 0) ||
        (length == 5u && memcmp(text, "false", 5u) == 0) ||
        (length == 4u && memcmp(text, "null", 4u) == 0))
        return 1;
    if (!length || length >= 128u) return 0;
    char number[128];
    memcpy(number, text, length);
    number[length] = '\0';
    char *end = NULL;
    errno = 0;
    const double value = strtod(number, &end);
    return !errno && end != number && *end == '\0' && isfinite(value);
}

static int valid_json_value(const char *text, size_t length)
{
    while (length && isspace((unsigned char)*text)) {
        ++text;
        --length;
    }
    while (length && isspace((unsigned char)text[length - 1u])) --length;
    if (!length || text[0] == '"') return 0;
    if (text[0] != '{' && text[0] != '[')
        return valid_json_primitive(text, length);
    jsmn_parser parser;
    jsmntok_t tokens[256];
    jsmn_init(&parser);
    const int count = jsmn_parse(&parser, text, length,
                                 tokens, sizeof(tokens) / sizeof(tokens[0]));
    return count > 0 && tokens[0].start == 0 &&
           tokens[0].end == (int)length &&
           ((text[0] == '{' && tokens[0].type == JSMN_OBJECT) ||
            (text[0] == '[' && tokens[0].type == JSMN_ARRAY));
}

static int parse_attribute(const char **cursor, const char *end,
                           const char *attribute,
                           const char **value, size_t *value_length)
{
    const size_t prefix_length = strlen(attribute);
    if ((size_t)(end - *cursor) < prefix_length ||
        memcmp(*cursor, attribute, prefix_length) != 0)
        return 0;
    *cursor += prefix_length;
    const char *closing = memchr(*cursor, '"', (size_t)(end - *cursor));
    if (!closing || closing == *cursor) return 0;
    *value = *cursor;
    *value_length = (size_t)(closing - *cursor);
    *cursor = closing + 1;
    return 1;
}

static int name_is_valid(const char *name, size_t length)
{
    if (!length || length > 128u) return 0;
    for (size_t i = 0; i < length; ++i) {
        const unsigned char c = (unsigned char)name[i];
        if (!(isalnum(c) || c == '_' || c == '-' || c == '.' || c == ':'))
            return 0;
    }
    return 1;
}

static int malformed_closing_tail(const char *cursor, const char *end)
{
    static const char *allowed[] = {
        "</｜DSML｜parameter>",
        "</｜DSML｜>",
        "</｜DSML｜inv>",
        "</｜DSML｜invoke>",
        "</｜DSML｜tool>",
        "</｜DSML｜tool_calls>"
    };
    int found = 0;
    while (cursor < end) {
        cursor = skip_space(cursor, end);
        if (cursor == end) return found;
        int matched = 0;
        for (size_t i = 0; i < sizeof(allowed) / sizeof(allowed[0]); ++i) {
            const size_t length = strlen(allowed[i]);
            if ((size_t)(end - cursor) >= length &&
                memcmp(cursor, allowed[i], length) == 0) {
                cursor += length;
                matched = 1;
                found = 1;
                break;
            }
        }
        if (!matched) return 0;
    }
    return found;
}

/* Returns 2 only for a complete invocation followed solely by a narrowly
 * recognized malformed closing tail emitted by unconstrained V4 decoding. */
static int parse_invoke(const char **cursor, const char *end,
                        DSV4DsmlToolCall *call,
                        char *error, size_t error_capacity)
{
    size_t invoke_start_length = sizeof(invoke_start) - 1u;
    if ((size_t)(end - *cursor) >= invoke_start_length &&
        memcmp(*cursor, invoke_start, invoke_start_length) == 0) {
        /* Standard V4 form. */
    } else if ((size_t)(end - *cursor) >=
                   sizeof(invoke_short_start) - 1u &&
               memcmp(*cursor, invoke_short_start,
                      sizeof(invoke_short_start) - 1u) == 0) {
        invoke_start_length = sizeof(invoke_short_start) - 1u;
    } else {
        set_error(error, error_capacity, "expected a DSML invoke block");
        return 0;
    }
    *cursor += invoke_start_length;
    if (*cursor >= end || !isspace((unsigned char)**cursor)) {
        set_error(error, error_capacity, "DSML invoke is missing whitespace");
        return 0;
    }
    *cursor = skip_space(*cursor, end);
    if ((size_t)(end - *cursor) >= 6u &&
        memcmp(*cursor, "invoke", 6u) == 0 &&
        (size_t)(end - *cursor) > 6u &&
        isspace((unsigned char)(*cursor)[6])) {
        *cursor += 6u;
        *cursor = skip_space(*cursor, end);
    }
    const char *name = NULL;
    size_t name_length = 0;
    if (!parse_attribute(cursor, end, "name=\"", &name, &name_length) ||
        !name_is_valid(name, name_length)) {
        set_error(error, error_capacity, "invalid DSML tool name");
        return 0;
    }
    *cursor = skip_space(*cursor, end);
    if (*cursor >= end || *(*cursor)++ != '>') {
        set_error(error, error_capacity, "DSML invoke tag is not closed");
        return 0;
    }
    call->name = copy_text(name, name_length);
    if (!call->name) {
        set_error(error, error_capacity, "out of memory parsing DSML");
        return 0;
    }
    DSV4Buffer arguments;
    dsv4_buffer_init(&arguments);
    int ok = dsv4_buffer_append_string(&arguments, "{");
    int recovered_tail = 0;
    size_t parameter_count = 0;
    char *parameter_names[128] = {0};
    while (ok) {
        *cursor = skip_space(*cursor, end);
        if (*cursor == end) {
            set_error(error, error_capacity, "incomplete DSML invoke block");
            ok = 0;
            break;
        }
        const size_t invoke_end_length = sizeof(invoke_end) - 1u;
        if ((size_t)(end - *cursor) >= invoke_end_length &&
            memcmp(*cursor, invoke_end, invoke_end_length) == 0) {
            *cursor += invoke_end_length;
            break;
        }
        if ((size_t)(end - *cursor) >=
                sizeof(invoke_dsml_short_end) - 1u &&
            memcmp(*cursor, invoke_dsml_short_end,
                   sizeof(invoke_dsml_short_end) - 1u) == 0) {
            *cursor += sizeof(invoke_dsml_short_end) - 1u;
            break;
        }
        if ((size_t)(end - *cursor) >= sizeof(invoke_short_end) - 1u &&
            memcmp(*cursor, invoke_short_end,
                   sizeof(invoke_short_end) - 1u) == 0) {
            *cursor += sizeof(invoke_short_end) - 1u;
            break;
        }
        if (parameter_count > 0 && malformed_closing_tail(*cursor, end)) {
            *cursor = end;
            recovered_tail = 1;
            break;
        }
        if (parameter_count == 128u) {
            set_error(error, error_capacity, "too many DSML parameters");
            ok = 0;
            break;
        }
        size_t parameter_start_length = sizeof(parameter_start) - 1u;
        int short_parameter = 0;
        if (*cursor < end && **cursor == '<' &&
            (size_t)(end - *cursor - 1) >= parameter_start_length &&
            memcmp(*cursor + 1, parameter_start,
                   parameter_start_length) == 0)
            ++*cursor;
        if ((size_t)(end - *cursor) >= parameter_start_length &&
            memcmp(*cursor, parameter_start, parameter_start_length) == 0) {
            /* Standard V4 form. */
        } else if ((size_t)(end - *cursor) >=
                       sizeof(parameter_short_start) - 1u &&
                   memcmp(*cursor, parameter_short_start,
                          sizeof(parameter_short_start) - 1u) == 0) {
            parameter_start_length = sizeof(parameter_short_start) - 1u;
            short_parameter = 1;
        } else {
            set_error(error, error_capacity, "invalid DSML invoke body");
            ok = 0;
            break;
        }
        *cursor += parameter_start_length;
        if (*cursor >= end || !isspace((unsigned char)**cursor)) {
            set_error(error, error_capacity,
                      "DSML parameter is missing whitespace");
            ok = 0;
            break;
        }
        *cursor = skip_space(*cursor, end);
        const char *parameter_name = NULL;
        size_t parameter_name_length = 0;
        if (!parse_attribute(cursor, end, "name=\"", &parameter_name,
                             &parameter_name_length) ||
            !name_is_valid(parameter_name, parameter_name_length)) {
            set_error(error, error_capacity, "invalid DSML parameter name");
            ok = 0;
            break;
        }
        *cursor = skip_space(*cursor, end);
        const char *string_value = NULL;
        size_t string_length = 0;
        int string_attribute_present = 0;
        if ((size_t)(end - *cursor) >= 8u &&
            memcmp(*cursor, "string=\"", 8u) == 0) {
            string_attribute_present = parse_attribute(
                cursor, end, "string=\"", &string_value, &string_length);
            if (!string_attribute_present ||
                !((string_length == 4u &&
                   memcmp(string_value, "true", 4u) == 0) ||
                  (string_length == 5u &&
                   memcmp(string_value, "false", 5u) == 0))) {
                set_error(error, error_capacity,
                          "invalid DSML string attribute");
                ok = 0;
                break;
            }
        }
        *cursor = skip_space(*cursor, end);
        if (*cursor >= end || *(*cursor)++ != '>') {
            set_error(error, error_capacity,
                      "DSML parameter tag is not closed");
            ok = 0;
            break;
        }
        const char *closing_text = short_parameter
                                 ? parameter_short_end : parameter_end;
        const size_t closing_length = strlen(closing_text);
        const char *parameter_close = find_bytes(
            *cursor, (size_t)(end - *cursor),
            closing_text, closing_length);
        if (!parameter_close) {
            set_error(error, error_capacity, "incomplete DSML parameter");
            ok = 0;
            break;
        }
        const size_t value_length = (size_t)(parameter_close - *cursor);
        const int value_is_json = valid_json_value(*cursor, value_length);
        for (size_t i = 0; i < parameter_count; ++i) {
            if (strlen(parameter_names[i]) == parameter_name_length &&
                memcmp(parameter_names[i], parameter_name,
                       parameter_name_length) == 0) {
                set_error(error, error_capacity,
                          "duplicate DSML parameter name");
                ok = 0;
                break;
            }
        }
        if (!ok) break;
        parameter_names[parameter_count] = copy_text(
            parameter_name, parameter_name_length);
        if (!parameter_names[parameter_count]) {
            set_error(error, error_capacity, "out of memory parsing DSML");
            ok = 0;
            break;
        }
        if (parameter_count &&
            !dsv4_buffer_append_string(&arguments, ",")) ok = 0;
        if (ok) ok = append_json_key(&arguments, parameter_name,
                                     parameter_name_length);
        if (!string_attribute_present && !short_parameter && !value_is_json) {
            set_error(error, error_capacity,
                      "missing string attribute on non-JSON parameter");
            ok = 0;
        }
        if (ok && ((short_parameter && !value_is_json) ||
                   (string_attribute_present && string_length == 4u)))
            ok = dsv4_buffer_append_json_string(&arguments, *cursor,
                                                 value_length);
        else if (ok) {
            if (!value_is_json) {
                set_error(error, error_capacity,
                          "invalid JSON in non-string DSML parameter");
                ok = 0;
            } else {
                ok = dsv4_buffer_append(&arguments, *cursor, value_length);
            }
        }
        if (!ok && (!error || !error_capacity || !error[0]))
            set_error(error, error_capacity, "out of memory parsing DSML");
        ++parameter_count;
        *cursor = parameter_close + closing_length;
    }
    for (size_t i = 0; i < parameter_count; ++i) free(parameter_names[i]);
    if (ok) ok = dsv4_buffer_append_string(&arguments, "}");
    if (!ok) {
        dsv4_buffer_free(&arguments);
        free(call->name);
        call->name = NULL;
        return 0;
    }
    call->arguments = arguments.data;
    return recovered_tail ? 2 : 1;
}

void dsv4_dsml_result_init(DSV4DsmlResult *result)
{
    if (result) memset(result, 0, sizeof(*result));
}

void dsv4_dsml_result_free(DSV4DsmlResult *result)
{
    if (!result) return;
    free(result->content);
    for (size_t i = 0; i < result->call_count; ++i) {
        free(result->calls[i].name);
        free(result->calls[i].arguments);
    }
    memset(result, 0, sizeof(*result));
}

int dsv4_dsml_parse(const char *text, size_t length,
                    DSV4DsmlResult *result,
                    char *error, size_t error_capacity)
{
    if (!text || !result) return 0;
    dsv4_dsml_result_init(result);
    if (error && error_capacity) error[0] = '\0';
    const char *start = find_bytes(text, length, tool_start,
                                   sizeof(tool_start) - 1u);
    if (!start) {
        result->content = copy_text(text, length);
        if (!result->content)
            set_error(error, error_capacity, "out of memory parsing output");
        return result->content != NULL;
    }
    result->content = copy_text(text, (size_t)(start - text));
    if (!result->content) {
        set_error(error, error_capacity, "out of memory parsing output");
        return 0;
    }
    const char *cursor = start + sizeof(tool_start) - 1u;
    const char *end = text + length;
    int closed = 0;
    int recovered = 0;
    while (cursor < end) {
        cursor = skip_space(cursor, end);
        while ((size_t)(end - cursor) >= sizeof(invoke_end) - 1u &&
               memcmp(cursor, invoke_end, sizeof(invoke_end) - 1u) == 0) {
            cursor += sizeof(invoke_end) - 1u;
            cursor = skip_space(cursor, end);
        }
        if ((size_t)(end - cursor) >= sizeof(tool_end) - 1u &&
            memcmp(cursor, tool_end, sizeof(tool_end) - 1u) == 0) {
            cursor += sizeof(tool_end) - 1u;
            closed = 1;
            break;
        }
        if (result->call_count == DSV4_DSML_MAX_CALLS) {
            set_error(error, error_capacity, "too many DSML tool calls");
            goto fail;
        }
        const int invoke_status = parse_invoke(
            &cursor, end, &result->calls[result->call_count],
            error, error_capacity);
        if (!invoke_status)
            goto fail;
        ++result->call_count;
        if (invoke_status == 2) {
            recovered = 1;
            break;
        }
    }
    if ((!closed && !recovered) || result->call_count == 0) {
        set_error(error, error_capacity, "incomplete DSML tool_calls block");
        goto fail;
    }
    cursor = skip_space(cursor, end);
    if (cursor != end) {
        set_error(error, error_capacity,
                  "text after DSML tool_calls is not supported");
        goto fail;
    }
    return 1;
fail:
    dsv4_dsml_result_free(result);
    return 0;
}

void dsv4_dsml_parser_init(DSV4DsmlParser *parser)
{
    if (parser) memset(parser, 0, sizeof(*parser));
}

void dsv4_dsml_parser_free(DSV4DsmlParser *parser)
{
    if (!parser) return;
    free(parser->data);
    memset(parser, 0, sizeof(*parser));
}

int dsv4_dsml_parser_write(DSV4DsmlParser *parser,
                           const char *data, size_t length)
{
    if (!parser || (!data && length) ||
        length > SIZE_MAX - parser->length - 1u)
        return 0;
    const size_t needed = parser->length + length + 1u;
    if (needed > parser->capacity) {
        size_t capacity = parser->capacity ? parser->capacity : 256u;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = needed;
                break;
            }
            capacity *= 2u;
        }
        char *larger = (char *)realloc(parser->data, capacity);
        if (!larger) return 0;
        parser->data = larger;
        parser->capacity = capacity;
    }
    if (length) memcpy(parser->data + parser->length, data, length);
    parser->length += length;
    parser->data[parser->length] = '\0';
    return 1;
}

int dsv4_dsml_parser_finish(DSV4DsmlParser *parser,
                            DSV4DsmlResult *result,
                            char *error, size_t error_capacity)
{
    if (!parser) return 0;
    return dsv4_dsml_parse(parser->data ? parser->data : "", parser->length,
                           result, error, error_capacity);
}
