/* SPDX-License-Identifier: Apache-2.0 */
#include "dsv4_dsml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static const char sample[] =
    "Let me check. <｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"run_command\">\n"
    "<｜DSML｜parameter name=\"command\" string=\"true\">"
    "git --version 2>&1</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "<｜DSML｜invoke name=\"plan_trip\">\n"
    "<｜DSML｜parameter name=\"days\" string=\"false\">3"
    "</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"flexible\" string=\"false\">false"
    "</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"cities\" string=\"false\">"
    "[\"北京\",\"上海\"]</｜DSML｜parameter>\n"
    "<｜DSML｜parameter name=\"notes\" string=\"true\">靠窗座位"
    "</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>";

static int verify_result(const DSV4DsmlResult *result)
{
    CHECK(strcmp(result->content, "Let me check. ") == 0);
    CHECK(result->call_count == 2u);
    CHECK(strcmp(result->calls[0].name, "run_command") == 0);
    CHECK(strcmp(result->calls[0].arguments,
                 "{\"command\":\"git --version 2>&1\"}") == 0);
    CHECK(strcmp(result->calls[1].name, "plan_trip") == 0);
    CHECK(strcmp(result->calls[1].arguments,
        "{\"days\":3,\"flexible\":false,\"cities\":[\"北京\",\"上海\"],"
        "\"notes\":\"靠窗座位\"}") == 0);
    return 0;
}

static int expect_error(const char *text, const char *needle)
{
    DSV4DsmlResult result;
    char error[256] = {0};
    const int ok = dsv4_dsml_parse(text, strlen(text), &result,
                                   error, sizeof(error));
    if (ok) dsv4_dsml_result_free(&result);
    return !ok && strstr(error, needle) != NULL;
}

int main(void)
{
    DSV4DsmlResult result;
    char error[256] = {0};
    CHECK(dsv4_dsml_parse(sample, sizeof(sample) - 1u,
                          &result, error, sizeof(error)));
    CHECK(verify_result(&result) == 0);
    dsv4_dsml_result_free(&result);
    static const char sdk_recovery[] =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke invoke name=\"add_numbers\">\n"
        "<｜DSML｜parameter name=\"a\">17</｜DSML｜parameter>\n"
        "<parameter name=\"b\">25</parameter>\n"
        "</｜DSML｜inv>\n</｜DSML｜tool_calls>";
    CHECK(dsv4_dsml_parse(sdk_recovery, sizeof(sdk_recovery) - 1u,
                          &result, error, sizeof(error)));
    CHECK(result.call_count == 1u);
    CHECK(strcmp(result.calls[0].arguments,
                 "{\"a\":17,\"b\":25}") == 0);
    dsv4_dsml_result_free(&result);
    static const char numeric_recovery[] =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"add_numbers\">\n"
        "<<｜DSML｜parameter name=\"a\">17</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"b\">25</｜DSML｜parameter>\n"
        "</｜DSML｜>\n</｜DSML｜inv>";
    CHECK(dsv4_dsml_parse(numeric_recovery,
                          sizeof(numeric_recovery) - 1u,
                          &result, error, sizeof(error)));
    CHECK(result.call_count == 1u);
    CHECK(strcmp(result.calls[0].arguments,
                 "{\"a\":17,\"b\":25}") == 0);
    dsv4_dsml_result_free(&result);
    CHECK(expect_error(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">"
        "<｜DSML｜parameter name=\"q\">plain text</｜DSML｜parameter>"
        "</｜DSML｜tool>",
        "missing string"));
    static const char parallel_recovery[] =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"get_weather\">\n"
        "<｜DSML｜parameter name=\"city\" string=\"true\">Beijing"
        "</｜DSML｜parameter>\n</｜DSML｜inv>\n"
        "<invoke name=\"get_weather\">\n"
        "<parameter name=\"city\">Shanghai</parameter>\n</inv>\n"
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    CHECK(dsv4_dsml_parse(parallel_recovery,
                          sizeof(parallel_recovery) - 1u,
                          &result, error, sizeof(error)));
    CHECK(result.call_count == 2u);
    CHECK(strcmp(result.calls[0].arguments,
                 "{\"city\":\"Beijing\"}") == 0);
    CHECK(strcmp(result.calls[1].arguments,
                 "{\"city\":\"Shanghai\"}") == 0);
    dsv4_dsml_result_free(&result);

    for (size_t chunk = 1u; chunk <= sizeof(sample) - 1u; ++chunk) {
        DSV4DsmlParser parser;
        dsv4_dsml_parser_init(&parser);
        for (size_t offset = 0; offset < sizeof(sample) - 1u; offset += chunk) {
            size_t length = sizeof(sample) - 1u - offset;
            if (length > chunk) length = chunk;
            CHECK(dsv4_dsml_parser_write(&parser, sample + offset, length));
        }
        CHECK(dsv4_dsml_parser_finish(&parser, &result,
                                       error, sizeof(error)));
        CHECK(verify_result(&result) == 0);
        dsv4_dsml_result_free(&result);
        dsv4_dsml_parser_free(&parser);
    }

    CHECK(dsv4_dsml_parse("ordinary answer", 15u,
                          &result, error, sizeof(error)));
    CHECK(strcmp(result.content, "ordinary answer") == 0);
    CHECK(result.call_count == 0u);
    dsv4_dsml_result_free(&result);

    CHECK(expect_error(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">",
        "incomplete"));
    CHECK(expect_error(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">"
        "<｜DSML｜parameter name=\"n\" string=\"false\">not-json"
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        "invalid JSON"));
    CHECK(expect_error(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">"
        "<｜DSML｜parameter name=\"n\" string=\"true\">1"
        "</｜DSML｜parameter><｜DSML｜parameter name=\"n\" string=\"true\">2"
        "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
        "duplicate"));
    static const char recoverable[] =
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"get_weather\">\n"
        "<｜DSML｜parameter name=\"city\" string=\"true\">Beijing"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜tool>\n</｜DSML｜tool>";
    CHECK(dsv4_dsml_parse(recoverable, sizeof(recoverable) - 1u,
                          &result, error, sizeof(error)));
    CHECK(result.call_count == 1u);
    CHECK(strcmp(result.calls[0].name, "get_weather") == 0);
    CHECK(strcmp(result.calls[0].arguments,
                 "{\"city\":\"Beijing\"}") == 0);
    dsv4_dsml_result_free(&result);
    CHECK(expect_error(
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">"
        "<｜DSML｜parameter name=\"n\" string=\"true\">1"
        "</｜DSML｜parameter>unexpected</｜DSML｜tool>",
        "invalid DSML"));
    CHECK(expect_error(
        "<｜DSML｜function_calls><｜DSML｜invoke name=\"x\">"
        "</｜DSML｜invoke></｜DSML｜function_calls>",
        "") == 0);

    puts("dsv4 DSML tests passed");
    return 0;
}
