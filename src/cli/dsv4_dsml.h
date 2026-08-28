/* SPDX-License-Identifier: Apache-2.0 */
#ifndef DSV4_DSML_H
#define DSV4_DSML_H

#include <stddef.h>

#define DSV4_DSML_MAX_CALLS 16u

typedef struct {
    char *name;
    char *arguments;
} DSV4DsmlToolCall;

typedef struct {
    char *content;
    DSV4DsmlToolCall calls[DSV4_DSML_MAX_CALLS];
    size_t call_count;
} DSV4DsmlResult;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} DSV4DsmlParser;

void dsv4_dsml_result_init(DSV4DsmlResult *result);
void dsv4_dsml_result_free(DSV4DsmlResult *result);

void dsv4_dsml_parser_init(DSV4DsmlParser *parser);
void dsv4_dsml_parser_free(DSV4DsmlParser *parser);
int dsv4_dsml_parser_write(DSV4DsmlParser *parser,
                           const char *data, size_t length);
int dsv4_dsml_parser_finish(DSV4DsmlParser *parser,
                            DSV4DsmlResult *result,
                            char *error, size_t error_capacity);

int dsv4_dsml_parse(const char *text, size_t length,
                    DSV4DsmlResult *result,
                    char *error, size_t error_capacity);

#endif
