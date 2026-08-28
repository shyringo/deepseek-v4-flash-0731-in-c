# GNUmakefile - DeepSeek-V4-Flash-0731 inference engine (pure C99 + OpenMP).
#
#   make -j               build bin/dsv4 (default, OpenMP when available)
#   make test             every weightless test
#   make strict           -Werror build + tests
#   make portable         no -march/-mcpu=native
#   make asan             address + UB sanitizers, OpenMP off
#   make bench-dsv4       kernel microbenchmarks
#   make c99              plain C99, no OpenMP, no native arch
#   make clean
#
# PLATFORMS. Linux/x86-64 is the reference; macOS/arm64 needs Homebrew's libomp
# (brew install libomp) because Apple Clang ships no OpenMP runtime. The
# platform block detects and wires it up without invoking `brew --prefix`.

CC       ?= cc
PYTHON   ?= python3
BUILD    ?= build
BIN      ?= bin
PREFIX   ?= /usr/local

# ---------------------------------------------------------------- platform --
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
  ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
    $(warning libomp not found at $(OMP_PREFIX). Install it with `brew install libomp`,)
    $(warning or point the build at another copy with `make OMP_PREFIX=/path/to/libomp`.)
  endif
else
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

WARN     := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter
THREAD_CFLAGS  ?= -pthread
THREAD_LDFLAGS ?= -pthread
CFLAGS   ?= -O3 -std=c99 $(WARN) $(ARCH) $(OMP_CFLAGS) $(THREAD_CFLAGS) -ffp-contract=off
LDFLAGS  ?= -lm $(OMP_LDFLAGS) $(THREAD_LDFLAGS)

INCLUDES := -Iinclude -Iinclude/dsv4 -Ithird_party -Isrc/dsv4 -Isrc/io -Isrc/cli

# ------------------------------------------------------------------ files --
ENGINE_SRC := src/dsv4/dsv4_config.c src/dsv4/dsv4_tensor.c \
              src/dsv4/dsv4_ops.c src/dsv4/dsv4_model.c \
              src/dsv4/dsv4_prompt.c \
              src/io/k3_st.c
ENGINE_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ENGINE_SRC))
HTTP_OBJ := $(BUILD)/src/cli/dsv4_http.o
DSML_OBJ := $(BUILD)/src/cli/dsv4_dsml.o

CLI_SRC := src/cli/dsv4_run.c
CLI_BIN := $(BIN)/dsv4

UNIT_TESTS := test_dsv4_config test_dsv4_ops test_dsv4_prompt test_dsv4_st \
              test_dsv4_tok test_dsv4_model test_dsv4_http test_dsv4_dsml
TEST_BINS  := $(addprefix $(BIN)/,$(UNIT_TESTS))

FIXTURES ?= tests/fixtures
MODEL    ?= model/DeepSeek-V4-Flash-0731

# ----------------------------------------------------------------- targets --
.PHONY: all test strict portable asan c99 bench-dsv4 clean

all: $(CLI_BIN)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) $(HTTP_OBJ) $(DSML_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) $(HTTP_OBJ) $(DSML_OBJ) -o $@ $(LDFLAGS)

$(BIN):
	@mkdir -p $(BIN)

$(BIN)/test_dsv4_config: tests/unit/test_dsv4_config.c $(BUILD)/src/dsv4/dsv4_config.o \
                        $(BUILD)/src/dsv4/dsv4_tensor.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_ops: tests/unit/test_dsv4_ops.c $(BUILD)/src/dsv4/dsv4_ops.o \
                      $(BUILD)/src/dsv4/dsv4_config.o $(BUILD)/src/dsv4/dsv4_tensor.o \
                      $(BUILD)/src/dsv4/dsv4_model.o $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_prompt: tests/unit/test_dsv4_prompt.c $(BUILD)/src/dsv4/dsv4_prompt.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_st: tests/unit/test_dsv4_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_tok: tests/unit/test_dsv4_tok.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $< -o $@

$(BIN)/dsv4_tok_cli: tools/dsv4_tok_cli.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $< -o $@

$(BIN)/test_dsv4_model: tests/unit/test_dsv4_model.c $(BUILD)/src/dsv4/dsv4_ops.o \
                        $(BUILD)/src/dsv4/dsv4_config.o $(BUILD)/src/dsv4/dsv4_tensor.o \
                        $(BUILD)/src/dsv4/dsv4_model.o $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_http: tests/unit/test_dsv4_http.c $(HTTP_OBJ) \
                       $(BUILD)/src/dsv4/dsv4_prompt.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_dsv4_dsml: tests/unit/test_dsv4_dsml.c $(DSML_OBJ) $(HTTP_OBJ) \
                       $(BUILD)/src/dsv4/dsv4_prompt.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/bench_dsv4_kernels: benchmarks/bench_dsv4_kernels.c $(BUILD)/src/dsv4/dsv4_ops.o \
                          $(BUILD)/src/dsv4/dsv4_model.o $(BUILD)/src/dsv4/dsv4_config.o \
                          $(BUILD)/src/dsv4/dsv4_tensor.o $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## test: everything that needs no model weights
test: $(TEST_BINS)
	@echo "== config reader ==";       ./$(BIN)/test_dsv4_config $(FIXTURES)/dsv4_config.json
	@echo "== op kernels ==";          ./$(BIN)/test_dsv4_ops
	@echo "== prompt formatter ==";    ./$(BIN)/test_dsv4_prompt
	@echo "== safetensors reader ==";  ./$(BIN)/test_dsv4_st $(FIXTURES)/st $(BUILD)/st_index.json
	@echo "== tokenizer ==";           ./$(BIN)/test_dsv4_tok $(MODEL)
	@echo "== tiny-model graph ==";    ./$(BIN)/test_dsv4_model $(FIXTURES)/tiny_dsv4
	@echo "== HTTP/JSON ==";           ./$(BIN)/test_dsv4_http
	@echo "== DSML tools ==";          ./$(BIN)/test_dsv4_dsml
	@echo
	@echo "ALL WEIGHTLESS TESTS PASSED"

## strict: -Werror build of everything the tests link
strict:
	$(MAKE) clean
	$(MAKE) CFLAGS="-O3 -std=c99 $(WARN) -Werror $(ARCH) $(OMP_CFLAGS) $(THREAD_CFLAGS) -ffp-contract=off" test

## portable: no -march/-mcpu=native
portable:
	$(MAKE) ARCH= test

## c99: plain C99, no OpenMP, no native arch
c99:
	$(MAKE) ARCH= OMP_CFLAGS= OMP_LDFLAGS= CFLAGS="-O2 -std=c99 $(WARN) $(THREAD_CFLAGS) -ffp-contract=off" test

## asan: address + UB sanitizers, OpenMP off
asan:
	$(MAKE) ARCH= OMP_CFLAGS= OMP_LDFLAGS= \
	    CFLAGS="-O1 -g -std=c99 $(WARN) $(THREAD_CFLAGS) -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer" \
	    LDFLAGS="-lm $(THREAD_LDFLAGS) -fsanitize=address,undefined" \
	    BUILD=build/asan BIN=build/asan/bin test

## bench-dsv4: kernel microbenchmarks
bench-dsv4: $(BIN)/bench_dsv4_kernels
	./$(BIN)/bench_dsv4_kernels

clean:
	rm -rf $(BUILD) $(BIN)

help:
	@echo "targets: all test strict portable c99 asan bench-dsv4 clean"
