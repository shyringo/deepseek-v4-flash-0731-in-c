# deepseek-v4-flash-0731-in-c - DeepSeek-V4-Flash-0731 CPU inference engine.
#
# The GNUmakefile is the reference build and what CI uses for the full matrix.
# This exists for IDE integration and for platforms where GNU make is awkward.
# Both produce the same bin/dsv4 binary.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.16)

project(deepseek-v4-flash-0731-in-c
        VERSION 0.3.1
        DESCRIPTION "Run DeepSeek-V4-Flash-0731 on CPU from native FP8/FP4 weights"
        LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -march=native is a real win on the expert matmuls but produces a binary tied to
# the build machine. Off by default so a packaged build is portable.
option(DSV4_NATIVE_ARCH "Optimise for the building CPU (-march=native)" OFF)
option(DSV4_SANITIZE    "Build with ASan + UBSan"                        OFF)

find_package(OpenMP)
find_package(Threads REQUIRED)

add_library(dsv4_flags INTERFACE)
target_compile_options(dsv4_flags INTERFACE
  -Wall -Wextra
  # Not cosmetic: weight pointers are `const void *`, and pointer arithmetic on void is a
  # GNU extension that strides by one byte. Without this the mistake compiles silently.
  -Wpointer-arith
  -Wshadow -Wvla
  -Wno-unused-parameter
  # Keeps floating-point results reproducible across compilers by disabling automatic FMA
  # contraction; the op tests compare against a reference at a fixed tolerance.
  -ffp-contract=off)

if(DSV4_NATIVE_ARCH)
  target_compile_options(dsv4_flags INTERFACE -march=native)
endif()

if(DSV4_SANITIZE)
  target_compile_options(dsv4_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(dsv4_flags INTERFACE -fsanitize=address,undefined)
endif()

# --------------------------------------------------------------------------- engine --
add_library(dsv4 STATIC
  src/dsv4/dsv4_config.c
  src/dsv4/dsv4_tensor.c
  src/dsv4/dsv4_ops.c
  src/dsv4/dsv4_model.c
  src/dsv4/dsv4_prompt.c
  src/io/k3_st.c)

target_include_directories(dsv4 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv4
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io)

target_link_libraries(dsv4 PUBLIC dsv4_flags m Threads::Threads)
if(OpenMP_C_FOUND)
  target_link_libraries(dsv4 PUBLIC OpenMP::OpenMP_C)
endif()

# ------------------------------------------------------------------------------ cli --
add_executable(dsv4_cli src/cli/dsv4_run.c src/cli/dsv4_http.c
  src/cli/dsv4_dsml.c)
set_target_properties(dsv4_cli PROPERTIES OUTPUT_NAME dsv4)
target_include_directories(dsv4_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cli)
target_link_libraries(dsv4_cli PRIVATE dsv4)

# ---------------------------------------------------------------------------- tests --
# Every test below runs WITHOUT model weights (the tiny-model graph uses synthetic
# fixtures committed under tests/fixtures/tiny_dsv4). The 166.9 GB checkpoint is never
# needed for correctness.
enable_testing()

function(dsv4_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE dsv4)
endfunction()

dsv4_add_test(test_dsv4_config tests/unit/test_dsv4_config.c)
dsv4_add_test(test_dsv4_ops    tests/unit/test_dsv4_ops.c)
dsv4_add_test(test_dsv4_prompt tests/unit/test_dsv4_prompt.c)
dsv4_add_test(test_dsv4_st     tests/unit/test_dsv4_st.c)
dsv4_add_test(test_dsv4_tok    tests/unit/test_dsv4_tok.c)
dsv4_add_test(test_dsv4_model  tests/unit/test_dsv4_model.c)
add_executable(test_dsv4_http tests/unit/test_dsv4_http.c src/cli/dsv4_http.c)
target_include_directories(test_dsv4_http PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cli)
target_link_libraries(test_dsv4_http PRIVATE dsv4)
add_executable(test_dsv4_dsml tests/unit/test_dsv4_dsml.c
  src/cli/dsv4_dsml.c src/cli/dsv4_http.c)
target_include_directories(test_dsv4_dsml PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cli)
target_link_libraries(test_dsv4_dsml PRIVATE dsv4)

set(FIX ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set(MODEL ${CMAKE_CURRENT_SOURCE_DIR}/model/DeepSeek-V4-Flash-0731)

add_test(NAME dsv4_config  COMMAND test_dsv4_config ${FIX}/dsv4_config.json)
add_test(NAME dsv4_ops     COMMAND test_dsv4_ops)
add_test(NAME dsv4_prompt  COMMAND test_dsv4_prompt)
add_test(NAME dsv4_st      COMMAND test_dsv4_st ${FIX}/st
         ${CMAKE_CURRENT_BINARY_DIR}/st_index.json)
add_test(NAME dsv4_model   COMMAND test_dsv4_model ${FIX}/tiny_dsv4)
add_test(NAME dsv4_http    COMMAND test_dsv4_http)
add_test(NAME dsv4_dsml    COMMAND test_dsv4_dsml)

# The tokenizer test needs the released vocabulary (6 MB), which is not in this
# repository. Registered UNCONDITIONALLY so ctest reports Skipped rather than
# omitting it; the wrapper resolves DSV4_MODEL_DIR at test time.
add_test(NAME dsv4_tokenizer
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_TOK=$<TARGET_FILE:test_dsv4_tok>
                 -DINPUT=${MODEL}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tok_test.cmake)
set_tests_properties(dsv4_tokenizer PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")

add_executable(bench_dsv4_kernels benchmarks/bench_dsv4_kernels.c)
target_link_libraries(bench_dsv4_kernels PRIVATE dsv4)

install(TARGETS dsv4_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/dsv4 DESTINATION include)
# dsv4.h includes only stdint/stddef; no vendored header leaks into the public API.
