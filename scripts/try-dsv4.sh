#!/usr/bin/env bash
# try-dsv4.sh - run bin/dsv4 with a memory plan sized to THIS machine.
#
#   scripts/try-dsv4.sh                    # resident interactive chat
#   scripts/try-dsv4.sh "your prompt here" # one-shot response
#   scripts/try-dsv4.sh --server 8080      # resident local HTTP API
#
# Detects usable RAM and logical CPU count. The default total-memory plan uses
# the smaller of 2/3 total RAM and 3/4 currently available RAM, rounded down,
# and OpenMP is capped at twelve threads; these GEMVs saturate laptop memory
# bandwidth before all
# logical CPUs are useful on larger machines.
# The values are detected at runtime, never hard-coded for one developer's box.
# Every knob can be overridden through the environment:
#
#   DSV4_MODEL_DIR   checkpoint directory (default $HOME/model/DeepSeek-V4-Flash-0731)
#   DSV4_MEMORY_GIB  total budget (GiB)
#   DSV4_THREADS     OpenMP threads
#   DSV4_CONTEXT     context length
#   DSV4_CACHE_GIB   advanced: expert cache only (mutually exclusive with MEMORY)
#   OMP_WAIT_POLICY  OpenMP worker wait policy (default PASSIVE for laptops)
#   DSV4_FULL_CHECK=1  run the download + doctor first
set -euo pipefail

export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT_DIR}/bin/dsv4"

if [ ! -x "$BIN" ]; then
    echo "[try-dsv4] bin/dsv4 not built; run 'make -j' in ${ROOT_DIR} first" >&2
    exit 1
fi

MODEL_DIR="${DSV4_MODEL_DIR:-${HOME}/model/DeepSeek-V4-Flash-0731}"

case "$(uname -r 2>/dev/null || true)" in
    *[Mm]icrosoft*)
        case "$MODEL_DIR" in
            /mnt/[a-zA-Z]/*)
                echo "[try-dsv4] refusing slow WSL2 DrvFS model path: $MODEL_DIR" >&2
                echo "[try-dsv4] download or place the model under $HOME/model/DeepSeek-V4-Flash-0731" >&2
                exit 1
                ;;
        esac
        ;;
esac
if [ ! -d "$MODEL_DIR" ]; then
    echo "[try-dsv4] model dir not found: $MODEL_DIR" >&2
    echo "[try-dsv4] run scripts/download-dsv4.sh $MODEL_DIR first" >&2
    exit 1
fi

# ---- detect RAM ----
if [ -n "${DSV4_MEMORY_GIB:-}" ]; then
    MEM_GIB="$DSV4_MEMORY_GIB"
elif [ -n "${DSV4_CACHE_GIB:-}" ]; then
    MEM_GIB=""   # advanced path handled below
else
    case "$(uname -s)" in
        Darwin)
            RAM_BYTES=$(sysctl -n hw.memsize)
            ;;
        *)
            RAM_KIB=$(awk '/MemTotal/ {print $2; exit}' /proc/meminfo)
            RAM_BYTES=$(( RAM_KIB * 1024 ))
            AVAILABLE_KIB=$(awk '/MemAvailable/ {print $2; exit}' /proc/meminfo)
            ;;
    esac
    GIB=$(( 1 << 30 ))
    RAM_GIB=$(( RAM_BYTES / GIB ))
    MEM_GIB=$(( RAM_BYTES * 2 / 3 / GIB ))
    if [ -n "${AVAILABLE_KIB:-}" ]; then
        AVAILABLE_GIB_PLAN=$(( AVAILABLE_KIB * 1024 * 3 / 4 / GIB ))
        if [ "$AVAILABLE_GIB_PLAN" -lt "$MEM_GIB" ]; then MEM_GIB="$AVAILABLE_GIB_PLAN"; fi
    fi
    if [ "$MEM_GIB" -lt 2 ]; then MEM_GIB=2; fi
    echo "[try-dsv4] detected ${RAM_GIB}+ GiB usable RAM -> ${MEM_GIB} GiB plan" >&2
fi

# ---- detect threads ----
if [ -n "${DSV4_THREADS:-}" ]; then
    THREADS="$DSV4_THREADS"
else
    case "$(uname -s)" in
        Darwin) THREADS=$(sysctl -n hw.ncpu) ;;
        *) THREADS=$(nproc) ;;
    esac
fi
if [ "$THREADS" -gt 12 ] && [ -z "${DSV4_THREADS:-}" ]; then THREADS=12; fi

# ---- context ----
# Leave context selection to the CLI's memory-aware policy unless explicitly
# overridden. This keeps the normal user-facing control surface to memory GiB.
CONTEXT="${DSV4_CONTEXT:-}"

# ---- full check ----
if [ "${DSV4_FULL_CHECK:-0}" = "1" ]; then
    echo "[try-dsv4] running download + doctor" >&2
    "${SCRIPT_DIR}/download-dsv4.sh" "$MODEL_DIR"
fi

ARGS=()
if [ -n "${DSV4_CACHE_GIB:-}" ]; then
    ARGS+=(--cache-gib "$DSV4_CACHE_GIB")
else
    ARGS+=(--memory-gib "$MEM_GIB")
fi
ARGS+=(--threads "$THREADS")
if [ -n "$CONTEXT" ]; then ARGS+=(--context "$CONTEXT"); fi

if [ "${1:-}" = "--server" ]; then
    if [ "$#" -lt 2 ]; then
        echo "[try-dsv4] --server needs a port" >&2
        exit 2
    fi
    ARGS+=(--server "$2")
    shift 2
    ARGS+=("$@")
elif [ "$#" -ge 1 ]; then
    ARGS+=(--prompt "$1")
else
    ARGS+=(--interactive)
fi

echo "[try-dsv4] bin/dsv4 --model $MODEL_DIR ${ARGS[*]}" >&2
exec "$BIN" --model "$MODEL_DIR" "${ARGS[@]}"
