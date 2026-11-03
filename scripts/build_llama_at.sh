#!/usr/bin/env bash
# Build libllama at one llama.cpp commit and link the InferCheck harness (llamatok) against it.
#
#   scripts/build_llama_at.sh LLAMA_SRC COMMIT OUT_DIR
#
# LLAMA_SRC is a llama.cpp git clone (left untouched apart from a temporary worktree).
# The result is OUT_DIR/llamatok; an existing one is reused. Only the CPU backend is built,
# since the harness loads vocabularies only (vocab_only), so a build takes a few minutes.
set -euo pipefail

SRC=$(cd "$1" && pwd)
COMMIT=$2
OUT=$3
HERE=$(cd "$(dirname "$0")/.." && pwd)
HARNESS="$HERE/harness/llama_tok/llamatok.cpp"

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
if [ -x "$OUT/llamatok" ]; then
  echo "$OUT/llamatok"
  exit 0
fi

if ! git -C "$SRC" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  echo "fetching $COMMIT into $SRC" >&2
  git -C "$SRC" fetch -q origin "$COMMIT" || git -C "$SRC" fetch -q origin
fi

WT="$OUT/src"
git -C "$SRC" worktree remove --force "$WT" >/dev/null 2>&1 || rm -rf "$WT"
git -C "$SRC" worktree prune
git -C "$SRC" worktree add -q --detach -f "$WT" "$COMMIT"
cleanup() { git -C "$SRC" worktree remove --force "$WT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# macOS: the newest SDK can be ahead of the linker ("tapi error ... unknown architecture"). Unless INFERCHECK_SDK
# is set, use the SDK that LLAMA_SRC's own build was configured with, since that one is known to link.
if [ -z "${INFERCHECK_SDK:-}" ] && [ "$(uname)" = Darwin ] && [ -f "$SRC/build/CMakeCache.txt" ]; then
  SDK=$(sed -n 's/^CMAKE_OSX_SYSROOT:[A-Z]*=//p' "$SRC/build/CMakeCache.txt" | head -1)
  if [ -z "$SDK" ]; then
    SDK=$(grep -o '/[^;:"= ]*SDKs/MacOSX[0-9.]*\.sdk' "$SRC/build/CMakeCache.txt" | head -1)
  fi
  if [ -n "$SDK" ] && [ -d "$SDK" ]; then
    INFERCHECK_SDK=$SDK
    echo "using the macOS SDK of $SRC/build: $SDK" >&2
  fi
fi
rm -rf "$OUT/build"  # a failed configure leaves a cache pinned to the wrong SDK

if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
  JOBS=$(sysctl -n hw.ncpu)
else
  JOBS=$(nproc 2>/dev/null || echo 4)
fi

# INFERCHECK_SDK=/path/to/MacOSX<version>.sdk pins the macOS SDK (when the newest one is ahead of the linker)
cmake -S "$WT" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  ${INFERCHECK_SDK:+-DCMAKE_OSX_SYSROOT=$INFERCHECK_SDK} \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF -Wno-dev > "$OUT/cmake.log" 2>&1 || { tail -30 "$OUT/cmake.log" >&2; exit 1; }
cmake --build "$OUT/build" --target llama -j "$JOBS" > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log" >&2; exit 1; }

# shared libraries moved between build/src, build/ggml/src and build/bin over time
LIBFLAGS=()
while IFS= read -r d; do
  LIBFLAGS+=("-L$d" "-Wl,-rpath,$d")
done < <(find "$OUT/build" \( -name 'libllama.*' -o -name 'libggml*.so*' -o -name 'libggml*.dylib' \) -exec dirname {} \; | sort -u)

# on macOS, link with the SDK CMake chose (the newest SDK can be ahead of the linker)
SYSROOT=${INFERCHECK_SDK:-$(sed -n 's/^CMAKE_OSX_SYSROOT:[A-Z]*=//p' "$OUT/build/CMakeCache.txt" | head -1)}
SYSFLAGS=()
if [ -n "$SYSROOT" ]; then SYSFLAGS=(-isysroot "$SYSROOT"); fi

c++ -O2 -std=c++17 "${SYSFLAGS[@]}" "$HARNESS" -I"$WT/include" -I"$WT/ggml/include" "${LIBFLAGS[@]}" \
  -lllama -lggml -lggml-base -o "$OUT/llamatok.tmp"
mv "$OUT/llamatok.tmp" "$OUT/llamatok"
git -C "$SRC" log -1 --format='%H %cd' --date=short "$COMMIT" > "$OUT/COMMIT"
echo "$OUT/llamatok"
