#!/usr/bin/env bash
# Build the LOCAL PTCG engine (ptcg_engine/ptcgProgram 22/, gitignored) into a
# parallel cg-lib package dir that swaps in via CG_LIB_PATH:
#
#     bash utils/build_libcg.sh
#     CG_LIB_PATH="$PWD/ptcg_engine/build/cg-lib" python utils/run_smoketest_battle.py
#
# cg/sim.py loads libcg.so from its own package directory, so we mirror the
# vendored wrapper .py files (symlinks) next to the locally built .so.
#
# LOCAL TOOL ONLY — never ship this build in a Kaggle submission (the bundle
# must use the stock cg/; see docs/reports/ptcg_engine_report.md for license).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/ptcg_engine/ptcgProgram 22"
OUT="$ROOT/ptcg_engine/build/cg-lib/cg"

[ -f "$SRC/Export.cpp" ] || { echo "engine source not found at $SRC" >&2; exit 1; }
mkdir -p "$OUT"

# -static-libstdc++: conda's Python ships an older libstdc++ than system g++.
# --exclude-libs,ALL keeps those statically-linked libstdc++ symbols hidden so
# they bind locally; otherwise a process that already loaded conda's older
# libstdc++ (e.g. via torch) interposes it into our code and crashes.
g++ -std=c++20 -O2 -fPIC -shared -DNDEBUG -static-libstdc++ -static-libgcc \
    -Wl,--exclude-libs,ALL "$SRC/Export.cpp" -o "$OUT/libcg.so"

for f in "$ROOT"/vendor/cg-lib/cg/*.py; do
    ln -sfn "$f" "$OUT/$(basename "$f")"
done

echo "built $OUT/libcg.so"
echo "use it with: CG_LIB_PATH=\"$ROOT/ptcg_engine/build/cg-lib\""
