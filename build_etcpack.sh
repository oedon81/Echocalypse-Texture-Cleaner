#!/usr/bin/env bash
# Builds native/libetcpack.so: the game's ETC2 encoder (Ericsson ETCPACK v2.74) as a shared library.
# ETCPACK is NOT part of this repository (its licence does not allow that); this script downloads it from Ericsson's
# GitHub, applies a one-line memory-leak fix, and compiles it together with native/etcpack_lib.cxx.
# Needs git and a C++ compiler: g++ or clang++, or no system compiler at all via  pip install ziglang.
set -euo pipefail
cd "$(dirname "$0")/native"
[ -d ETCPACK ] || git clone --depth 1 https://github.com/Ericsson/ETCPACK
python3 - <<'PY'
p = 'ETCPACK/source/etcpack.cxx'; b = open(p, 'rb').read()
old = b'\tfree(D_matrix->data); free(D_matrix);'; new = b'\tfree(b_vector.data); free(D_matrix->data); free(D_matrix);'
if new not in b:
    i = b.rfind(old); assert b.count(old) == 3 and i > 0, 'unexpected ETCPACK source; leak fix not applied'
    open(p, 'wb').write(b[:i] + new + b[i + len(old):]); print('leak fix applied (compressBlockPlanar57 never freed b_vector)')
PY
if command -v g++ >/dev/null; then CXX="g++"; elif command -v clang++ >/dev/null; then CXX="clang++"; else CXX="${PYTHON:-python3} -m ziglang c++"; fi
echo "compiling with: $CXX"
$CXX -O2 -w -std=gnu++03 -fpermissive -shared -fPIC etcpack_lib.cxx ETCPACK/source/etcdec.cxx ETCPACK/source/image.cxx -o libetcpack.so
echo "built $(pwd)/libetcpack.so"
