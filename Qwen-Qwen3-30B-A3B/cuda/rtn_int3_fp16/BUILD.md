# Reproducing the two CUDA runtimes
These are two source builds of ONNX Runtime, not two released wheels or a
provider-library swap. Both use main base
`36f07f760dccb74a96d0a3c851163d6448bdfd21`:

| Runtime | Pinned source SHA |
| --- | --- |
| PR #33555 | `d75ac16f377c0c783d289046f5cf2efe6bb5f9bc` |
| PR #33569 | `4ba6c38384cf5b12d4aa4570cd15af2461c4d515` |

## Recorded build boundary
Evidence is in container `jiafa-dev`: both
`/tmp/int3-pr{33555,33569}-compare/logs/configure-settings.json`, their
`configure-command.sh` files, #33555's `logs/configured-CMakeCache.txt`, and
#33569's `Release/CMakeCache.txt`. #33555's actual source/build directories
were `ort-int3-pr2-pr3` and `/tmp/int3-pr2-pr3-clean/Release`, respectively;
its logs and staged package live under `/tmp/int3-pr33555-compare`.

Both configurations use Linux x86-64, Ninja, Release, CUDA 13, SM80 only,
Python bindings and a shared ORT library. Recorded compiler paths are
`/usr/bin/cc`, `/usr/bin/c++`, and `/usr/local/cuda/bin/nvcc`.
Tools inspected in `jiafa-dev`: GCC/G++ `13.3.0-6ubuntu2~24.04.1`,
NVCC `13.0.88`, CMake `4.3.2`, Ninja
`1.13.0.git.kitware.jobserver-pipe-1`, Python `3.12.3`, cuDNN `9.20.0`.
The build interpreter is `/tmp/qmoe-validation-venv/bin/python`; cuDNN is
`/usr/local/lib/python3.12/dist-packages/nvidia/cudnn`.
C, C++, and CUDA Release flags are `-O3 -DNDEBUG`; LTO is off.

Important effective CMake settings (applied equally below):
`onnxruntime_USE_FP4_QMOE=ON`, `onnxruntime_USE_FP8_QMOE=OFF`,
`onnxruntime_USE_FPA_INTB_GEMM=ON`, `onnxruntime_USE_FPA_INTB_GEMM_FULL=OFF`,
`onnxruntime_USE_FLASH_ATTENTION=OFF`, `onnxruntime_USE_LEAN_ATTENTION=OFF`,
`onnxruntime_USE_FP8_KV_CACHE=ON`, `onnxruntime_USE_CUDA_NHWC_OPS=ON`,
`onnxruntime_NVCC_THREADS=4`, `onnxruntime_FLASH_NVCC_THREADS=4`,
`onnxruntime_BUILD_UNIT_TESTS=ON`, `onnxruntime_BUILD_CUDA_TESTS=ON`.
`--skip_tests` skips execution, not compilation of configured test targets.
Training, minimal build, and CUDA EP plugin mode are off.

**Dependency exception:** the ORT snapshots pin ONNX submodule
`e1ac8e74b620bbdc7ba07f0440806bd93f651a28`, but the actual builds override
`FETCHCONTENT_SOURCE_DIR_ONNX` with ONNX source at
`ee3ccbd2b2344299d3a4506c2954a47b2181a485`. Reproducing the default submodule
pin is therefore NOT reproducing these builds.

The other 19 source overrides reuse populated dependency caches:
`ABSEIL_CPP`, `CPP_CLIENT_TELEMETRY`, `CUDNN_FRONTEND`, `CUTLASS`, `DATE`,
`DLPACK`, `EIGEN3`, `FLATBUFFERS`, `GOOGLETEST`, `GSL`, `MATSDK_CURL`,
`MATSDK_MBEDTLS`, `MP11`, `NLOHMANN_JSON`, `PROTOBUF`, `PYBIND11_PROJECT`,
`PYTORCH_CPUINFO`, `RE2`, `SAFEINT`. They are source reuse, not reuse of ORT
objects or binaries. `FETCHCONTENT_FULLY_DISCONNECTED=ON` requires these
sources to exist. Preserve their exact contents/revisions for reproduction;
the recipe below consumes the recorded mapping rather than listing 19 local
paths. Without that cache snapshot, matching only the ORT SHAs is insufficient.

## Fresh checkouts
Run on the host with Git. Choose a NEW directory on a volume visible at the
same path inside `jiafa-dev`; never detach or modify existing user worktrees.
The following commands create independent repositories, not worktrees:

```bash
set -euo pipefail
export REPRO=/datadisks/disk1/jiafa/accuracy/int3-runtime-reproduction-new
test ! -e "$REPRO"
mkdir -p "$REPRO"
for entry in \
  33555:d75ac16f377c0c783d289046f5cf2efe6bb5f9bc \
  33569:4ba6c38384cf5b12d4aa4570cd15af2461c4d515; do
  pr=${entry%%:*}; sha=${entry#*:}
  git clone --no-checkout https://github.com/microsoft/onnxruntime.git "$REPRO/src-$pr"
  git -C "$REPRO/src-$pr" fetch origin "$sha"
  git -C "$REPRO/src-$pr" checkout --detach "$sha"
  git -C "$REPRO/src-$pr" submodule update --init --recursive
done
```

## Build and stage
Run this block inside `jiafa-dev` with `REPRO` set as above. Host-initialized
submodules permit `--skip_submodule_sync` because this container lacks Git.
The same flags can be passed to each checkout's `./build.sh`; explicitly using
`tools/ci_build/build.py` selects the recorded Python interpreter. Its parser
supports `--build_wheel`, `--parallel 8`, and all flags below. This is a
conventional full build plus wheel packaging, not the historical targeted
`ninja -j8 onnxruntime onnxruntime_pybind11_state onnxruntime_providers_cuda`.
Use the same prepared Python environment (including wheel build prerequisites)
for both. No published ORT wheel is installed.

```bash
set -euo pipefail
: "${REPRO:?Set the new shared reproduction directory}"
PY=/tmp/qmoe-validation-venv/bin/python
CUDA_HOME=/usr/local/cuda
CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export CC=/usr/bin/cc CXX=/usr/bin/c++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mapfile -t deps < <("$PY" - <<'PY'
import json
from pathlib import Path
settings = json.loads(Path('/tmp/int3-pr33569-compare/logs/configure-settings.json').read_text())
for key, setting in sorted(settings.items()):
    if key.startswith('FETCHCONTENT_SOURCE_DIR_'):
        source = Path(setting['value'])
        if not source.is_dir():
            raise SystemExit(f'Missing dependency source: {source}')
        print(f'{key}={source}')
PY
)
test "${#deps[@]}" -eq 20
root_args=()
if [[ $(id -u) == 0 ]]; then root_args+=(--allow_running_as_root); fi
for pr in 33555 33569; do
  src="$REPRO/src-$pr"; build="$REPRO/build-$pr"; package="$REPRO/package-$pr"
  test ! -e "$build" && test ! -e "$package"
  "$PY" "$src/tools/ci_build/build.py" "${root_args[@]}" \
    --config Release --build_dir "$build" --use_cuda \
    --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
    --parallel 8 --skip_tests --build_shared_lib --enable_pybind \
    --build_wheel --skip_submodule_sync --cmake_generator Ninja --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=80 FETCHCONTENT_FULLY_DISCONNECTED=ON \
    onnxruntime_USE_FP4_QMOE=ON onnxruntime_USE_FP8_QMOE=OFF \
    onnxruntime_USE_FPA_INTB_GEMM=ON onnxruntime_USE_FPA_INTB_GEMM_FULL=OFF \
    onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_LEAN_ATTENTION=OFF \
    onnxruntime_USE_FP8_KV_CACHE=ON onnxruntime_USE_CUDA_NHWC_OPS=ON \
    onnxruntime_NVCC_THREADS=4 onnxruntime_FLASH_NVCC_THREADS=4 \
    onnxruntime_BUILD_UNIT_TESTS=ON onnxruntime_BUILD_CUDA_TESTS=ON "${deps[@]}"
  wheels=("$build"/Release/dist/*.whl)
  test "${#wheels[@]}" -eq 1 && test -f "${wheels[0]}"
  "$PY" -m pip install --no-deps --target "$package" "${wheels[0]}"
done
```

Each `package-33555` / `package-33569` is a complete, separate Python package
root for `run_comparison.py`. Run consumers outside the source checkout with
only the chosen root on `PYTHONPATH`. Never mix pybind, core, shared-provider,
or CUDA-provider binaries from different builds. Keep the CUDA/cuDNN library
search path above when using either root.

The measured staging instead copied Python files from each matching source
tree and symlinked its matching build's core, pybind, and provider libraries
into `python-runtime/onnxruntime/capi`; it was **not a wheel installation**.
That layout is an alternative only with the staging manifest, complete library
set, and binary hashes preserved; the wheel route does not claim identical hashes.

## Results provenance
The parent comparison summary is to add [results/build-settings.json](results/build-settings.json)
with both SHAs/base, effective configure flags, compiler/tool versions, all 20
resolved source overrides and dependency revisions/hashes, staging method,
wheel/binary hashes, and links to the original configure commands/caches.
That summary is not supplied here. These instructions assert neither
benchmark results nor byte-for-byte reproducibility; no build or benchmark was
run while writing this document.