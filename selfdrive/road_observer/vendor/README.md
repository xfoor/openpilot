# Road observer runtime

The `onnxruntime` directory is the minimal Python 3.12 ARM64 CPU runtime needed
by the comma 4 speed-sign detector. It is taken without modification from the
official `onnxruntime` 1.29.0 manylinux ARM64 wheel.

Only inference bindings, their Python loader, and license notices are retained.
The package is MIT licensed; see `onnxruntime/LICENSE` and
`onnxruntime/ThirdPartyNotices.txt`.

Packaged binary SHA-256 values:

- `onnxruntime_pybind11_state.cpython-312-aarch64-linux-gnu.so`:
  `03c666fb98294529206d599f602c6786f2ee2503462c4cceaa4a9e4033119499`
- `libonnxruntime_providers_shared.so`:
  `3b6be288fbfb7dff8770d08a23defdde18e8f7e0f5a2b344a0e5e238c999ea88`

The runtime is deliberately scoped to the speed-sign models. The primary
openpilot driving model and the road observer's YOLOX model continue to use the
Qualcomm backend.
