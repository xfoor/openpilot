# Road perception model

`yolox_nano.onnx` uses the weights and graph from the official YOLOX-Nano
ONNX export in Megvii's YOLOX 0.1.1rc0 release. Its fully convolutional input
and declared output dimensions are set to 320x320 and 2100 rows to meet the
comma 4 latency budget.

- Original 416x416 SHA-256:
  `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- Packaged 320x320 SHA-256:
  `12013ea6a069abb31f7391e0f7186414bc5eae17239ff06a5a3e1e517758b9dd`
- Compiled QCOM SHA-256:
  `c7e07cdfa1f850c07b473c53395b6ee2c087f5d85873c9dd857e003de1d81a77`
- Source: `https://github.com/Megvii-BaseDetection/YOLOX`
- License: Apache License 2.0

`yolox_nano_tinygrad.pkl` is a trusted, comma-4 QCOM TinyJit compiled artifact
of that 320x320 graph. It was produced on the target using the same tinygrad
flags as openpilot's own model builds:

```bash
DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 OPENPILOT_HACKS=1 \
  python tinygrad_repo/examples/openpilot/compile3.py \
  selfdrive/road_observer/models/yolox_nano.onnx \
  selfdrive/road_observer/models/yolox_nano_tinygrad.pkl
```

The road observer only uses COCO person, bicycle, and traffic-light classes.
Model detections are filtered by confidence, position, and temporal
confirmation before they can become an advisory voice prompt.
