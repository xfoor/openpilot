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

## Speed-limit model

`speed_sign_yolov8n.onnx` is a static raw-head ONNX export of the nano
detector from `Ayaan-Ali-Khan/Traffic-Signs-Detection`. It locates candidate
speed signs but does not decide the number. `speed_sign_classifier.onnx` is a
static export of the 197k-parameter GTSRB classifier from
`Ayaan-Ali-Khan/GTSRB`; it reads each detector crop and rejects non-speed
classes.

- Original detector PyTorch SHA-256:
  `49880ef1f751ebb714247ade08cdcb1d1772a55e3a5d7d43235fc937055e0c0b`
- Packaged detector ONNX SHA-256:
  `24f9adcba720df18048bf5a2611e68f3a394806ccebe6c9fabbeac7c3eef8237`
- Original classifier H5 SHA-256:
  `376f99703f76eb981d2934bffa705f91380a7acf33401835c087b87730109673`
- Packaged classifier ONNX SHA-256:
  `890426119a7afeee4e5a4f790756973d6e5abe26b10afd73cda859842cc149e8`
- Detector source:
  `https://github.com/Ayaan-Ali-Khan/Traffic-Signs-Detection`
- Classifier source: `https://github.com/Ayaan-Ali-Khan/GTSRB`
- Source repositories license: MIT

The model is not trusted from a single image. A spatial track, temporal vote,
minimum confidence, and approaching-sign size growth are required before an
advisory can be emitted. See `docs/ROAD_OBSERVER.md` for replay limitations and
the no-actuation boundary.

Bounding-box distribution decoding, candidate suppression, crop resizing, and
detector/classifier confidence fusion run in NumPy after inference. Keeping
detection-head decoding out of the graph avoids a Qualcomm compiler failure in
the exported YOLOv8 head. Both sign models run in the vendored ONNX Runtime CPU
backend, so they do not occupy the Qualcomm accelerator used by openpilot's
driving model. The detector is limited to two CPU threads; its median inference
time on the comma 4 is about 162 ms. The classifier takes about 4 ms. The
detector uses a 256x256 direct NV12 crop of the right roadside to preserve sign
pixels without spending the device budget on the sky, hood, or far-left scene.
