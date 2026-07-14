# Road perception model

`yolox_nano.onnx` is the official YOLOX-Nano 416x416 ONNX export from
Megvii's YOLOX 0.1.1rc0 release.

- SHA-256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- Source: `https://github.com/Megvii-BaseDetection/YOLOX`
- License: Apache License 2.0

The road observer only uses COCO person, bicycle, and traffic-light classes.
Model detections are filtered by confidence, position, and temporal
confirmation before they can become an advisory voice prompt.
