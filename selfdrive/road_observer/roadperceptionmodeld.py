#!/usr/bin/env python3
import json
import time
from pathlib import Path

import numpy as np

from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.road_observer.perception import (
  MODEL_SIZE,
  PERCEPTION_EVENT_PARAM,
  SceneEvent,
  SceneInterpreter,
  decode_yolox,
  preprocess_nv12,
)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano.onnx"
INFERENCE_INTERVAL = 0.5


class YoloXDetector:
  def __init__(self, model_path: Path = MODEL_PATH, device: str = "QCOM"):
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.tensor import Tensor

    self.Tensor = Tensor
    self.device = device
    self.runner = OnnxRunner(str(model_path))

  def infer(self, model_input: np.ndarray):
    tensor = self.Tensor(model_input[np.newaxis], device=self.device)
    outputs = self.runner({"images": tensor})
    return next(iter(outputs.values())).numpy()


def serialize_observation(frame_id: int, execution_time: float, observation,
                          detections, voice_enabled: bool) -> bytes:
  payload = {
    "version": 1,
    "frameId": frame_id,
    "executionTime": round(execution_time, 4),
    "event": observation.event.value,
    "confidence": round(observation.confidence, 4),
    "voice": bool(voice_enabled and observation.voice_eligible),
    "detections": [
      {
        "classId": detection.class_id,
        "score": round(detection.score, 4),
        "bbox": [round(value, 4) for value in detection.bbox],
      }
      for detection in detections[:12]
    ],
  }
  return json.dumps(payload, separators=(",", ":")).encode()


def main() -> None:
  params = Params()
  sm = messaging.SubMaster(["carState"])
  pm = messaging.PubMaster(["customReservedRawData0"])
  interpreter = SceneInterpreter()

  cloudlog.warning("road perception connecting to road camera")
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not vipc_client.connect(False):
    time.sleep(0.1)

  detector = None
  last_inference = -INFERENCE_INTERVAL
  cloudlog.warning(f"road perception connected at {vipc_client.width}x{vipc_client.height}")

  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    now = time.monotonic()
    if now - last_inference < INFERENCE_INTERVAL:
      continue
    last_inference = now
    sm.update(0)

    if not params.get_bool("RoadPerceptionEnabled"):
      continue

    try:
      if detector is None:
        detector = YoloXDetector()
        cloudlog.warning(f"road perception model loaded: {MODEL_PATH.name} ({MODEL_SIZE}x{MODEL_SIZE})")

      started = time.perf_counter()
      model_input, image_bgr = preprocess_nv12(buf)
      detections = decode_yolox(detector.infer(model_input))
      observation = interpreter.update(detections, image_bgr, sm["carState"].vEgo, now)
      execution_time = time.perf_counter() - started
      event_param = PERCEPTION_EVENT_PARAM.get(observation.event)
      voice_enabled = (
        params.get_bool("RoadPerceptionVoiceEnabled")
        and event_param is not None
        and params.get_bool(event_param)
      )

      msg = messaging.new_message("customReservedRawData0", valid=True)
      msg.customReservedRawData0 = serialize_observation(
        vipc_client.frame_id,
        execution_time,
        observation,
        detections,
        voice_enabled,
      )
      pm.send("customReservedRawData0", msg)
    except Exception:
      cloudlog.exception("road perception inference failed")
      detector = None
      time.sleep(2.0)


if __name__ == "__main__":
  main()
