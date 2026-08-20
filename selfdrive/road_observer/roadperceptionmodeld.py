#!/usr/bin/env python3
import json
import pickle
import time
from pathlib import Path

import numpy as np

from cereal import log
from cereal import messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.road_observer.perception import (
  MODEL_SIZE,
  PERCEPTION_EVENT_PARAM,
  RoadGeometry,
  SceneInterpreter,
  decode_yolox,
  preprocess_nv12,
  remap_detections,
)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano.onnx"
COMPILED_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano_tinygrad.pkl"
INFERENCE_INTERVAL = 0.5


class YoloXDetector:
  def __init__(self, model_path: Path = MODEL_PATH, compiled_model_path: Path = COMPILED_MODEL_PATH,
               device: str = "QCOM"):
    from tinygrad import TinyJit
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.tensor import Tensor

    self.Tensor = Tensor
    self.device = device
    self.precompiled = device == "QCOM" and compiled_model_path.is_file()
    if self.precompiled:
      with compiled_model_path.open("rb") as model_file:
        self.run = pickle.load(model_file)
      self.input_device = "NPY"
    else:
      runner = OnnxRunner(str(model_path))
      self.run = TinyJit(
        lambda images: next(iter(runner({"images": images}).values())).cast("float32"),
        prune=True,
      )
      self.input_device = device

  def infer(self, model_input: np.ndarray):
    tensor = self.Tensor(model_input[np.newaxis], device=self.input_device).realize()
    return self.run(images=tensor).numpy()


def serialize_observation(frame_id: int, execution_time: float, observation,
                          detections, voice_enabled: bool) -> bytes:
  payload = {
    "version": 3,
    "frameId": frame_id,
    "executionTime": round(execution_time, 4),
    "event": observation.event.value,
    "confidence": round(observation.confidence, 4),
    "voice": bool(voice_enabled and observation.voice_eligible),
    "risk": {
      "reason": observation.reason,
      "trackId": observation.track_id,
      "distance": round(observation.distance, 2) if observation.distance is not None else None,
      "pathOffset": round(observation.path_offset, 2) if observation.path_offset is not None else None,
      "lateralSpeed": round(observation.lateral_speed, 2) if observation.lateral_speed is not None else None,
    },
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


def build_road_geometry(sm: messaging.SubMaster) -> RoadGeometry | None:
  services = ("modelV2", "liveCalibration", "deviceState", "roadCameraState")
  if not all(sm.valid[service] and sm.alive[service] for service in services):
    return None

  calibration = sm["liveCalibration"]
  if (
    calibration.calStatus != log.LiveCalibrationData.Status.calibrated
    or len(calibration.rpyCalib) != 3
    or not calibration.height
  ):
    return None

  camera_key = (str(sm["deviceState"].deviceType), str(sm["roadCameraState"].sensor))
  camera = DEVICE_CAMERAS.get(camera_key)
  if camera is None:
    return None

  path = sm["modelV2"].position
  return RoadGeometry.build(
    path.x,
    path.y,
    calibration.rpyCalib,
    calibration.height[0],
    camera.fcam.intrinsics,
    camera.fcam.size,
  )


def main() -> None:
  params = Params()
  sm = messaging.SubMaster(["carState", "modelV2", "liveCalibration", "deviceState", "roadCameraState"])
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
        backend = COMPILED_MODEL_PATH.name if detector.precompiled else MODEL_PATH.name
        cloudlog.warning(f"road perception model loaded: {backend} ({MODEL_SIZE}x{MODEL_SIZE})")

      started = time.perf_counter()
      model_input, image_bgr = preprocess_nv12(buf)
      detections = decode_yolox(detector.infer(model_input))
      detections = remap_detections(detections, image_bgr.shape[1], image_bgr.shape[0])
      geometry = build_road_geometry(sm)
      car_state = sm["carState"]
      turning = (
        car_state.leftBlinker
        or car_state.rightBlinker
        or abs(car_state.steeringAngleDeg) >= 25.0
      )
      observation = interpreter.update(
        detections,
        image_bgr,
        car_state.vEgo,
        now,
        geometry,
        turning=turning,
      )
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
