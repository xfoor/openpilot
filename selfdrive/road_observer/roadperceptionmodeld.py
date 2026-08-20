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
from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler
from openpilot.selfdrive.road_observer.perception import (
  MODEL_SIZE,
  PERCEPTION_EVENT_PARAM,
  RoadGeometry,
  SceneInterpreter,
  classify_driver_gaze,
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
                          detections, event_id: int, camera_source: str) -> bytes:
  payload = {
    "version": 4,
    "frameId": frame_id,
    "executionTime": round(execution_time, 4),
    "event": observation.event.value,
    "confidence": round(observation.confidence, 4),
    "voice": event_id > 0,
    "eventId": event_id,
    "camera": camera_source,
    "side": observation.side,
    "risk": {
      "reason": observation.reason,
      "trackId": observation.track_id,
      "distance": round(observation.distance, 2) if observation.distance is not None else None,
      "pathOffset": round(observation.path_offset, 2) if observation.path_offset is not None else None,
      "lateralSpeed": round(observation.lateral_speed, 2) if observation.lateral_speed is not None else None,
      "driverCheckedSide": observation.driver_checked_side,
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


def camera_calibration_rpy(rpy_calib, wide_from_device_euler,
                           camera_source: str) -> np.ndarray | None:
  rpy = np.asarray(rpy_calib, dtype=np.float64)
  if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
    return None
  if camera_source != "wide":
    return rpy

  wide_rpy = np.asarray(wide_from_device_euler, dtype=np.float64)
  if wide_rpy.shape != (3,) or not np.all(np.isfinite(wide_rpy)):
    return None
  return euler_from_rot(rot_from_euler(wide_rpy) @ rot_from_euler(rpy))


def build_road_geometry(sm: messaging.SubMaster, camera_source: str = "road") -> RoadGeometry | None:
  camera_state_service = "wideRoadCameraState" if camera_source == "wide" else "roadCameraState"
  services = ("modelV2", "liveCalibration", "deviceState", "roadCameraState", camera_state_service)
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

  calibration_rpy = camera_calibration_rpy(
    calibration.rpyCalib,
    calibration.wideFromDeviceEuler,
    camera_source,
  )
  if calibration_rpy is None:
    return None

  path = sm["modelV2"].position
  return RoadGeometry.build(
    path.x,
    path.y,
    calibration_rpy,
    calibration.height[0],
    camera.ecam.intrinsics if camera_source == "wide" else camera.fcam.intrinsics,
    camera.ecam.size if camera_source == "wide" else camera.fcam.size,
  )


def main() -> None:
  params = Params()
  sm = messaging.SubMaster([
    "carState",
    "modelV2",
    "liveCalibration",
    "deviceState",
    "roadCameraState",
    "wideRoadCameraState",
    "driverMonitoringState",
  ])
  pm = messaging.PubMaster(["customReservedRawData0"])
  interpreter = SceneInterpreter()

  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      break
    time.sleep(0.1)
  wide_available = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams

  cloudlog.warning(f"road perception connecting to cameras (wide={wide_available})")
  road_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not road_client.connect(False):
    time.sleep(0.1)
  wide_client = None
  if wide_available:
    wide_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)
    while not wide_client.connect(False):
      time.sleep(0.1)

  detector = None
  last_inference = -INFERENCE_INTERVAL
  cloudlog.warning(f"road perception road camera connected at {road_client.width}x{road_client.height}")

  while True:
    road_buf = road_client.recv()
    if road_buf is None:
      continue

    now = time.monotonic()
    sm.update(0)
    monitoring = sm["driverMonitoringState"].visionPolicyState
    driver_gaze_side = classify_driver_gaze(
      monitoring.pose.yaw,
      monitoring.faceDetected,
      monitoring.pose.uncertainty,
    ) if sm.valid["driverMonitoringState"] and sm.alive["driverMonitoringState"] else None
    interpreter.note_driver_gaze(driver_gaze_side, now)
    if now - last_inference < INFERENCE_INTERVAL:
      continue
    last_inference = now

    if not params.get_bool("RoadPerceptionEnabled"):
      continue

    try:
      car_state = sm["carState"]
      turn_signal = car_state.leftBlinker or car_state.rightBlinker
      turning = turn_signal or abs(car_state.steeringAngleDeg) >= 25.0
      camera_source = "road"
      camera_client = road_client
      buf = road_buf
      if wide_client is not None and turning and car_state.vEgo < 15.0:
        wide_buf = wide_client.recv()
        if wide_buf is not None:
          camera_source = "wide"
          camera_client = wide_client
          buf = wide_buf

      if detector is None:
        detector = YoloXDetector()
        backend = COMPILED_MODEL_PATH.name if detector.precompiled else MODEL_PATH.name
        cloudlog.warning(f"road perception model loaded: {backend} ({MODEL_SIZE}x{MODEL_SIZE})")

      started = time.perf_counter()
      model_input, image_bgr = preprocess_nv12(buf)
      detections = decode_yolox(detector.infer(model_input))
      detections = remap_detections(detections, image_bgr.shape[1], image_bgr.shape[0])
      geometry = build_road_geometry(sm, camera_source)
      observation = interpreter.update(
        detections,
        image_bgr,
        car_state.vEgo,
        now,
        geometry,
        turning=turning,
        turn_signal=turn_signal,
        camera_source=camera_source,
        driver_gaze_side=driver_gaze_side,
      )
      execution_time = time.perf_counter() - started
      event_param = PERCEPTION_EVENT_PARAM.get(observation.event)
      voice_enabled = (
        params.get_bool("RoadPerceptionVoiceEnabled")
        and event_param is not None
        and params.get_bool(event_param)
      )
      event_id = time.monotonic_ns() if voice_enabled and observation.voice_eligible else 0

      msg = messaging.new_message("customReservedRawData0", valid=True)
      msg.customReservedRawData0 = serialize_observation(
        camera_client.frame_id,
        execution_time,
        observation,
        detections,
        event_id,
        camera_source,
      )
      pm.send("customReservedRawData0", msg)
    except Exception:
      cloudlog.exception("road perception inference failed")
      detector = None
      time.sleep(2.0)


if __name__ == "__main__":
  main()
