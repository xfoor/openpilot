#!/usr/bin/env python3
import json
import pickle
import sys
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
  SceneEvent,
  SceneInterpreter,
  SceneObservation,
  classify_driver_gaze,
  decode_yolox,
  preprocess_nv12,
  remap_detections,
)
from openpilot.selfdrive.road_observer.speed_limit import (
  SPEED_LIMIT_MODEL_SIZE,
  SPEED_SIGN_ROI,
  SpeedLimitObservation,
  SpeedLimitTracker,
  classify_speed_limit_candidates,
  decode_speed_sign_yolov8,
  project_speed_limit_detections,
  remap_speed_sign_candidates,
)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano.onnx"
COMPILED_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolox_nano_tinygrad.pkl"
SPEED_SIGN_MODEL_PATH = Path(__file__).resolve().parent / "models" / "speed_sign_yolov8n.onnx"
SPEED_CLASSIFIER_MODEL_PATH = Path(__file__).resolve().parent / "models" / "speed_sign_classifier.onnx"
ONNX_RUNTIME_VENDOR_PATH = Path(__file__).resolve().parent / "vendor"
INFERENCE_INTERVAL = 0.5


class YoloXDetector:
  def __init__(self, model_path: Path = MODEL_PATH, compiled_model_path: Path = COMPILED_MODEL_PATH,
               device: str = "QCOM", input_name: str = "images"):
    from tinygrad import TinyJit
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.tensor import Tensor

    self.Tensor = Tensor
    self.device = device
    self.input_name = input_name
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
    return self.run(**{self.input_name: tensor}).numpy()


class OnnxRuntimeCpuModel:
  def __init__(self, model_path: Path, intra_op_num_threads: int):
    vendor_path = str(ONNX_RUNTIME_VENDOR_PATH)
    if vendor_path not in sys.path:
      sys.path.insert(0, vendor_path)

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_num_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    self.session = ort.InferenceSession(
      str(model_path),
      sess_options=options,
      providers=["CPUExecutionProvider"],
    )
    inputs = self.session.get_inputs()
    if len(inputs) != 1:
      raise ValueError(f"unexpected ONNX model input count: {len(inputs)}")
    self.input_name = inputs[0].name

  def infer(self, model_input: np.ndarray):
    batched_input = np.ascontiguousarray(model_input[np.newaxis], dtype=np.float32)
    return self.session.run(None, {self.input_name: batched_input})[0]


class SpeedSignDetector(OnnxRuntimeCpuModel):
  def __init__(self, model_path: Path = SPEED_SIGN_MODEL_PATH):
    super().__init__(model_path, intra_op_num_threads=2)


class SpeedSignClassifier(OnnxRuntimeCpuModel):
  def __init__(self, model_path: Path = SPEED_CLASSIFIER_MODEL_PATH):
    super().__init__(model_path, intra_op_num_threads=1)


def serialize_observation(frame_id: int, execution_time: float, observation,
                          detections, event_id: int, camera_source: str,
                          speed_limit: SpeedLimitObservation | None = None,
                          speed_limit_detections=(), speed_limit_event_id: int = 0,
                          speed_limit_overspeed: bool = False) -> bytes:
  speed_limit = speed_limit or SpeedLimitObservation()
  speed_limit_voice = event_id == 0 and speed_limit_event_id > 0
  event = SceneEvent.SPEED_LIMIT.value if speed_limit_voice else observation.event.value
  confidence = speed_limit.confidence if speed_limit_voice else observation.confidence
  voice_event_id = speed_limit_event_id if speed_limit_voice else event_id
  payload = {
    "version": 5,
    "frameId": frame_id,
    "executionTime": round(execution_time, 4),
    "event": event,
    "confidence": round(confidence, 4),
    "voice": voice_event_id > 0,
    "eventId": voice_event_id,
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
    "speedLimit": {
      "limitKph": speed_limit.limit_kph,
      "confidence": round(speed_limit.confidence, 4),
      "confirmed": speed_limit.confirmed,
      "announce": speed_limit.announce,
      "trackId": speed_limit.track_id,
      "reason": speed_limit.reason,
      "overspeed": speed_limit_overspeed,
      "detections": [
        {
          "limitKph": detection.limit_kph,
          "score": round(detection.score, 4),
          "bbox": [round(value, 4) for value in detection.bbox],
        }
        for detection in speed_limit_detections[:6]
      ],
    },
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
  speed_limit_tracker = SpeedLimitTracker()

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
  speed_sign_detector = None
  speed_sign_classifier = None
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

    perception_enabled = params.get_bool("RoadPerceptionEnabled")
    speed_limit_enabled = params.get_bool("RoadSpeedLimitEnabled")
    if not perception_enabled and not speed_limit_enabled:
      continue

    try:
      car_state = sm["carState"]
      turn_signal = car_state.leftBlinker or car_state.rightBlinker
      turning = turn_signal or abs(car_state.steeringAngleDeg) >= 25.0
      camera_source = "road"
      camera_client = road_client
      buf = road_buf
      if perception_enabled and wide_client is not None and turning and car_state.vEgo < 15.0:
        wide_buf = wide_client.recv()
        if wide_buf is not None:
          camera_source = "wide"
          camera_client = wide_client
          buf = wide_buf

      started = time.perf_counter()
      detections = []
      observation = SceneObservation(SceneEvent.NONE, 0.0, False)
      if perception_enabled:
        if detector is None:
          detector = YoloXDetector()
          backend = COMPILED_MODEL_PATH.name if detector.precompiled else MODEL_PATH.name
          cloudlog.warning(f"road perception model loaded: {backend} ({MODEL_SIZE}x{MODEL_SIZE})")
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

      speed_limit_detections = []
      speed_limit = SpeedLimitObservation()
      if speed_limit_enabled:
        if speed_sign_detector is None:
          speed_sign_detector = SpeedSignDetector()
          speed_sign_classifier = SpeedSignClassifier()
          cloudlog.warning(
            f"speed sign models loaded: {SPEED_SIGN_MODEL_PATH.name} + {SPEED_CLASSIFIER_MODEL_PATH.name} (ONNX Runtime CPU)",
          )
        speed_input, speed_image = preprocess_nv12(
          road_buf,
          SPEED_LIMIT_MODEL_SIZE,
          SPEED_SIGN_ROI,
        )
        speed_input = np.ascontiguousarray(speed_input[::-1]) * (1.0 / 255.0)
        speed_sign_candidates = decode_speed_sign_yolov8(speed_sign_detector.infer(speed_input))
        speed_sign_candidates = remap_speed_sign_candidates(
          speed_sign_candidates,
          speed_image.shape[1],
          speed_image.shape[0],
        )
        speed_limit_detections = classify_speed_limit_candidates(
          np.ascontiguousarray(speed_image[:, :, ::-1]),
          speed_sign_candidates,
          speed_sign_classifier.infer,
        )
        speed_limit_detections = project_speed_limit_detections(speed_limit_detections)
        speed_limit = speed_limit_tracker.update(speed_limit_detections, now)

      execution_time = time.perf_counter() - started
      event_param = PERCEPTION_EVENT_PARAM.get(observation.event)
      voice_enabled = (
        params.get_bool("RoadPerceptionVoiceEnabled")
        and event_param is not None
        and params.get_bool(event_param)
      )
      event_id = time.monotonic_ns() if voice_enabled and observation.voice_eligible else 0
      speed_limit_overspeed = (
        speed_limit.limit_kph is not None
        and (
          car_state.vEgo * 3.6 > speed_limit.limit_kph + 3.0
          or (
            car_state.cruiseState.enabled
            and car_state.cruiseState.speed * 3.6 > speed_limit.limit_kph + 3.0
          )
        )
      )
      speed_limit_event_id = (
        time.monotonic_ns()
        if (
          event_id == 0
          and speed_limit.announce
          and params.get_bool("RoadSpeedLimitVoiceEnabled")
        )
        else 0
      )
      if speed_limit_event_id:
        speed_limit_tracker.mark_announced(speed_limit, now)
      output_camera = "road" if speed_limit_event_id else camera_source
      output_frame_id = road_client.frame_id if speed_limit_event_id else camera_client.frame_id

      msg = messaging.new_message("customReservedRawData0", valid=True)
      msg.customReservedRawData0 = serialize_observation(
        output_frame_id,
        execution_time,
        observation,
        detections,
        event_id,
        output_camera,
        speed_limit,
        speed_limit_detections,
        speed_limit_event_id,
        speed_limit_overspeed,
      )
      pm.send("customReservedRawData0", msg)
    except Exception:
      cloudlog.exception("road perception inference failed")
      detector = None
      speed_sign_detector = None
      speed_sign_classifier = None
      time.sleep(2.0)


if __name__ == "__main__":
  main()
