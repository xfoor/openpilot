import enum
import json
import math
from dataclasses import dataclass

import numpy as np


MODEL_SIZE = 416
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_TRAFFIC_LIGHT = 9
RELEVANT_CLASSES = (COCO_PERSON, COCO_BICYCLE, COCO_TRAFFIC_LIGHT)


class SceneEvent(str, enum.Enum):
  NONE = "none"
  PEDESTRIAN_RISK = "pedestrianRisk"
  CYCLIST_RISK = "cyclistRisk"
  TRAFFIC_LIGHT_RED = "trafficLightRed"
  TRAFFIC_LIGHT_YELLOW = "trafficLightYellow"
  TRAFFIC_LIGHT_GREEN = "trafficLightGreen"


PERCEPTION_PROMPT_MAP = {
  SceneEvent.PEDESTRIAN_RISK.value: 4,
  SceneEvent.CYCLIST_RISK.value: 5,
  SceneEvent.TRAFFIC_LIGHT_RED.value: 6,
  SceneEvent.TRAFFIC_LIGHT_YELLOW.value: 7,
  SceneEvent.TRAFFIC_LIGHT_GREEN.value: 8,
}

PERCEPTION_EVENT_PARAM = {
  SceneEvent.PEDESTRIAN_RISK: "RoadPerceptionPedestrianEnabled",
  SceneEvent.CYCLIST_RISK: "RoadPerceptionCyclistEnabled",
  SceneEvent.TRAFFIC_LIGHT_RED: "RoadPerceptionTrafficLightEnabled",
  SceneEvent.TRAFFIC_LIGHT_YELLOW: "RoadPerceptionTrafficLightEnabled",
  SceneEvent.TRAFFIC_LIGHT_GREEN: "RoadPerceptionTrafficLightEnabled",
}


def get_perception_prompt(raw_data) -> int:
  try:
    payload = json.loads(bytes(raw_data))
  except (TypeError, ValueError, UnicodeDecodeError):
    return 0

  if not payload.get("voice", False):
    return 0
  return PERCEPTION_PROMPT_MAP.get(payload.get("event"), 0)


@dataclass(frozen=True)
class Detection:
  class_id: int
  score: float
  bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneObservation:
  event: SceneEvent
  confidence: float
  voice_eligible: bool


def preprocess_nv12(buf, model_size: int = MODEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
  """Resize an NV12 VisionBuf directly into a padded BGR model image."""
  ratio = min(model_size / buf.width, model_size / buf.height)
  resized_w = max(1, int(buf.width * ratio))
  resized_h = max(1, int(buf.height * ratio))

  raw = np.frombuffer(buf.data, dtype=np.uint8)
  y_plane = raw[:buf.uv_offset].reshape((-1, buf.stride))
  uv_height = ((buf.height // 2) + 15) // 16 * 16
  uv_plane = raw[buf.uv_offset:buf.uv_offset + buf.stride * uv_height].reshape((-1, buf.stride))

  xs = np.minimum((np.arange(resized_w) / ratio).astype(np.int32), buf.width - 1)
  ys = np.minimum((np.arange(resized_h) / ratio).astype(np.int32), buf.height - 1)

  y = y_plane[ys[:, None], xs[None, :]].astype(np.int32)
  uv_xs = (xs // 2) * 2
  uv_ys = ys // 2
  u = uv_plane[uv_ys[:, None], uv_xs[None, :]].astype(np.int32)
  v = uv_plane[uv_ys[:, None], uv_xs[None, :] + 1].astype(np.int32)

  c = y - 16
  d = u - 128
  e = v - 128
  red = np.clip((298 * c + 409 * e + 128) >> 8, 0, 255)
  green = np.clip((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255)
  blue = np.clip((298 * c + 516 * d + 128) >> 8, 0, 255)

  image = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
  image[:resized_h, :resized_w] = np.stack((blue, green, red), axis=-1).astype(np.uint8)
  model_input = image.transpose(2, 0, 1).astype(np.float32)
  return model_input, image


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
  if len(boxes) == 0:
    return []

  x1, y1, x2, y2 = boxes.T
  areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
  order = scores.argsort()[::-1]
  keep = []

  while order.size:
    current = int(order[0])
    keep.append(current)
    if order.size == 1:
      break

    rest = order[1:]
    xx1 = np.maximum(x1[current], x1[rest])
    yy1 = np.maximum(y1[current], y1[rest])
    xx2 = np.minimum(x2[current], x2[rest])
    yy2 = np.minimum(y2[current], y2[rest])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    union = areas[current] + areas[rest] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    order = rest[iou <= threshold]

  return keep


def decode_yolox(output: np.ndarray, model_size: int = MODEL_SIZE,
                 score_threshold: float = 0.35, nms_threshold: float = 0.45) -> list[Detection]:
  predictions = np.asarray(output, dtype=np.float32).reshape((-1, 85)).copy()
  grids = []
  expanded_strides = []
  for stride in (8, 16, 32):
    grid_size = model_size // stride
    yv, xv = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    grids.append(np.stack((xv, yv), axis=-1).reshape((-1, 2)))
    expanded_strides.append(np.full((grid_size * grid_size, 1), stride))

  grid = np.concatenate(grids, axis=0)
  strides = np.concatenate(expanded_strides, axis=0)
  if predictions.shape[0] != grid.shape[0]:
    raise ValueError(f"unexpected YOLOX output rows: {predictions.shape[0]}")

  predictions[:, :2] = (predictions[:, :2] + grid) * strides
  predictions[:, 2:4] = np.exp(np.clip(predictions[:, 2:4], -10, 10)) * strides

  detections = []
  for class_id in RELEVANT_CLASSES:
    scores = predictions[:, 4] * predictions[:, 5 + class_id]
    selected = scores >= score_threshold
    if not np.any(selected):
      continue

    xywh = predictions[selected, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    boxes = np.clip(boxes / model_size, 0.0, 1.0)
    class_scores = scores[selected]

    for index in _nms(boxes, class_scores, nms_threshold):
      detections.append(Detection(
        class_id=class_id,
        score=float(class_scores[index]),
        bbox=tuple(float(value) for value in boxes[index]),
      ))

  return sorted(detections, key=lambda detection: detection.score, reverse=True)


def classify_traffic_light(image_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[str | None, float]:
  height, width = image_bgr.shape[:2]
  x1, y1, x2, y2 = bbox
  left, right = int(x1 * width), int(math.ceil(x2 * width))
  top, bottom = int(y1 * height), int(math.ceil(y2 * height))
  crop = image_bgr[max(0, top):min(height, bottom), max(0, left):min(width, right)]
  if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 2:
    return None, 0.0

  blue, green, red = np.moveaxis(crop.astype(np.float32), -1, 0)
  brightness = np.maximum.reduce((red, green, blue))
  bright_pixels = max(1, int(np.count_nonzero(brightness > 80)))

  masks = {
    "red": (red > 110) & (red > green * 1.25) & (red > blue * 1.25),
    "yellow": (red > 110) & (green > 90) & (np.abs(red - green) < 65) & (blue < np.minimum(red, green) * 0.75),
    "green": (green > 95) & (green > red * 1.20) & (green > blue * 1.10),
  }
  ratios = {color: float(np.count_nonzero(mask)) / bright_pixels for color, mask in masks.items()}
  color = max(ratios, key=ratios.get)
  confidence = ratios[color]
  return (color, confidence) if confidence >= 0.08 else (None, confidence)


class SceneInterpreter:
  CONFIRMATIONS = 3
  EVENT_COOLDOWN = 8.0

  def __init__(self):
    self.candidate = SceneEvent.NONE
    self.candidate_count = 0
    self.last_spoken_at = dict.fromkeys(SceneEvent, -math.inf)
    self.last_confirmed_light: str | None = None

  @staticmethod
  def _in_driving_corridor(detection: Detection) -> bool:
    x1, _, x2, y2 = detection.bbox
    center_x = (x1 + x2) / 2
    half_width = 0.08 + 0.18 * y2
    return y2 >= 0.50 and abs(center_x - 0.5) <= half_width

  def _candidate_event(self, detections: list[Detection], image_bgr: np.ndarray,
                       ego_speed: float) -> tuple[SceneEvent, float, bool]:
    for detection in detections:
      if detection.class_id in (COCO_PERSON, COCO_BICYCLE) and self._in_driving_corridor(detection):
        event = SceneEvent.PEDESTRIAN_RISK if detection.class_id == COCO_PERSON else SceneEvent.CYCLIST_RISK
        voice_eligible = detection.score >= 0.60 and detection.bbox[3] >= 0.58 and ego_speed > 1.0
        return event, detection.score, voice_eligible

    for detection in detections:
      x1, _, x2, y2 = detection.bbox
      center_x = (x1 + x2) / 2
      if detection.class_id != COCO_TRAFFIC_LIGHT or detection.score < 0.45 or not 0.25 <= center_x <= 0.75 or y2 > 0.65:
        continue

      color, color_confidence = classify_traffic_light(image_bgr, detection.bbox)
      if color is None:
        continue
      event = SceneEvent(f"trafficLight{color.title()}")
      confidence = min(detection.score, color_confidence)
      voice_eligible = (
        (color == "red" and ego_speed > 1.0)
        or (color == "yellow" and ego_speed > 2.0)
        or (color == "green" and ego_speed < 1.5 and self.last_confirmed_light == "red")
      )
      return event, confidence, voice_eligible

    return SceneEvent.NONE, 0.0, False

  def update(self, detections: list[Detection], image_bgr: np.ndarray,
             ego_speed: float, now: float) -> SceneObservation:
    event, confidence, voice_eligible = self._candidate_event(detections, image_bgr, ego_speed)
    if event == SceneEvent.NONE:
      self.candidate = SceneEvent.NONE
      self.candidate_count = 0
      return SceneObservation(event, confidence, False)

    if event == self.candidate:
      self.candidate_count += 1
    else:
      self.candidate = event
      self.candidate_count = 1

    if self.candidate_count < self.CONFIRMATIONS:
      return SceneObservation(SceneEvent.NONE, confidence, False)

    if event.value.startswith("trafficLight"):
      self.last_confirmed_light = event.value.removeprefix("trafficLight").lower()

    ready = now - self.last_spoken_at[event] >= self.EVENT_COOLDOWN
    if voice_eligible and ready:
      self.last_spoken_at[event] = now
      return SceneObservation(event, confidence, True)
    return SceneObservation(event, confidence, False)
