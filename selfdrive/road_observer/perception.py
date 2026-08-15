import enum
import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from openpilot.common.transformations.camera import get_view_frame_from_calib_frame


MODEL_SIZE = 320
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_TRAFFIC_LIGHT = 9
RELEVANT_CLASSES = (COCO_PERSON, COCO_BICYCLE, COCO_TRAFFIC_LIGHT)
MIN_GROUND_DISTANCE = 1.5
MAX_GROUND_DISTANCE = 80.0
PEDESTRIAN_CORRIDOR_HALF_WIDTH = 1.6
CYCLIST_CORRIDOR_HALF_WIDTH = 1.8


class SceneEvent(enum.StrEnum):
  NONE = "none"
  PEDESTRIAN_RISK = "pedestrianRisk"
  CYCLIST_RISK = "cyclistRisk"
  TRAFFIC_LIGHT_RED = "trafficLightRed"
  TRAFFIC_LIGHT_YELLOW = "trafficLightYellow"
  TRAFFIC_LIGHT_GREEN = "trafficLightGreen"


PERCEPTION_PROMPT_MAP = {
  SceneEvent.PEDESTRIAN_RISK.value: 8,
  SceneEvent.CYCLIST_RISK.value: 9,
}

PERCEPTION_EVENT_PARAM = {
  SceneEvent.PEDESTRIAN_RISK: "RoadPerceptionPedestrianEnabled",
  SceneEvent.CYCLIST_RISK: "RoadPerceptionCyclistEnabled",
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
  reason: str = "none"
  track_id: int | None = None
  distance: float | None = None
  path_offset: float | None = None
  lateral_speed: float | None = None


@dataclass(frozen=True)
class RoadGeometry:
  path_x: np.ndarray
  path_y: np.ndarray
  intrinsics: np.ndarray
  view_from_calib: np.ndarray
  camera_height: float
  camera_width: int
  camera_height_pixels: int

  @classmethod
  def build(cls, path_x, path_y, rpy, camera_height: float,
            intrinsics: np.ndarray, camera_size: tuple[int, int]) -> "RoadGeometry | None":
    path_x_array = np.asarray(path_x, dtype=np.float64)
    path_y_array = np.asarray(path_y, dtype=np.float64)
    rpy_array = np.asarray(rpy, dtype=np.float64)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float64)
    width, height = camera_size

    valid = (
      path_x_array.ndim == 1
      and path_y_array.shape == path_x_array.shape
      and len(path_x_array) >= 2
      and rpy_array.shape == (3,)
      and intrinsics_array.shape == (3, 3)
      and np.all(np.isfinite(path_x_array))
      and np.all(np.isfinite(path_y_array))
      and np.all(np.isfinite(rpy_array))
      and np.all(np.isfinite(intrinsics_array))
      and math.isfinite(camera_height)
      and 0.5 <= camera_height <= 3.0
      and width > 0
      and height > 0
    )
    if not valid:
      return None

    order = np.argsort(path_x_array)
    path_x_array = path_x_array[order]
    path_y_array = path_y_array[order]
    unique = np.concatenate(([True], np.diff(path_x_array) > 1e-3))
    path_x_array = path_x_array[unique]
    path_y_array = path_y_array[unique]
    if len(path_x_array) < 2 or path_x_array[-1] < MIN_GROUND_DISTANCE:
      return None

    view_from_calib = get_view_frame_from_calib_frame(*rpy_array, camera_height)
    return cls(
      path_x=path_x_array,
      path_y=path_y_array,
      intrinsics=intrinsics_array,
      view_from_calib=view_from_calib,
      camera_height=float(camera_height),
      camera_width=int(width),
      camera_height_pixels=int(height),
    )

  def ground_position(self, detection: Detection) -> tuple[float, float] | None:
    x1, _, x2, y2 = detection.bbox
    pixel = np.array([
      (x1 + x2) * 0.5 * self.camera_width,
      y2 * self.camera_height_pixels,
      1.0,
    ])
    try:
      ray_view = np.linalg.solve(self.intrinsics, pixel)
    except np.linalg.LinAlgError:
      return None

    rotation = self.view_from_calib[:, :3]
    translation = self.view_from_calib[:, 3]
    ray_calib = rotation.T @ ray_view
    camera_origin = -rotation.T @ translation
    if ray_calib[2] <= 1e-5:
      return None

    scale = -camera_origin[2] / ray_calib[2]
    point = camera_origin + scale * ray_calib
    distance = float(point[0])
    if (
      not np.all(np.isfinite(point))
      or not MIN_GROUND_DISTANCE <= distance <= MAX_GROUND_DISTANCE
      or distance < self.path_x[0]
      or distance > self.path_x[-1]
    ):
      return None

    path_lateral = float(np.interp(distance, self.path_x, self.path_y))
    return distance, float(point[1] - path_lateral)


@dataclass(frozen=True)
class TrackedDetection:
  detection: Detection
  track_id: int
  hits: int
  distance: float | None
  path_offset: float | None
  lateral_speed: float | None


@dataclass
class _Track:
  track_id: int
  class_id: int
  bbox: tuple[float, float, float, float]
  last_seen: float
  hits: int
  ground_history: deque[tuple[float, float, float]]


def _bbox_iou(first: tuple[float, float, float, float],
              second: tuple[float, float, float, float]) -> float:
  left = max(first[0], second[0])
  top = max(first[1], second[1])
  right = min(first[2], second[2])
  bottom = min(first[3], second[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
  second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
  union = first_area + second_area - intersection
  return intersection / union if union > 0 else 0.0


def _bbox_center_distance(first: tuple[float, float, float, float],
                          second: tuple[float, float, float, float]) -> float:
  first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
  second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
  return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


class DetectionTracker:
  MAX_AGE = 1.1
  HISTORY_SECONDS = 1.6

  def __init__(self):
    self.tracks: dict[int, _Track] = {}
    self.next_track_id = 1

  @staticmethod
  def _lateral_speed(history: deque[tuple[float, float, float]]) -> float | None:
    if len(history) < 3 or history[-1][0] - history[0][0] < 0.8:
      return None
    times = np.array([sample[0] for sample in history], dtype=np.float64)
    offsets = np.array([sample[2] for sample in history], dtype=np.float64)
    times -= times[-1]
    speed = float(np.polyfit(times, offsets, 1)[0])
    return speed if math.isfinite(speed) and abs(speed) <= 5.0 else None

  def _match(self, detection: Detection, available: set[int]) -> int | None:
    best_id = None
    best_cost = math.inf
    box_height = detection.bbox[3] - detection.bbox[1]
    for track_id in available:
      track = self.tracks[track_id]
      if track.class_id != detection.class_id:
        continue
      iou = _bbox_iou(track.bbox, detection.bbox)
      center_distance = _bbox_center_distance(track.bbox, detection.bbox)
      max_center_distance = max(0.06, box_height * 0.65)
      if iou < 0.05 and center_distance > max_center_distance:
        continue
      cost = center_distance + 0.15 * (1.0 - iou)
      if cost < best_cost:
        best_id = track_id
        best_cost = cost
    return best_id

  def update(self, detections: list[Detection], geometry: RoadGeometry | None,
             now: float) -> list[TrackedDetection]:
    self.tracks = {
      track_id: track for track_id, track in self.tracks.items()
      if now - track.last_seen <= self.MAX_AGE
    }
    available = set(self.tracks)
    tracked = []

    for detection in detections:
      if detection.class_id not in (COCO_PERSON, COCO_BICYCLE):
        continue
      track_id = self._match(detection, available)
      if track_id is None:
        track_id = self.next_track_id
        self.next_track_id += 1
        track = _Track(
          track_id=track_id,
          class_id=detection.class_id,
          bbox=detection.bbox,
          last_seen=now,
          hits=0,
          ground_history=deque(),
        )
        self.tracks[track_id] = track
      else:
        track = self.tracks[track_id]
        available.remove(track_id)

      track.bbox = detection.bbox
      track.last_seen = now
      track.hits += 1
      ground = geometry.ground_position(detection) if geometry is not None else None
      if ground is not None:
        distance, path_offset = ground
        track.ground_history.append((now, distance, path_offset))
        while track.ground_history and now - track.ground_history[0][0] > self.HISTORY_SECONDS:
          track.ground_history.popleft()
      else:
        distance = None
        path_offset = None

      tracked.append(TrackedDetection(
        detection=detection,
        track_id=track.track_id,
        hits=track.hits,
        distance=distance,
        path_offset=path_offset,
        lateral_speed=self._lateral_speed(track.ground_history),
      ))

    return tracked


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

  resized_image = np.stack((blue, green, red), axis=-1).astype(np.uint8)
  model_image = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
  model_image[:resized_h, :resized_w] = resized_image
  model_input = model_image.transpose(2, 0, 1).astype(np.float32)
  return model_input, resized_image


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


def remap_detections(detections: list[Detection], image_width: int, image_height: int,
                     model_size: int = MODEL_SIZE) -> list[Detection]:
  """Map boxes from the square model canvas to the unpadded camera image."""
  x_scale = model_size / image_width
  y_scale = model_size / image_height
  remapped = []
  for detection in detections:
    x1, y1, x2, y2 = detection.bbox
    bbox = (
      float(np.clip(x1 * x_scale, 0.0, 1.0)),
      float(np.clip(y1 * y_scale, 0.0, 1.0)),
      float(np.clip(x2 * x_scale, 0.0, 1.0)),
      float(np.clip(y2 * y_scale, 0.0, 1.0)),
    )
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
      remapped.append(Detection(detection.class_id, detection.score, bbox))
  return remapped


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
    self.tracker = DetectionTracker()
    self.candidate = SceneEvent.NONE
    self.candidate_count = 0
    self.last_spoken_at = dict.fromkeys(SceneEvent, -math.inf)

  @staticmethod
  def _warning_distance(ego_speed: float) -> float:
    speed = max(0.0, ego_speed)
    reaction_distance = speed * 1.5
    braking_distance = speed * speed / (2.0 * 4.5)
    return float(np.clip(reaction_distance + braking_distance + 5.0, 12.0, MAX_GROUND_DISTANCE))

  @staticmethod
  def _road_user_observation(tracked: TrackedDetection, ego_speed: float,
                             warning_distance: float) -> SceneObservation | None:
    if tracked.distance is None or tracked.path_offset is None:
      return None

    is_pedestrian = tracked.detection.class_id == COCO_PERSON
    corridor_half_width = PEDESTRIAN_CORRIDOR_HALF_WIDTH if is_pedestrian else CYCLIST_CORRIDOR_HALF_WIDTH
    event = SceneEvent.PEDESTRIAN_RISK if is_pedestrian else SceneEvent.CYCLIST_RISK
    distance_relevant = MIN_GROUND_DISTANCE <= tracked.distance <= warning_distance
    in_corridor = abs(tracked.path_offset) <= corridor_half_width

    crossing = False
    if tracked.lateral_speed is not None and abs(tracked.path_offset) > corridor_half_width:
      toward_speed = -tracked.lateral_speed * math.copysign(1.0, tracked.path_offset)
      if toward_speed >= 0.5:
        time_to_corridor = (abs(tracked.path_offset) - corridor_half_width) / toward_speed
        distance_at_entry = tracked.distance - ego_speed * time_to_corridor
        crossing = 0.0 <= time_to_corridor <= 2.5 and MIN_GROUND_DISTANCE <= distance_at_entry <= warning_distance

    if not distance_relevant or not (in_corridor or crossing):
      return None

    required_hits = 2 if in_corridor else 3
    confidence_threshold = 0.60 if in_corridor else 0.65
    voice_eligible = (
      ego_speed > 1.0
      and tracked.hits >= required_hits
      and tracked.detection.score >= confidence_threshold
    )
    reason = "pathOccupied" if in_corridor else "crossingPredicted"
    return SceneObservation(
      event=event,
      confidence=tracked.detection.score,
      voice_eligible=voice_eligible,
      reason=reason,
      track_id=tracked.track_id,
      distance=tracked.distance,
      path_offset=tracked.path_offset,
      lateral_speed=tracked.lateral_speed,
    )

  @staticmethod
  def _traffic_light_observation(detections: list[Detection],
                                 image_bgr: np.ndarray) -> SceneObservation | None:
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
      return SceneObservation(
        event=event,
        confidence=confidence,
        voice_eligible=False,
        reason="shadowOnlyNoLaneAssociation",
      )

    return None

  def _candidate_observation(self, detections: list[Detection], image_bgr: np.ndarray,
                             ego_speed: float, now: float,
                             geometry: RoadGeometry | None) -> SceneObservation:
    warning_distance = self._warning_distance(ego_speed)
    road_users = []
    for tracked in self.tracker.update(detections, geometry, now):
      observation = self._road_user_observation(tracked, ego_speed, warning_distance)
      if observation is not None:
        road_users.append(observation)

    if road_users:
      return min(
        road_users,
        key=lambda observation: (not observation.voice_eligible, observation.distance or math.inf),
      )

    return self._traffic_light_observation(detections, image_bgr) or SceneObservation(
      SceneEvent.NONE, 0.0, False,
    )

  def update(self, detections: list[Detection], image_bgr: np.ndarray,
             ego_speed: float, now: float,
             geometry: RoadGeometry | None = None) -> SceneObservation:
    observation = self._candidate_observation(detections, image_bgr, ego_speed, now, geometry)
    if observation.event == SceneEvent.NONE:
      self.candidate = SceneEvent.NONE
      self.candidate_count = 0
      return observation

    if observation.event == self.candidate:
      self.candidate_count += 1
    else:
      self.candidate = observation.event
      self.candidate_count = 1

    if observation.event.value.startswith("trafficLight") and self.candidate_count < self.CONFIRMATIONS:
      return SceneObservation(
        SceneEvent.NONE,
        observation.confidence,
        False,
        reason=observation.reason,
      )

    ready = now - self.last_spoken_at[observation.event] >= self.EVENT_COOLDOWN
    if observation.voice_eligible and ready:
      self.last_spoken_at[observation.event] = now
      return observation
    return SceneObservation(
      event=observation.event,
      confidence=observation.confidence,
      voice_eligible=False,
      reason=observation.reason,
      track_id=observation.track_id,
      distance=observation.distance,
      path_offset=observation.path_offset,
      lateral_speed=observation.lateral_speed,
    )
