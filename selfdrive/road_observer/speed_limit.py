import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.road_observer.perception import _bbox_center_distance, _bbox_iou, _nms


SPEED_LIMIT_MODEL_SIZE = 256
SPEED_LIMIT_CLASSIFIER_SIZE = 30
SPEED_SIGN_ROI = (0.5, 0.0, 1.0, 0.6)
SPEED_SIGN_DETECTOR_CLASS_IDS = tuple(range(2, 14))
SPEED_LIMIT_CLASS_VALUES = {
  0: 20,
  1: 30,
  2: 50,
  3: 60,
  4: 70,
  5: 80,
  7: 100,
  8: 120,
}
VALID_SPEED_LIMITS = frozenset(SPEED_LIMIT_CLASS_VALUES.values())
YOLOV8_DETECTOR_CLASS_COUNT = 15
YOLOV8_REG_MAX = 16


@dataclass(frozen=True)
class SpeedSignCandidate:
  score: float
  bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SpeedLimitDetection:
  limit_kph: int
  score: float
  bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SpeedLimitObservation:
  limit_kph: int | None = None
  confidence: float = 0.0
  confirmed: bool = False
  announce: bool = False
  track_id: int | None = None
  reason: str = "none"


@dataclass
class _SpeedLimitTrack:
  track_id: int
  bbox: tuple[float, float, float, float]
  last_seen: float
  history: deque[tuple[float, int, float, float]]


def decode_speed_sign_yolov8(
  output: np.ndarray,
  model_size: int = SPEED_LIMIT_MODEL_SIZE,
  score_threshold: float = 0.03,
  nms_threshold: float = 0.45,
) -> list[SpeedSignCandidate]:
  predictions = np.asarray(output, dtype=np.float32)
  if predictions.ndim == 3:
    if predictions.shape[0] != 1:
      raise ValueError(f"unexpected YOLOv8 batch size: {predictions.shape[0]}")
    predictions = predictions[0]
  if predictions.ndim != 2:
    raise ValueError(f"unexpected YOLOv8 output shape: {predictions.shape}")
  expected_rows = 4 * YOLOV8_REG_MAX + YOLOV8_DETECTOR_CLASS_COUNT
  if predictions.shape[0] == expected_rows:
    pass
  elif predictions.shape[1] == expected_rows:
    predictions = predictions.T
  else:
    raise ValueError(f"unexpected YOLOv8 output shape: {predictions.shape}")

  box_logits = predictions[:4 * YOLOV8_REG_MAX].reshape((4, YOLOV8_REG_MAX, -1))
  box_logits -= np.max(box_logits, axis=1, keepdims=True)
  box_probabilities = np.exp(box_logits)
  box_probabilities /= np.sum(box_probabilities, axis=1, keepdims=True)
  distances = np.sum(
    box_probabilities * np.arange(YOLOV8_REG_MAX, dtype=np.float32)[None, :, None],
    axis=1,
  )

  anchors = []
  strides = []
  for stride in (8, 16, 32):
    grid_size = model_size // stride
    grid_y, grid_x = np.meshgrid(
      np.arange(grid_size, dtype=np.float32),
      np.arange(grid_size, dtype=np.float32),
      indexing="ij",
    )
    anchors.append(np.stack((grid_x, grid_y), axis=0).reshape((2, -1)) + 0.5)
    strides.append(np.full(grid_size * grid_size, stride, dtype=np.float32))
  anchor_points = np.concatenate(anchors, axis=1)
  stride_values = np.concatenate(strides)
  if distances.shape[1] != anchor_points.shape[1]:
    raise ValueError(f"unexpected YOLOv8 raw output columns: {distances.shape[1]}")

  left = anchor_points[0] - distances[0]
  top = anchor_points[1] - distances[1]
  right = anchor_points[0] + distances[2]
  bottom = anchor_points[1] + distances[3]
  xywh = np.stack((
    (left + right) * 0.5,
    (top + bottom) * 0.5,
    right - left,
    bottom - top,
  ), axis=1) * stride_values[:, None]
  class_scores = 1.0 / (
    1.0 + np.exp(-np.clip(predictions[4 * YOLOV8_REG_MAX:], -30.0, 30.0))
  )
  scores = np.max(class_scores[np.asarray(SPEED_SIGN_DETECTOR_CLASS_IDS)], axis=0)
  selected = scores >= score_threshold
  if not np.any(selected):
    return []

  selected_xywh = xywh[selected]
  boxes = np.empty_like(selected_xywh)
  boxes[:, 0] = selected_xywh[:, 0] - selected_xywh[:, 2] / 2
  boxes[:, 1] = selected_xywh[:, 1] - selected_xywh[:, 3] / 2
  boxes[:, 2] = selected_xywh[:, 0] + selected_xywh[:, 2] / 2
  boxes[:, 3] = selected_xywh[:, 1] + selected_xywh[:, 3] / 2
  boxes = np.clip(boxes / model_size, 0.0, 1.0)
  selected_scores = scores[selected]

  candidates = []
  for index in _nms(boxes, selected_scores, nms_threshold):
    bbox = tuple(float(value) for value in boxes[index])
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
      continue
    candidates.append(SpeedSignCandidate(
      score=float(selected_scores[index]),
      bbox=bbox,
    ))
  return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def remap_speed_sign_candidates(
  candidates: list[SpeedSignCandidate],
  image_width: int,
  image_height: int,
  model_size: int = SPEED_LIMIT_MODEL_SIZE,
  image_roi: tuple[float, float, float, float] | None = None,
) -> list[SpeedSignCandidate]:
  x_scale = model_size / image_width
  y_scale = model_size / image_height
  roi_x1, roi_y1, roi_x2, roi_y2 = image_roi or (0.0, 0.0, 1.0, 1.0)
  roi_width = roi_x2 - roi_x1
  roi_height = roi_y2 - roi_y1
  remapped = []
  for candidate in candidates:
    x1, y1, x2, y2 = candidate.bbox
    bbox = (
      float(np.clip(roi_x1 + x1 * x_scale * roi_width, 0.0, 1.0)),
      float(np.clip(roi_y1 + y1 * y_scale * roi_height, 0.0, 1.0)),
      float(np.clip(roi_x1 + x2 * x_scale * roi_width, 0.0, 1.0)),
      float(np.clip(roi_y1 + y2 * y_scale * roi_height, 0.0, 1.0)),
    )
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
      remapped.append(SpeedSignCandidate(candidate.score, bbox))
  return remapped


def _resize_image_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
  if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 2 or image.shape[1] < 2:
    raise ValueError(f"unexpected speed sign crop shape: {image.shape}")
  ys = np.linspace(0.0, image.shape[0] - 1, height, dtype=np.float32)
  xs = np.linspace(0.0, image.shape[1] - 1, width, dtype=np.float32)
  y0 = np.floor(ys).astype(np.int32)
  x0 = np.floor(xs).astype(np.int32)
  y1 = np.minimum(y0 + 1, image.shape[0] - 1)
  x1 = np.minimum(x0 + 1, image.shape[1] - 1)
  wy = (ys - y0)[:, None, None]
  wx = (xs - x0)[None, :, None]
  top = image[y0[:, None], x0[None, :]] * (1.0 - wx) + image[y0[:, None], x1[None, :]] * wx
  bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx) + image[y1[:, None], x1[None, :]] * wx
  return np.ascontiguousarray(top * (1.0 - wy) + bottom * wy, dtype=np.float32)


def classify_speed_limit_candidates(
  image_rgb: np.ndarray,
  candidates: list[SpeedSignCandidate],
  infer,
  classifier_threshold: float = 0.18,
  crop_padding: float = 0.08,
) -> list[SpeedLimitDetection]:
  height, width = image_rgb.shape[:2]
  detections = []
  for candidate in candidates[:6]:
    x1, y1, x2, y2 = candidate.bbox
    padding = crop_padding * max(x2 - x1, y2 - y1)
    left = max(0, int((x1 - padding) * width))
    top = max(0, int((y1 - padding) * height))
    right = min(width, int(math.ceil((x2 + padding) * width)))
    bottom = min(height, int(math.ceil((y2 + padding) * height)))
    if right - left < 4 or bottom - top < 4:
      continue

    crop = _resize_image_bilinear(
      image_rgb[top:bottom, left:right],
      SPEED_LIMIT_CLASSIFIER_SIZE,
      SPEED_LIMIT_CLASSIFIER_SIZE,
    ) * (1.0 / 255.0)
    probabilities = np.asarray(infer(crop), dtype=np.float32).reshape(-1)
    if probabilities.size <= max(SPEED_LIMIT_CLASS_VALUES):
      raise ValueError(f"unexpected speed classifier output shape: {probabilities.shape}")
    class_id = int(np.argmax(probabilities))
    classifier_score = float(probabilities[class_id])
    if class_id not in SPEED_LIMIT_CLASS_VALUES or classifier_score < classifier_threshold:
      continue
    combined_score = math.sqrt(max(0.0, candidate.score * classifier_score))
    if combined_score < 0.08:
      continue
    detections.append(SpeedLimitDetection(
      limit_kph=SPEED_LIMIT_CLASS_VALUES[class_id],
      score=combined_score,
      bbox=candidate.bbox,
    ))
  return sorted(detections, key=lambda detection: detection.score, reverse=True)


def project_speed_limit_detections(
  detections: list[SpeedLimitDetection],
  image_roi: tuple[float, float, float, float] = SPEED_SIGN_ROI,
) -> list[SpeedLimitDetection]:
  roi_x1, roi_y1, roi_x2, roi_y2 = image_roi
  roi_width = roi_x2 - roi_x1
  roi_height = roi_y2 - roi_y1
  return [
    SpeedLimitDetection(
      detection.limit_kph,
      detection.score,
      (
        roi_x1 + detection.bbox[0] * roi_width,
        roi_y1 + detection.bbox[1] * roi_height,
        roi_x1 + detection.bbox[2] * roi_width,
        roi_y1 + detection.bbox[3] * roi_height,
      ),
    )
    for detection in detections
  ]


class SpeedLimitTracker:
  MAX_TRACK_AGE = 1.25
  HISTORY_SECONDS = 3.0
  MIN_CONFIRMATIONS = 2
  MIN_CONFIRMATION_SPAN = 0.4
  MIN_VOTE_SHARE = 0.65
  MIN_PEAK_SCORE = 0.25
  MIN_BOX_HEIGHT = 0.018
  MIN_GROWTH = 1.15
  REANNOUNCE_SECONDS = 30.0

  def __init__(self):
    self.tracks: dict[int, _SpeedLimitTrack] = {}
    self.next_track_id = 1
    self.last_announced_limit: int | None = None
    self.last_announced_at = -math.inf

  @staticmethod
  def _valid_detection(detection: SpeedLimitDetection) -> bool:
    x1, y1, x2, y2 = detection.bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) * 0.5
    aspect = width / height if height > 0.0 else 0.0
    return (
      detection.limit_kph in VALID_SPEED_LIMITS
      and math.isfinite(detection.score)
      and detection.score >= 0.08
      and 0.04 <= center_x <= 0.96
      and 0.0 <= y1 < y2 <= 0.92
      and 0.012 <= height <= 0.35
      and 0.28 <= aspect <= 1.8
    )

  def _match(self, detection: SpeedLimitDetection, available: set[int]) -> int | None:
    best_id = None
    best_cost = math.inf
    height = detection.bbox[3] - detection.bbox[1]
    for track_id in available:
      track = self.tracks[track_id]
      iou = _bbox_iou(track.bbox, detection.bbox)
      center_distance = _bbox_center_distance(track.bbox, detection.bbox)
      if iou < 0.03 and center_distance > max(0.06, height * 1.8):
        continue
      cost = center_distance + 0.12 * (1.0 - iou)
      if cost < best_cost:
        best_id = track_id
        best_cost = cost
    return best_id

  @staticmethod
  def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

  def _observation(self, track: _SpeedLimitTrack, now: float) -> SpeedLimitObservation:
    history = track.history
    if len(history) < self.MIN_CONFIRMATIONS:
      return SpeedLimitObservation(track_id=track.track_id, reason="awaitingFrames")
    if history[-1][0] - history[0][0] < self.MIN_CONFIRMATION_SPAN:
      return SpeedLimitObservation(track_id=track.track_id, reason="awaitingTime")

    votes = dict.fromkeys(VALID_SPEED_LIMITS, 0.0)
    for _, limit, score, area in history:
      votes[limit] += score * math.sqrt(max(area, 1e-6))
    limit_kph, best_vote = max(votes.items(), key=lambda item: item[1])
    total_vote = sum(votes.values())
    vote_share = best_vote / total_vote if total_vote > 0.0 else 0.0

    winning_samples = [sample for sample in history if sample[1] == limit_kph]
    first_area = max(winning_samples[0][3], 1e-6)
    peak_area = max(sample[3] for sample in winning_samples)
    peak_score = max(sample[2] for sample in winning_samples)
    latest_height = track.bbox[3] - track.bbox[1]
    approaching = peak_area >= first_area * self.MIN_GROWTH
    confirmed = (
      len(winning_samples) >= self.MIN_CONFIRMATIONS
      and winning_samples[-1][0] - winning_samples[0][0] >= self.MIN_CONFIRMATION_SPAN
      and history[-1][1] == limit_kph
      and vote_share >= self.MIN_VOTE_SHARE
      and peak_score >= self.MIN_PEAK_SCORE
      and latest_height >= self.MIN_BOX_HEIGHT
      and approaching
    )
    if not confirmed:
      return SpeedLimitObservation(
        limit_kph=limit_kph,
        confidence=vote_share,
        track_id=track.track_id,
        reason="conflictingReadings",
      )

    confidence = float(np.clip(0.7 * vote_share + 0.3 * min(1.0, peak_score / 0.5), 0.0, 1.0))
    announce = (
      limit_kph != self.last_announced_limit
      or now - self.last_announced_at >= self.REANNOUNCE_SECONDS
    )
    return SpeedLimitObservation(
      limit_kph=limit_kph,
      confidence=confidence,
      confirmed=True,
      announce=announce,
      track_id=track.track_id,
      reason="multiFrameAgreement",
    )

  def mark_announced(self, observation: SpeedLimitObservation, now: float) -> None:
    if observation.confirmed and observation.limit_kph in VALID_SPEED_LIMITS:
      self.last_announced_limit = observation.limit_kph
      self.last_announced_at = now

  def update(self, detections: list[SpeedLimitDetection], now: float) -> SpeedLimitObservation:
    self.tracks = {
      track_id: track
      for track_id, track in self.tracks.items()
      if now - track.last_seen <= self.MAX_TRACK_AGE
    }
    available = set(self.tracks)
    observations = []

    for detection in detections:
      if not self._valid_detection(detection):
        continue
      track_id = self._match(detection, available)
      if track_id is None:
        track_id = self.next_track_id
        self.next_track_id += 1
        track = _SpeedLimitTrack(
          track_id=track_id,
          bbox=detection.bbox,
          last_seen=now,
          history=deque(),
        )
        self.tracks[track_id] = track
      else:
        track = self.tracks[track_id]
        available.remove(track_id)

      track.bbox = detection.bbox
      track.last_seen = now
      track.history.append((now, detection.limit_kph, detection.score, self._area(detection.bbox)))
      while track.history and now - track.history[0][0] > self.HISTORY_SECONDS:
        track.history.popleft()
      observations.append(self._observation(track, now))

    if not observations:
      return SpeedLimitObservation()
    return max(
      observations,
      key=lambda observation: (
        observation.confirmed,
        observation.announce,
        observation.confidence,
      ),
    )
