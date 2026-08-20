import numpy as np
import pytest

from openpilot.selfdrive.road_observer.speed_limit import (
  SpeedSignCandidate,
  SpeedLimitDetection,
  SpeedLimitTracker,
  classify_speed_limit_candidates,
  decode_speed_sign_yolov8,
  project_speed_limit_detections,
  remap_speed_sign_candidates,
)


def raw_model_output(class_id: int = 11, logit: float = 20.0) -> np.ndarray:
  output = np.full((1, 79, 1344), -20.0, dtype=np.float32)
  anchor_index = 10 * 32 + 10
  for side in range(4):
    output[0, side * 16 + 1, anchor_index] = 20.0
  output[0, 64 + class_id, anchor_index] = logit
  return output


def detection(limit_kph: int, score: float, size: float, center_x: float = 0.7) -> SpeedLimitDetection:
  return SpeedLimitDetection(
    limit_kph=limit_kph,
    score=score,
    bbox=(center_x - size / 2, 0.3 - size / 2, center_x + size / 2, 0.3 + size / 2),
  )


def test_decode_speed_sign_yolov8_rejects_unexpected_shape():
  with pytest.raises(ValueError):
    decode_speed_sign_yolov8(np.zeros((2, 79, 1344), dtype=np.float32))


def test_decode_speed_sign_yolov8_raw_head():
  candidates = decode_speed_sign_yolov8(raw_model_output())

  assert len(candidates) == 1
  assert candidates[0].score == pytest.approx(1.0)
  assert candidates[0].bbox == pytest.approx((76 / 256, 76 / 256, 92 / 256, 92 / 256))


def test_speed_limit_boxes_map_from_letterbox_to_camera():
  canvas = SpeedSignCandidate(0.8, (0.5, 0.2, 0.6, 0.3))

  remapped = remap_speed_sign_candidates([canvas], image_width=256, image_height=144)

  assert remapped[0].bbox == pytest.approx((0.5, 0.3555556, 0.6, 0.5333333))


def test_speed_limit_boxes_map_from_right_side_roi():
  canvas = SpeedSignCandidate(0.8, (0.0, 0.0, 1.0, 0.5))

  remapped = remap_speed_sign_candidates(
    [canvas],
    image_width=256,
    image_height=192,
    image_roi=(0.5, 0.0, 1.0, 0.6),
  )

  assert remapped[0].bbox == pytest.approx((0.5, 0.0, 1.0, 0.4))


def test_project_speed_limit_detection_to_full_image():
  detection = SpeedLimitDetection(70, 0.8, (0.0, 0.0, 1.0, 0.5))

  projected = project_speed_limit_detections([detection])

  assert projected[0].bbox == pytest.approx((0.5, 0.0, 1.0, 0.3))


def test_classifier_reads_candidate_and_combines_confidence():
  image = np.full((100, 200, 3), (20, 40, 220), dtype=np.uint8)
  candidate = SpeedSignCandidate(0.64, (0.4, 0.2, 0.6, 0.6))
  received = []

  def infer(crop):
    received.append(crop)
    output = np.zeros(43, dtype=np.float32)
    output[4] = 0.81
    return output

  detections = classify_speed_limit_candidates(image, [candidate], infer)

  assert len(detections) == 1
  assert detections[0].limit_kph == 70
  assert detections[0].score == pytest.approx(0.72)
  assert received[0].shape == (30, 30, 3)
  assert received[0][15, 15] == pytest.approx(np.asarray((20, 40, 220)) / 255, abs=0.01)


def test_classifier_rejects_non_speed_class():
  output = np.zeros(43, dtype=np.float32)
  output[14] = 0.9

  detections = classify_speed_limit_candidates(
    np.zeros((100, 100, 3), dtype=np.uint8),
    [SpeedSignCandidate(0.8, (0.2, 0.2, 0.5, 0.5))],
    lambda _crop: output,
  )

  assert detections == []


def test_tracker_requires_multiple_approaching_frames():
  tracker = SpeedLimitTracker()

  assert not tracker.update([detection(70, 0.25, 0.025)], 0.0).confirmed
  observation = tracker.update([detection(70, 0.35, 0.040)], 0.5)

  assert observation.confirmed
  assert observation.announce
  assert observation.limit_kph == 70
  assert observation.reason == "multiFrameAgreement"


def test_tracker_rejects_two_weak_readings():
  tracker = SpeedLimitTracker()

  tracker.update([detection(70, 0.15, 0.025)], 0.0)
  observation = tracker.update([detection(70, 0.20, 0.040)], 0.5)

  assert not observation.confirmed
  assert not observation.announce


def test_tracker_uses_later_high_confidence_readings():
  tracker = SpeedLimitTracker()
  samples = (
    (0.0, 50, 0.11, 0.020),
    (0.5, 70, 0.19, 0.026),
    (1.0, 80, 0.20, 0.035),
    (1.5, 80, 0.65, 0.050),
    (2.0, 80, 0.60, 0.060),
  )

  observation = None
  for now, limit, score, size in samples:
    observation = tracker.update([detection(limit, score, size)], now)

  assert observation is not None
  assert observation.confirmed
  assert observation.limit_kph == 80


def test_tracker_rejects_single_frame_false_positive():
  tracker = SpeedLimitTracker()

  observation = tracker.update([detection(80, 0.9, 0.06)], 0.0)

  assert not observation.confirmed
  assert not observation.announce


def test_tracker_accepts_perspective_narrowed_sign_boxes():
  tracker = SpeedLimitTracker()
  observations = []
  for index, height in enumerate((0.04, 0.055, 0.075)):
    observations.append(tracker.update([
      SpeedLimitDetection(
        limit_kph=70,
        score=0.3,
        bbox=(0.55, 0.25, 0.55 + height * 0.35, 0.25 + height),
      ),
    ], index * 0.5))

  assert observations[-1].confirmed
  assert observations[-1].limit_kph == 70


def test_tracker_rejects_conflicting_readings():
  tracker = SpeedLimitTracker()
  readings = ((20, 0.4), (50, 0.4), (80, 0.4))
  for index, (limit, score) in enumerate(readings):
    observation = tracker.update(
      [detection(limit, score, 0.025 + index * 0.01)],
      index * 0.5,
    )

  assert not observation.confirmed
  assert observation.reason == "conflictingReadings"


def test_single_late_misread_does_not_replace_confirmed_limit():
  tracker = SpeedLimitTracker()
  for index, size in enumerate((0.025, 0.035, 0.050)):
    confirmed = tracker.update([detection(70, 0.25, size)], index * 0.5)
  assert confirmed.confirmed
  assert confirmed.limit_kph == 70

  misread = tracker.update([detection(100, 0.8, 0.08)], 1.5)

  assert not misread.confirmed
  assert not misread.announce


def test_same_limit_does_not_repeat_immediately():
  tracker = SpeedLimitTracker()
  for index in range(3):
    first = tracker.update([detection(50, 0.5, 0.025 + index * 0.01)], index * 0.5)
  assert first.announce
  tracker.mark_announced(first, 1.0)

  tracker.tracks.clear()
  for index in range(3):
    repeated = tracker.update([detection(50, 0.5, 0.025 + index * 0.01)], 5.0 + index * 0.5)

  assert repeated.confirmed
  assert not repeated.announce


def test_unannounced_limit_remains_eligible_after_other_alert():
  tracker = SpeedLimitTracker()
  for index in range(3):
    observation = tracker.update([detection(70, 0.5, 0.025 + index * 0.01)], index * 0.5)
  assert observation.announce

  next_observation = tracker.update([detection(70, 0.5, 0.060)], 1.5)

  assert next_observation.announce
