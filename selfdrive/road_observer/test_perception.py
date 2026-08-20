import numpy as np
import pytest

from openpilot.selfdrive.road_observer.perception import (
  COCO_PERSON,
  COCO_CAR,
  COCO_TRAFFIC_LIGHT,
  Detection,
  MODEL_SIZE,
  PERCEPTION_EVENT_PARAM,
  PERCEPTION_PROMPT_MAP,
  RoadGeometry,
  SceneEvent,
  SceneInterpreter,
  SceneObservation,
  classify_driver_gaze,
  classify_traffic_light,
  decode_yolox,
  get_perception_alert,
  get_perception_prompt,
  preprocess_nv12,
  remap_detections,
)
from openpilot.common.transformations.camera import get_view_frame_from_calib_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.road_observer.roadperceptionmodeld import camera_calibration_rpy, serialize_observation
from openpilot.selfdrive.road_observer.speed_limit import SpeedLimitObservation


class FakeVisionBuf:
  width = 4
  height = 4
  stride = 4
  uv_offset = 16

  def __init__(self):
    y = np.full(self.uv_offset, 82, dtype=np.uint8)
    uv = np.tile(np.array([90, 240], dtype=np.uint8), 16)
    self.data = np.concatenate((y, uv)).tobytes()


CAMERA_SIZE = (1000, 600)
CAMERA_HEIGHT = 1.2
INTRINSICS = np.array([
  [1000.0, 0.0, 500.0],
  [0.0, 1000.0, 300.0],
  [0.0, 0.0, 1.0],
])


def straight_geometry() -> RoadGeometry:
  geometry = RoadGeometry.build(
    path_x=np.linspace(0.0, 100.0, 33),
    path_y=np.zeros(33),
    rpy=np.zeros(3),
    camera_height=CAMERA_HEIGHT,
    intrinsics=INTRINSICS,
    camera_size=CAMERA_SIZE,
  )
  assert geometry is not None
  return geometry


def road_user_detection(distance: float, lateral: float, class_id: int = COCO_PERSON,
                        score: float = 0.85) -> Detection:
  road_point = np.array([distance, lateral, 0.0, 1.0])
  extrinsics = get_view_frame_from_calib_frame(0.0, 0.0, 0.0, CAMERA_HEIGHT)
  projected = INTRINSICS @ extrinsics @ road_point
  u = projected[0] / projected[2] / CAMERA_SIZE[0]
  bottom = projected[1] / projected[2] / CAMERA_SIZE[1]
  box_height = min(0.45, 2.2 / distance)
  box_width = box_height * 0.4
  return Detection(class_id, score, (
    u - box_width / 2,
    bottom - box_height,
    u + box_width / 2,
    bottom,
  ))


def test_preprocess_nv12_returns_padded_bgr_input():
  model_input, image = preprocess_nv12(FakeVisionBuf(), model_size=8)

  assert model_input.shape == (3, 8, 8)
  assert image.shape == (8, 8, 3)
  assert image[0, 0, 2] > image[0, 0, 1]
  assert image[0, 0, 2] > image[0, 0, 0]


def test_preprocess_nv12_crops_before_resizing():
  model_input, image = preprocess_nv12(
    FakeVisionBuf(),
    model_size=8,
    image_roi=(0.5, 0.0, 1.0, 1.0),
  )

  assert model_input.shape == (3, 8, 8)
  assert image.shape == (8, 4, 3)
  assert np.all(model_input[:, :, 4:] == 114)


def test_decode_yolox_keeps_relevant_class():
  rows = sum((MODEL_SIZE // stride) ** 2 for stride in (8, 16, 32))
  output = np.zeros((1, rows, 85), dtype=np.float32)
  output[0, 0, 4] = 0.9
  output[0, 0, 5 + COCO_PERSON] = 0.9

  detections = decode_yolox(output)

  assert len(detections) == 1
  assert detections[0].class_id == COCO_PERSON
  assert detections[0].score > 0.8


def test_comma_4_letterbox_box_maps_to_camera_image():
  canvas_detection = Detection(COCO_PERSON, 0.8, (0.42, 0.20, 0.58, 0.54))
  detections = remap_detections([canvas_detection], 320, 180)

  assert detections[0].bbox[3] == pytest.approx(0.96, abs=0.002)


def test_ground_projection_uses_calibration_and_curved_future_path():
  path_x = np.linspace(0.0, 100.0, 33)
  geometry = RoadGeometry.build(
    path_x=path_x,
    path_y=path_x * 0.05,
    rpy=np.zeros(3),
    camera_height=CAMERA_HEIGHT,
    intrinsics=INTRINSICS,
    camera_size=CAMERA_SIZE,
  )
  assert geometry is not None
  detection = road_user_detection(20.0, 2.1)

  position = geometry.ground_position(detection)

  assert position is not None
  assert position[0] == pytest.approx(20.0)
  assert position[1] == pytest.approx(1.1)


def test_classify_red_traffic_light():
  image = np.zeros((100, 100, 3), dtype=np.uint8)
  image[10:30, 45:55, 2] = 255

  color, confidence = classify_traffic_light(image, (0.4, 0.05, 0.6, 0.35))

  assert color == "red"
  assert confidence > 0.5


def test_pedestrian_requires_calibrated_path_and_two_hits():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  first = interpreter.update([road_user_detection(20.0, 0.2)], image, 10.0, 0.0, geometry)
  observation = interpreter.update([road_user_detection(15.0, 0.2)], image, 10.0, 0.5, geometry)

  assert first.event == SceneEvent.PEDESTRIAN_RISK
  assert not first.voice_eligible
  assert observation.event == SceneEvent.PEDESTRIAN_RISK
  assert observation.voice_eligible
  assert observation.reason == "pathOccupied"
  assert observation.distance == pytest.approx(15.0)

  uncalibrated = SceneInterpreter()
  for index in range(3):
    observation = uncalibrated.update(
      [road_user_detection(20.0, 0.2)],
      image,
      10.0,
      index * 0.5,
      geometry=None,
    )
  assert observation.event == SceneEvent.NONE


def test_crossing_pedestrian_uses_tracked_lateral_motion():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(distance, lateral)],
      image,
      10.0,
      index * 0.5,
      geometry,
    )
    for index, (distance, lateral) in enumerate(((30.0, 3.0), (25.0, 2.6), (20.0, 2.2)))
  ]

  assert observations[-1].event == SceneEvent.PEDESTRIAN_RISK
  assert observations[-1].voice_eligible
  assert observations[-1].reason == "crossingPredicted"
  assert observations[-1].lateral_speed == pytest.approx(-0.8)


def test_static_pedestrian_outside_path_does_not_alert():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(distance, 3.0)],
      image,
      10.0,
      index * 0.5,
      geometry,
    )
    for index, distance in enumerate((30.0, 25.0, 20.0))
  ]

  assert all(observation.event == SceneEvent.NONE for observation in observations)


def test_crossing_vehicle_warns_from_wide_camera_during_turn():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(14.0, lateral, class_id=COCO_CAR)],
      image,
      5.0,
      index * 0.5,
      geometry,
      turning=True,
      camera_source="wide",
    )
    for index, lateral in enumerate((3.6, 3.1, 2.6))
  ]

  assert observations[-1].event == SceneEvent.CROSS_TRAFFIC_RISK
  assert observations[-1].voice_eligible
  assert observations[-1].reason == "movingPathConflict"
  assert observations[-1].side == "left"


def test_crossing_vehicle_remains_shadow_only_without_wide_camera():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(14.0, lateral, class_id=COCO_CAR)],
      image,
      5.0,
      index * 0.5,
      geometry,
      turning=True,
    )
    for index, lateral in enumerate((3.6, 3.1, 2.6))
  ]

  assert observations[-1].event == SceneEvent.CROSS_TRAFFIC_RISK
  assert not observations[-1].voice_eligible


def test_unchecked_stationary_junction_vehicle_warns_after_four_wide_frames():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(12.0, 4.0, class_id=COCO_CAR, score=0.8)],
      image,
      3.0,
      index * 0.5,
      geometry,
      turning=True,
      turn_signal=True,
      camera_source="wide",
      driver_gaze_side="right",
    )
    for index in range(4)
  ]

  assert observations[-1].event == SceneEvent.CROSS_TRAFFIC_RISK
  assert observations[-1].voice_eligible
  assert observations[-1].reason == "uncheckedJunctionVehicle"
  assert observations[-1].side == "left"


def test_recent_side_check_suppresses_stationary_junction_reminder():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(12.0, 4.0, class_id=COCO_CAR, score=0.8)],
      image,
      3.0,
      index * 0.5,
      geometry,
      turning=True,
      turn_signal=True,
      camera_source="wide",
      driver_gaze_side="left",
    )
    for index in range(4)
  ]

  assert all(observation.event == SceneEvent.NONE for observation in observations)


def test_vehicle_outside_path_is_ignored_without_crossing_motion():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  geometry = straight_geometry()

  observations = [
    interpreter.update(
      [road_user_detection(distance, 3.5, class_id=COCO_CAR)],
      image,
      5.0,
      index * 0.5,
      geometry,
      turning=True,
    )
    for index, distance in enumerate((30.0, 27.0, 24.0))
  ]

  assert all(observation.event == SceneEvent.NONE for observation in observations)


def test_traffic_light_is_shadow_only_even_after_confirmation():
  interpreter = SceneInterpreter()
  red_image = np.zeros((416, 416, 3), dtype=np.uint8)
  red_image[40:90, 190:220, 2] = 255
  light = Detection(COCO_TRAFFIC_LIGHT, 0.9, (190 / 416, 40 / 416, 220 / 416, 90 / 416))

  observations = [
    interpreter.update([light], red_image, 5.0, float(index))
    for index in range(3)
  ]

  assert observations[-1].event == SceneEvent.TRAFFIC_LIGHT_RED
  assert not observations[-1].voice_eligible
  assert observations[-1].reason == "shadowOnlyNoLaneAssociation"


def test_perception_prompt_parser():
  assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":true}') == 8
  alert = get_perception_alert(
    b'{"event":"crossTrafficRisk","side":"right","voice":true,"eventId":123,"confidence":0.82}',
  )
  assert alert.prompt == 17
  assert alert.event_id == 123
  assert alert.confidence == pytest.approx(0.82)


def test_speed_limit_prompt_parser_carries_limit_and_overspeed():
  alert = get_perception_alert(
    b'{"event":"speedLimit","voice":true,"eventId":456,"confidence":0.88,"speedLimit":{"limitKph":70,"overspeed":true}}',
  )

  assert alert.prompt == 18
  assert alert.speed_limit_kph == 70
  assert alert.overspeed
  assert get_perception_prompt(b'{"event":"trafficLightRed","voice":true}') == 0
  assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":false}') == 0
  assert get_perception_prompt(b'not-json') == 0


def test_serialized_junction_alert_keeps_radio_event_and_direction():
  observation = SceneObservation(
    SceneEvent.CROSS_TRAFFIC_RISK,
    0.84,
    True,
    reason="movingPathConflict",
    side="right",
  )

  raw = serialize_observation(42, 0.035, observation, [], 987, "wide")
  alert = get_perception_alert(raw)

  assert alert.prompt == 17
  assert alert.event_id == 987
  assert alert.confidence == pytest.approx(0.84)


def test_serialized_speed_limit_alert_keeps_dynamic_value():
  observation = SceneObservation(SceneEvent.NONE, 0.0, False)
  speed_limit = SpeedLimitObservation(
    limit_kph=80,
    confidence=0.91,
    confirmed=True,
    announce=True,
    track_id=3,
    reason="multiFrameAgreement",
  )

  raw = serialize_observation(
    42,
    0.12,
    observation,
    [],
    0,
    "road",
    speed_limit=speed_limit,
    speed_limit_event_id=988,
    speed_limit_overspeed=True,
  )
  alert = get_perception_alert(raw)

  assert alert.prompt == 18
  assert alert.event_id == 988
  assert alert.confidence == pytest.approx(0.91)
  assert alert.speed_limit_kph == 80
  assert alert.overspeed


def test_wide_camera_calibration_composes_camera_and_road_rotations():
  road_rpy = np.radians([0.2, -1.1, 0.7])
  wide_rpy = np.radians([-0.4, 0.8, -1.5])

  combined = camera_calibration_rpy(road_rpy, wide_rpy, "wide")

  assert combined is not None
  np.testing.assert_allclose(
    rot_from_euler(combined),
    rot_from_euler(wide_rpy) @ rot_from_euler(road_rpy),
    atol=1e-9,
  )
  np.testing.assert_allclose(
    camera_calibration_rpy(road_rpy, (), "road"),
    road_rpy,
  )
  assert camera_calibration_rpy(road_rpy, (), "wide") is None


def test_driver_gaze_requires_confident_side_look():
  assert classify_driver_gaze(np.radians(-22.0), True, np.radians(5.0)) == "left"
  assert classify_driver_gaze(np.radians(22.0), True, np.radians(5.0)) == "right"
  assert classify_driver_gaze(np.radians(10.0), True, np.radians(5.0)) is None
  assert classify_driver_gaze(np.radians(22.0), False, np.radians(5.0)) is None
  assert classify_driver_gaze(np.radians(22.0), True, np.radians(20.0)) is None


def test_every_spoken_perception_event_has_a_setting():
  spoken_events = {
    event for event in SceneEvent
    if event.value in PERCEPTION_PROMPT_MAP or event == SceneEvent.CROSS_TRAFFIC_RISK
  }
  assert set(PERCEPTION_EVENT_PARAM) == spoken_events
