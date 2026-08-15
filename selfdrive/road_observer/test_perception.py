import numpy as np
import pytest

from openpilot.selfdrive.road_observer.perception import (
  COCO_PERSON,
  COCO_TRAFFIC_LIGHT,
  Detection,
  MODEL_SIZE,
  PERCEPTION_EVENT_PARAM,
  PERCEPTION_PROMPT_MAP,
  RoadGeometry,
  SceneEvent,
  SceneInterpreter,
  classify_traffic_light,
  decode_yolox,
  get_perception_prompt,
  preprocess_nv12,
  remap_detections,
)
from openpilot.common.transformations.camera import get_view_frame_from_calib_frame


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
  assert get_perception_prompt(b'{"event":"trafficLightRed","voice":true}') == 0
  assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":false}') == 0
  assert get_perception_prompt(b'not-json') == 0


def test_every_spoken_perception_event_has_a_setting():
  spoken_events = {
    event for event in SceneEvent
    if event.value in PERCEPTION_PROMPT_MAP
  }
  assert set(PERCEPTION_EVENT_PARAM) == spoken_events
