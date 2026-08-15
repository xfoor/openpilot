import numpy as np
import pytest

from openpilot.selfdrive.road_observer.perception import (
  COCO_PERSON,
  COCO_TRAFFIC_LIGHT,
  Detection,
  PERCEPTION_EVENT_PARAM,
  SceneEvent,
  SceneInterpreter,
  classify_traffic_light,
  decode_yolox,
  get_perception_prompt,
  preprocess_nv12,
  remap_detections,
)


class FakeVisionBuf:
  width = 4
  height = 4
  stride = 4
  uv_offset = 16

  def __init__(self):
    y = np.full(self.uv_offset, 82, dtype=np.uint8)
    uv = np.tile(np.array([90, 240], dtype=np.uint8), 16)
    self.data = np.concatenate((y, uv)).tobytes()


def test_preprocess_nv12_returns_padded_bgr_input():
  model_input, image = preprocess_nv12(FakeVisionBuf(), model_size=8)

  assert model_input.shape == (3, 8, 8)
  assert image.shape == (8, 8, 3)
  assert image[0, 0, 2] > image[0, 0, 1]
  assert image[0, 0, 2] > image[0, 0, 0]


def test_decode_yolox_keeps_relevant_class():
  output = np.zeros((1, 3549, 85), dtype=np.float32)
  output[0, 0, 4] = 0.9
  output[0, 0, 5 + COCO_PERSON] = 0.9

  detections = decode_yolox(output)

  assert len(detections) == 1
  assert detections[0].class_id == COCO_PERSON
  assert detections[0].score > 0.8


def test_comma_4_letterbox_box_maps_to_camera_corridor():
  canvas_detection = Detection(COCO_PERSON, 0.8, (0.42, 0.20, 0.58, 0.54))
  detections = remap_detections([canvas_detection], 416, 235)

  assert detections[0].bbox[3] == pytest.approx(0.956, abs=0.002)

  interpreter = SceneInterpreter()
  image = np.zeros((235, 416, 3), dtype=np.uint8)
  observations = [
    interpreter.update(detections, image, 10.0, index * 0.5)
    for index in range(3)
  ]

  assert observations[-1].event == SceneEvent.PEDESTRIAN_RISK
  assert observations[-1].voice_eligible


def test_classify_red_traffic_light():
  image = np.zeros((100, 100, 3), dtype=np.uint8)
  image[10:30, 45:55, 2] = 255

  color, confidence = classify_traffic_light(image, (0.4, 0.05, 0.6, 0.35))

  assert color == "red"
  assert confidence > 0.5


def test_pedestrian_requires_confirmation_and_corridor():
  interpreter = SceneInterpreter()
  image = np.zeros((416, 416, 3), dtype=np.uint8)
  pedestrian = Detection(COCO_PERSON, 0.8, (0.42, 0.25, 0.58, 0.85))

  assert interpreter.update([pedestrian], image, 10.0, 0.0).event == SceneEvent.NONE
  assert interpreter.update([pedestrian], image, 10.0, 0.5).event == SceneEvent.NONE
  observation = interpreter.update([pedestrian], image, 10.0, 1.0)

  assert observation.event == SceneEvent.PEDESTRIAN_RISK
  assert observation.voice_eligible


def test_green_voice_only_after_confirmed_red_while_stopped():
  interpreter = SceneInterpreter()
  red_image = np.zeros((416, 416, 3), dtype=np.uint8)
  green_image = red_image.copy()
  red_image[40:90, 190:220, 2] = 255
  green_image[40:90, 190:220, 1] = 255
  light = Detection(COCO_TRAFFIC_LIGHT, 0.9, (190 / 416, 40 / 416, 220 / 416, 90 / 416))

  for index in range(3):
    interpreter.update([light], red_image, 5.0, float(index))
  observations = [
    interpreter.update([light], green_image, 0.0, 3.0 + index)
    for index in range(3)
  ]

  assert observations[-1].event == SceneEvent.TRAFFIC_LIGHT_GREEN
  assert observations[-1].voice_eligible


def test_perception_prompt_parser():
  assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":true}') == 8
  assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":false}') == 0
  assert get_perception_prompt(b'not-json') == 0


def test_every_spoken_perception_event_has_a_setting():
  spoken_events = set(SceneEvent) - {SceneEvent.NONE}
  assert set(PERCEPTION_EVENT_PARAM) == spoken_events
