import time

import numpy as np

from cereal import car
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.road_observer.perception import get_perception_alert, get_perception_prompt
from openpilot.selfdrive.ui.soundd import (
  SELFDRIVE_STATE_TIMEOUT,
  Soundd,
  check_selfdrive_timeout_alert,
  observer_sound_list,
)

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class TestSoundd:
  def test_all_observer_sounds_load(self):
    soundd = Soundd()
    assert set(soundd.loaded_observer_sounds) == set(observer_sound_list)

  def test_check_selfdrive_timeout_alert(self):
    sm = SubMaster(['selfdriveState'])
    pm = PubMaster(['selfdriveState'])

    for _ in range(100):
      cs = messaging.new_message('selfdriveState')
      cs.selfdriveState.enabled = True

      pm.send("selfdriveState", cs)

      time.sleep(0.01)

      sm.update(0)

      assert not check_selfdrive_timeout_alert(sm)

    for _ in range(SELFDRIVE_STATE_TIMEOUT * 110):
      sm.update(0)
      time.sleep(0.01)

    assert check_selfdrive_timeout_alert(sm)

  def test_stock_alert_interrupts_observer_prompt(self):
    soundd = Soundd.__new__(Soundd)
    soundd.current_alert = AudibleAlert.none
    soundd.current_sound_frame = 0
    soundd.current_observer_prompt = 1
    soundd.current_observer_sound_frame = 0
    soundd.current_observer_event_id = 0
    soundd.pending_observer_prompt = 0
    soundd.pending_observer_event_id = 0
    soundd.current_volume = 1.0
    soundd.ramp_start_volume = 0.1
    soundd.ramp_start_time = 0.0
    soundd.loaded_sounds = {AudibleAlert.warningImmediate: np.ones(8, dtype=np.float32)}
    soundd.loaded_observer_sounds = {1: np.full(8, 0.5, dtype=np.float32)}

    soundd.update_alert(AudibleAlert.warningImmediate)

    assert soundd.current_observer_prompt == 0
    assert np.all(soundd.get_sound_data(4) == 1.0)

  def test_observer_prompt_plays_once(self):
    soundd = Soundd.__new__(Soundd)
    soundd.current_alert = AudibleAlert.none
    soundd.current_sound_frame = 0
    soundd.current_observer_prompt = 1
    soundd.current_observer_sound_frame = 0
    soundd.current_observer_event_id = 0
    soundd.current_volume = 1.0
    soundd.loaded_sounds = {}
    soundd.loaded_observer_sounds = {1: np.full(4, 0.5, dtype=np.float32)}

    assert np.all(soundd.get_sound_data(4) == 0.5)
    assert soundd.current_observer_prompt == 0
    assert np.all(soundd.get_sound_data(4) == 0.0)

  def test_radio_ack_suppresses_pending_local_prompt(self, monkeypatch):
    soundd = Soundd.__new__(Soundd)
    soundd.current_observer_prompt = 0
    soundd.current_observer_sound_frame = 0
    soundd.current_observer_event_id = 0
    soundd.pending_observer_prompt = 1
    soundd.pending_observer_event_id = 123
    soundd.pending_observer_deadline = 10.0
    monkeypatch.setattr("openpilot.selfdrive.ui.soundd.radio_acknowledged", lambda event_id: event_id == 123)

    soundd.process_pending_observer(now=1.0)

    assert soundd.pending_observer_prompt == 0
    assert soundd.current_observer_prompt == 0

  def test_local_prompt_starts_when_radio_misses_deadline(self, monkeypatch):
    soundd = Soundd.__new__(Soundd)
    soundd.current_observer_prompt = 0
    soundd.current_observer_sound_frame = 0
    soundd.current_observer_event_id = 0
    soundd.pending_observer_prompt = 2
    soundd.pending_observer_event_id = 456
    soundd.pending_observer_deadline = 1.0
    monkeypatch.setattr("openpilot.selfdrive.ui.soundd.radio_acknowledged", lambda event_id: False)

    soundd.process_pending_observer(now=1.1)

    assert soundd.current_observer_prompt == 2
    assert soundd.current_observer_event_id == 456

  def test_perception_prompt_requires_voice_flag(self):
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":false}') == 0
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":true}') == 8
    alert = get_perception_alert(
      b'{"event":"crossTrafficRisk","side":"left","voice":true,"eventId":88,"confidence":0.9}',
    )
    assert alert.prompt == 16
    assert alert.event_id == 88
    assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":true}') == 0
    assert get_perception_prompt(b'not-json') == 0

  # TODO: add test with micd for checking that soundd actually outputs sounds
