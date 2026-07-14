import time

import numpy as np

from cereal import car
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.ui.soundd import SELFDRIVE_STATE_TIMEOUT, Soundd, check_selfdrive_timeout_alert, get_perception_prompt

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class TestSoundd:
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
    soundd.current_volume = 1.0
    soundd.loaded_sounds = {}
    soundd.loaded_observer_sounds = {1: np.full(4, 0.5, dtype=np.float32)}

    assert np.all(soundd.get_sound_data(4) == 0.5)
    assert soundd.current_observer_prompt == 0
    assert np.all(soundd.get_sound_data(4) == 0.0)

  def test_perception_prompt_requires_voice_flag(self):
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":false}') == 0
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":true}') == 4
    assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":true}') == 8
    assert get_perception_prompt(b'not-json') == 0

  # TODO: add test with micd for checking that soundd actually outputs sounds
