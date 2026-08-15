import time
from types import SimpleNamespace

import numpy as np

from cereal import car
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from openpilot.selfdrive.road_observer.lane_departure import LaneDepartureAlerter, LanePrompt
from openpilot.selfdrive.ui.soundd import (
  SELFDRIVE_STATE_TIMEOUT,
  Soundd,
  check_selfdrive_timeout_alert,
  get_perception_prompt,
  observer_sound_list,
)

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
VisualAlert = car.CarControl.HUDControl.VisualAlert


class FakeParams:
  def __init__(self, disabled=()):
    self.disabled = set(disabled)

  def get_bool(self, key):
    return key not in self.disabled


class FakeLaneSubMaster:
  def __init__(self, left=True, right=False, lateral_active=False):
    self.updated = {
      "selfdriveState": True,
      "roadObserverState": False,
      "customReservedRawData0": False,
    }
    self.messages = {
      "selfdriveState": SimpleNamespace(
        alertSound=SimpleNamespace(raw=AudibleAlert.prompt),
        alertHudVisual=VisualAlert.ldw,
      ),
      "driverAssistance": SimpleNamespace(
        leftLaneDeparture=left,
        rightLaneDeparture=right,
      ),
      "carControl": SimpleNamespace(latActive=lateral_active),
    }

  def __getitem__(self, service):
    return self.messages[service]

  def all_checks(self, services):
    return True


def make_lane_soundd(disabled=()):
  soundd = Soundd.__new__(Soundd)
  soundd.params = FakeParams(disabled)
  soundd.lane_alerter = LaneDepartureAlerter()
  soundd.current_alert = AudibleAlert.none
  soundd.current_sound_frame = 0
  soundd.current_observer_prompt = 0
  soundd.current_observer_sound_frame = 0
  soundd.current_volume = 1.0
  soundd.ramp_start_volume = 0.1
  soundd.ramp_start_time = 0.0
  soundd.selfdrive_timeout_alert = False
  soundd.suppress_ldw_prompt = False
  soundd.observer_quiet_until = 0
  soundd.observer_quiet_checked_at = time.monotonic()
  soundd.loaded_sounds = {AudibleAlert.prompt: np.ones(8, dtype=np.float32)}
  soundd.loaded_observer_sounds = {
    LanePrompt.DEPARTURE_LEFT: np.full(8, 0.5, dtype=np.float32),
  }
  return soundd


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

  def test_lane_voice_replaces_stock_lane_chime(self):
    soundd = make_lane_soundd()
    sm = FakeLaneSubMaster()

    soundd.get_audible_alert(sm)

    assert soundd.current_alert == AudibleAlert.none
    assert soundd.current_observer_prompt == LanePrompt.DEPARTURE_LEFT
    assert soundd.suppress_ldw_prompt
    assert np.all(soundd.get_sound_data(4) == 0.5)

    soundd.get_audible_alert(sm)
    assert soundd.current_alert == AudibleAlert.none

  def test_stock_lane_chime_remains_when_lane_voice_is_disabled(self):
    soundd = make_lane_soundd(disabled={"RoadObserverAttentionEnabled"})
    sm = FakeLaneSubMaster()

    soundd.get_audible_alert(sm)

    assert soundd.current_alert == AudibleAlert.prompt
    assert soundd.current_observer_prompt == 0
    assert not soundd.suppress_ldw_prompt

  def test_directional_voice_remains_when_repeated_drift_voice_is_disabled(self):
    soundd = make_lane_soundd(disabled={"RoadObserverDriverHealthEnabled"})
    now = time.monotonic()
    soundd.lane_alerter.departures.extend([now - 40.0, now - 30.0, now - 20.0, now - 10.0])
    sm = FakeLaneSubMaster()

    soundd.get_audible_alert(sm)

    assert soundd.current_observer_prompt == LanePrompt.DEPARTURE_LEFT

  def test_stock_lane_chime_remains_when_other_observer_voice_is_playing(self):
    soundd = make_lane_soundd()
    soundd.current_observer_prompt = 1
    soundd.current_alert = AudibleAlert.prompt
    sm = FakeLaneSubMaster()

    soundd.update_lane_departure(sm)

    assert soundd.current_alert == AudibleAlert.prompt
    assert soundd.current_observer_prompt == 1
    assert not soundd.suppress_ldw_prompt

  def test_perception_prompt_requires_voice_flag(self):
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":false}') == 0
    assert get_perception_prompt(b'{"event":"pedestrianRisk","voice":true}') == 8
    assert get_perception_prompt(b'{"event":"trafficLightGreen","voice":true}') == 0
    assert get_perception_prompt(b'not-json') == 0

  # TODO: add test with micd for checking that soundd actually outputs sounds
