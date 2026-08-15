import pytest

from openpilot.selfdrive.road_observer.roadobserverd import (
  DriveTimeState,
  ObserverInput,
  Prompt,
  RoadObserver,
  restore_drive_time_state,
  serialize_drive_time_state,
  PROMPT_PARAM,
)


def state(now: float, **kwargs) -> ObserverInput:
  values = {
    "ego_speed": 0.0,
    "ego_acceleration": 0.0,
    "brake_pressed": False,
    "lead_status": False,
    "lead_distance": 0.0,
    "lead_relative_speed": 0.0,
    "lead_speed": 0.0,
    "lead_acceleration": 0.0,
    "lead_probability": 0.0,
    "driver_alert_level": 0,
    "driver_distracted": False,
    "driver_awareness_percent": 100,
    "driver_face_probability": 0.0,
    "driver_sleep_probability": 0.0,
    "driver_sunglasses_probability": 0.0,
  }
  values.update(kwargs)
  return ObserverInput(now=now, **values)


def test_every_prompt_has_a_setting():
  assert set(PROMPT_PARAM) == set(Prompt) - {Prompt.NONE}


def test_attention_only_on_alert_transition():
  observer = RoadObserver()
  assert observer.update(state(0.0, driver_alert_level=1))[0] == Prompt.ATTENTION
  assert observer.update(state(1.0, driver_alert_level=1))[0] == Prompt.NONE


def test_attention_can_precede_stock_alert():
  observer = RoadObserver()
  assert observer.update(state(0.0, driver_distracted=True, driver_awareness_percent=90))[0] == Prompt.NONE
  assert observer.update(state(1.0, driver_distracted=True, driver_awareness_percent=85))[0] == Prompt.ATTENTION


def test_lead_departed_after_confirmed_stop():
  observer = RoadObserver()
  assert observer.update(state(0.0, lead_status=True, lead_distance=8.0))[0] == Prompt.NONE
  assert observer.update(state(2.1, lead_status=True, lead_distance=8.0))[0] == Prompt.NONE
  assert observer.update(state(2.3, lead_status=True, lead_distance=9.0, lead_speed=1.2))[0] == Prompt.LEAD_DEPARTED


def test_brief_lead_stop_does_not_alert():
  observer = RoadObserver()
  observer.update(state(0.0, lead_status=True, lead_distance=8.0))
  assert observer.update(state(0.5, lead_status=True, lead_distance=9.0, lead_speed=1.2))[0] == Prompt.NONE


def test_slowing_traffic_requires_sustained_closing():
  observer = RoadObserver()
  closing = {
    "ego_speed": 20.0,
    "lead_status": True,
    "lead_distance": 30.0,
    "lead_relative_speed": -10.0,
    "lead_speed": 10.0,
  }
  assert observer.update(state(0.0, **closing))[0] == Prompt.NONE
  prompt, confidence = observer.update(state(0.7, **closing))
  assert prompt == Prompt.SLOWING_TRAFFIC
  assert confidence == pytest.approx(0.8)


def test_slowing_traffic_cooldown():
  observer = RoadObserver()
  closing = {
    "ego_speed": 20.0,
    "lead_status": True,
    "lead_distance": 30.0,
    "lead_relative_speed": -10.0,
    "lead_speed": 10.0,
  }
  observer.update(state(0.0, **closing))
  assert observer.update(state(0.7, **closing))[0] == Prompt.SLOWING_TRAFFIC
  observer.update(state(1.0))
  observer.update(state(2.0, **closing))
  assert observer.update(state(2.7, **closing))[0] == Prompt.NONE


def test_lead_braking_requires_sustained_deceleration():
  observer = RoadObserver()
  braking_lead = {
    "ego_speed": 20.0,
    "lead_status": True,
    "lead_distance": 40.0,
    "lead_relative_speed": -2.0,
    "lead_speed": 18.0,
    "lead_acceleration": -1.2,
    "lead_probability": 0.9,
  }

  assert observer.update(state(0.0, **braking_lead))[0] == Prompt.NONE
  prompt, confidence = observer.update(state(0.6, **braking_lead))
  assert prompt == Prompt.LEAD_BRAKING
  assert confidence == pytest.approx(0.85)


@pytest.mark.parametrize(
  "reaction",
  (
    {"brake_pressed": True},
    {"ego_acceleration": -1.6},
  ),
)
def test_lead_braking_suppressed_after_driver_reacts(reaction):
  observer = RoadObserver()
  braking_lead = {
    "ego_speed": 20.0,
    "lead_status": True,
    "lead_distance": 40.0,
    "lead_relative_speed": -2.0,
    "lead_speed": 18.0,
    "lead_acceleration": -1.2,
    "lead_probability": 0.9,
    **reaction,
  }

  observer.update(state(0.0, **braking_lead))
  assert observer.update(state(1.0, **braking_lead))[0] == Prompt.NONE


def test_drowsiness_requires_sustained_sleep_probability():
  observer = RoadObserver()
  drowsy = {
    "ego_speed": 20.0,
    "driver_face_probability": 0.9,
    "driver_sleep_probability": 0.8,
    "driver_sunglasses_probability": 0.1,
  }

  assert observer.update(state(0.0, **drowsy))[0] == Prompt.NONE
  assert observer.update(state(1.9, **drowsy))[0] == Prompt.NONE
  assert observer.update(state(2.1, **drowsy))[0] == Prompt.DROWSINESS


def test_rest_recommendation_resets_after_short_break():
  observer = RoadObserver()
  observer.REST_RECOMMENDED_AFTER = 2.0
  observer.SHORT_BREAK_SECONDS = 2.0

  assert observer.update(state(0.0, ego_speed=20.0))[0] == Prompt.NONE
  assert observer.update(state(1.0, ego_speed=20.0))[0] == Prompt.NONE
  assert observer.update(state(2.0, ego_speed=20.0))[0] == Prompt.REST_RECOMMENDED

  observer.update(state(3.0))
  observer.update(state(4.0))
  assert observer.drive_time.health_drive_seconds == 0.0


def test_rest_required_after_driving_limit():
  drive_time = DriveTimeState(duty_drive_seconds=RoadObserver.REST_REQUIRED_AFTER - 1.0)
  observer = RoadObserver(drive_time)

  observer.update(state(0.0, ego_speed=20.0))
  assert observer.update(state(1.0, ego_speed=20.0))[0] == Prompt.REST_REQUIRED


def test_rest_alerts_wait_until_car_is_moving():
  drive_time = DriveTimeState(
    health_drive_seconds=RoadObserver.REST_RECOMMENDED_AFTER,
    duty_drive_seconds=RoadObserver.REST_REQUIRED_AFTER,
  )
  observer = RoadObserver(drive_time)

  assert observer.update(state(0.0))[0] == Prompt.NONE
  assert observer.update(state(1.0, ego_speed=2.0))[0] == Prompt.NONE
  assert observer.update(state(2.0, ego_speed=20.0))[0] == Prompt.REST_REQUIRED


def test_drive_time_state_restores_breaks_across_process_restart():
  drive_time = DriveTimeState(
    health_drive_seconds=7_200.0,
    duty_drive_seconds=12_000.0,
    stationary_seconds=0.0,
  )
  raw = serialize_drive_time_state(drive_time, wall_time=1_000.0)

  short_break = restore_drive_time_state(
    raw,
    wall_time=1_000.0 + RoadObserver.SHORT_BREAK_SECONDS,
  )
  assert short_break.health_drive_seconds == 0.0
  assert short_break.duty_drive_seconds == 12_000.0

  full_break = restore_drive_time_state(
    raw,
    wall_time=1_000.0 + RoadObserver.FULL_BREAK_SECONDS,
  )
  assert full_break.health_drive_seconds == 0.0
  assert full_break.duty_drive_seconds == 0.0
