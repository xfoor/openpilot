import pytest

from openpilot.selfdrive.road_observer.roadobserverd import ObserverInput, Prompt, RoadObserver


def state(now: float, **kwargs) -> ObserverInput:
  values = {
    "ego_speed": 0.0,
    "lead_status": False,
    "lead_distance": 0.0,
    "lead_relative_speed": 0.0,
    "lead_speed": 0.0,
    "driver_alert_level": 0,
    "driver_distracted": False,
    "driver_awareness_percent": 100,
  }
  values.update(kwargs)
  return ObserverInput(now=now, **values)


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
