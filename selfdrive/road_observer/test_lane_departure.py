from openpilot.selfdrive.road_observer.lane_departure import (
  LANE_PROMPT_PARAM,
  LaneDepartureAlerter,
  LanePrompt,
)


def test_every_lane_prompt_has_a_setting():
  assert set(LANE_PROMPT_PARAM) == set(LanePrompt) - {LanePrompt.NONE}


def test_lane_departure_is_directional_and_debounced():
  alerter = LaneDepartureAlerter()

  assert alerter.update(0.0, left=True, right=False, lateral_active=False) == LanePrompt.DEPARTURE_LEFT
  assert alerter.update(0.5, left=True, right=False, lateral_active=False) == LanePrompt.NONE
  assert alerter.update(1.0, left=False, right=False, lateral_active=False) == LanePrompt.NONE
  assert alerter.update(1.5, left=False, right=True, lateral_active=False) == LanePrompt.NONE
  assert alerter.update(2.0, left=False, right=False, lateral_active=False) == LanePrompt.NONE
  assert alerter.update(2.1, left=False, right=True, lateral_active=False) == LanePrompt.DEPARTURE_RIGHT


def test_lane_departure_does_not_warn_while_lateral_control_is_active():
  alerter = LaneDepartureAlerter()

  assert alerter.update(0.0, left=True, right=False, lateral_active=True) == LanePrompt.NONE


def test_repeated_lane_departures_recommend_a_break():
  alerter = LaneDepartureAlerter()
  prompt = LanePrompt.NONE

  for index in range(alerter.REPEATED_COUNT):
    now = index * 10.0
    prompt = alerter.update(now, left=False, right=True, lateral_active=False)
    alerter.update(now + 0.5, left=False, right=False, lateral_active=False)

  assert prompt == LanePrompt.REPEATED_DEPARTURE
  alerter.update(60.0, left=False, right=False, lateral_active=False)
  assert alerter.update(70.0, left=False, right=True, lateral_active=False) == LanePrompt.DEPARTURE_RIGHT


def test_old_departures_leave_repeated_warning_window():
  alerter = LaneDepartureAlerter()

  for index in range(alerter.REPEATED_COUNT - 1):
    now = index * 10.0
    alerter.update(now, left=True, right=False, lateral_active=False)
    alerter.update(now + 0.5, left=False, right=False, lateral_active=False)

  later = alerter.REPEATED_WINDOW + 60.0
  assert alerter.update(later, left=True, right=False, lateral_active=False) == LanePrompt.DEPARTURE_LEFT
