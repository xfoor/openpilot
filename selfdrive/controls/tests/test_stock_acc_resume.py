import pytest

from openpilot.selfdrive.controls.controlsd import should_auto_resume_from_standstill


@pytest.mark.parametrize(
  "enabled, standstill, should_stop",
  (
    (False, True, False),
    (True, False, False),
    (True, True, True),
  ),
)
def test_resume_requires_active_departure(enabled, standstill, should_stop):
  assert not should_auto_resume_from_standstill(enabled, standstill, should_stop, "honda", True)


def test_volkswagen_stock_acc_requires_driver_start():
  assert not should_auto_resume_from_standstill(True, True, False, "volkswagen", True)


def test_volkswagen_openpilot_longitudinal_can_resume():
  assert should_auto_resume_from_standstill(True, True, False, "volkswagen", False)


def test_other_stock_acc_behavior_is_unchanged():
  assert should_auto_resume_from_standstill(True, True, False, "honda", True)
