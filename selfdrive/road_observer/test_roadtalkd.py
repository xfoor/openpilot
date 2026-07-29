import hashlib
import hmac
import time

from openpilot.selfdrive.road_observer.roadcapture import CAPTURE_NAME_PATTERN, RoadCapture
from openpilot.selfdrive.road_observer.roadtalkd import (
  DISCOVERY_REQUEST,
  DISCOVERY_RESPONSE,
  RoadTalkCore,
  _canonical_request,
  _is_private_peer,
)


class FakeParams:
  values = {
    "IsEngaged": True,
    "IsOnroad": True,
    "Version": b"0.11.1-roadtalk",
  }

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def get(self, key):
    return self.values.get(key)


def test_private_peer_filter():
  assert _is_private_peer("192.168.43.2")
  assert _is_private_peer("10.0.0.5")
  assert not _is_private_peer("127.0.0.1")
  assert not _is_private_peer("8.8.8.8")


def test_canonical_request_signature_is_stable():
  timestamp = str(int(time.time()))
  canonical = _canonical_request("POST", "/v1/command", timestamp, "0123456789abcdef", b'{"action":"observer_mute"}')
  signature = hmac.new(b"secret", canonical, hashlib.sha256).hexdigest()
  assert len(signature) == 64
  assert hmac.compare_digest(signature, hmac.new(b"secret", canonical, hashlib.sha256).hexdigest())


def test_discovery_protocol_is_versioned():
  assert DISCOVERY_REQUEST == b"ROADTALK_DISCOVER_V1"
  assert DISCOVERY_RESPONSE == b"ROADTALK_COMMA4_V1"


def test_video_duration_is_bounded():
  capture = RoadCapture()
  status, _ = capture.start_video(3)
  assert status == 400


def test_capture_names_are_strictly_allowlisted():
  assert CAPTURE_NAME_PATTERN.fullmatch("road-20260717-101112.jpg")
  assert CAPTURE_NAME_PATTERN.fullmatch("road-20260717-101112-2m.ts")
  assert not CAPTURE_NAME_PATTERN.fullmatch("../params")
  assert not CAPTURE_NAME_PATTERN.fullmatch("road-20260717-101112.ts.part")


def test_stock_status_does_not_claim_observer_features():
  status = RoadTalkCore(FakeParams()).status()
  assert status["onroad"]
  assert status["engaged"]
  assert not status["observerEnabled"]
  assert not status["perceptionEnabled"]
  assert status["quietUntil"] == 0


def test_stock_release_rejects_observer_commands():
  core = RoadTalkCore(FakeParams())
  for action in ("observer_mute", "observer_unmute", "observer_quiet"):
    status, response = core.command({"action": action, "minutes": 15})
    assert status == 409
    assert "not installed" in response["error"]
