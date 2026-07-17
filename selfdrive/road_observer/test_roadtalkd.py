import hashlib
import hmac
import time

from openpilot.selfdrive.road_observer.roadcapture import RoadCapture
from openpilot.selfdrive.road_observer.roadtalkd import (
  DISCOVERY_REQUEST,
  DISCOVERY_RESPONSE,
  _canonical_request,
  _is_private_peer,
)


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
