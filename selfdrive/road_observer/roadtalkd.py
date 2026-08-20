#!/usr/bin/env python3
import base64
from collections import deque
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.road_observer.perception import get_perception_alert
from openpilot.selfdrive.road_observer.roadalert import AlertBroker, radio_audio_ready
from openpilot.selfdrive.road_observer.roadcapture import RoadCapture

PORT = 7766
DISCOVERY_PORT = 7767
DISCOVERY_REQUEST = b"ROADTALK_DISCOVER_V1"
DISCOVERY_RESPONSE = b"ROADTALK_COMMA4_V1"
MAX_BODY_BYTES = 4096
MAX_CLOCK_SKEW_SECONDS = 30
NONCE_CACHE_SIZE = 512
MUTE_UNTIL = 4102444800  # 2100-01-01; unmute explicitly to restore observer audio.
STATE_DIR = Path("/persist/roadtalk")
SECRET_PATH = STATE_DIR / "shared_secret"
QUIET_UNTIL_PATH = STATE_DIR / "quiet_until"


def _json_bytes(value: dict[str, Any]) -> bytes:
  return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _is_private_peer(address: str) -> bool:
  try:
    ip = ipaddress.ip_address(address)
    return ip.is_private and not ip.is_loopback and not ip.is_multicast
  except ValueError:
    return False


def _canonical_request(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
  body_hash = hashlib.sha256(body).hexdigest()
  return f"{method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}".encode()


class RoadTalkCore:
  def __init__(self, params: Params | None = None, alerts: AlertBroker | None = None):
    self.params = params or Params()
    self.alerts = alerts or AlertBroker()
    self.capture = RoadCapture()
    self.nonces: deque[str] = deque(maxlen=NONCE_CACHE_SIZE)
    self.nonce_lock = threading.Lock()
    self.pair_lock = threading.Lock()

  @staticmethod
  def _read(path: Path) -> str:
    try:
      return path.read_text(encoding="utf-8").strip()
    except OSError:
      return ""

  @staticmethod
  def _write_private(path: Path, value: str) -> None:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)

  def pair(self) -> tuple[int, dict[str, Any]]:
    if not self.params.get_bool("IsOffroad") or not self.params.get_bool("AdbEnabled"):
      return 403, {"error": "Park the car and enable ADB on the Comma 4"}
    with self.pair_lock:
      if SECRET_PATH.exists():
        return 409, {"error": "Comma 4 is already paired"}
      secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
      self._write_private(SECRET_PATH, secret)
      return 200, {"secret": secret, "protocol": 2}

  def authenticate(self, method: str, path: str, headers: Any, body: bytes) -> bool:
    secret = self._read(SECRET_PATH)
    timestamp = headers.get("X-RoadTalk-Timestamp", "")
    nonce = headers.get("X-RoadTalk-Nonce", "")
    signature = headers.get("X-RoadTalk-Signature", "")
    if not secret or not timestamp.isdigit() or len(nonce) < 16:
      return False
    if abs(time.time() - int(timestamp)) > MAX_CLOCK_SKEW_SECONDS:  # noqa: TID251
      return False

    expected = hmac.new(
      secret.encode(),
      _canonical_request(method, path, timestamp, nonce, body),
      hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
      return False
    with self.nonce_lock:
      if nonce in self.nonces:
        return False
      self.nonces.append(nonce)
    return True

  def status(self) -> dict[str, Any]:
    quiet_until_raw = self._read(QUIET_UNTIL_PATH) or "0"
    quiet_until = int(quiet_until_raw) if quiet_until_raw.isdigit() else 0
    status = {
      "protocol": 2,
      "onroad": self.params.get_bool("IsOnroad"),
      "engaged": self.params.get_bool("IsEngaged"),
      "observerEnabled": self.params.get_bool("RoadObserverEnabled"),
      "perceptionEnabled": self.params.get_bool("RoadPerceptionEnabled"),
      "radioAudioReady": radio_audio_ready(ready_path=self.alerts.ready_path),
      "quietUntil": quiet_until,
      "version": self.params.get("Version") or "unknown",
    }
    status.update(self.capture.status())
    return status

  def telemetry(self) -> dict[str, Any]:
    gps = self.alerts.telemetry()
    return {
      "protocol": 2,
      "onroad": self.params.get_bool("IsOnroad"),
      "gps": gps,
    }

  def command(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    action = payload.get("action")
    now = int(time.time())  # noqa: TID251
    if action == "observer_mute":
      self._write_private(QUIET_UNTIL_PATH, str(MUTE_UNTIL))
      return 200, {"ok": True, "message": "Comma observer muted"}
    if action == "observer_unmute":
      self._write_private(QUIET_UNTIL_PATH, "0")
      return 200, {"ok": True, "message": "Comma observer unmuted"}
    if action == "observer_quiet":
      minutes = payload.get("minutes")
      if not isinstance(minutes, int) or minutes not in (15, 30, 60, 120):
        return 400, {"error": "Quiet time must be 15, 30, 60, or 120 minutes"}
      self._write_private(QUIET_UNTIL_PATH, str(now + minutes * 60))
      return 200, {"ok": True, "message": f"Comma observer quiet for {minutes} minutes"}
    if action == "take_road_photo":
      if not self.params.get_bool("IsOnroad"):
        return 409, {"error": "The road camera is available while the car is on"}
      return self.capture.take_photo()
    if action == "record_road_video":
      if not self.params.get_bool("IsOnroad"):
        return 409, {"error": "The road camera is available while the car is on"}
      minutes = payload.get("minutes")
      if not isinstance(minutes, int):
        return 400, {"error": "Road video duration is required"}
      return self.capture.start_video(minutes)
    return 400, {"error": "Command is not allowlisted"}


class RoadTalkHandler(BaseHTTPRequestHandler):
  server_version = "RoadTalk/2"

  @property
  def core(self) -> RoadTalkCore:
    return self.server.core  # type: ignore[attr-defined]

  def _reply(self, status: int, value: dict[str, Any]) -> None:
    body = _json_bytes(value)
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    self.wfile.write(body)

  def _reply_jpeg(self, body: bytes) -> None:
    self.send_response(200)
    self.send_header("Content-Type", "image/jpeg")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    self.wfile.write(body)

  def _reply_file(self, path: Path) -> None:
    content_type = "image/jpeg" if path.suffix == ".jpg" else "video/mp2t"
    try:
      size = path.stat().st_size
      stream = path.open("rb")
    except OSError:
      self._reply(404, {"error": "Capture not found"})
      return
    with stream:
      self.send_response(200)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(size))
      self.send_header("Cache-Control", "no-store")
      self.send_header("X-Content-Type-Options", "nosniff")
      self.end_headers()
      while chunk := stream.read(1024 * 1024):
        self.wfile.write(chunk)

  def _body(self) -> bytes | None:
    try:
      length = int(self.headers.get("Content-Length", "0"))
    except ValueError:
      return None
    if length < 0 or length > MAX_BODY_BYTES:
      return None
    return self.rfile.read(length)

  def _private_peer(self) -> bool:
    return _is_private_peer(self.client_address[0])

  def _next_alert(self) -> None:
    query = parse_qs(urlsplit(self.path).query)
    try:
      after = int(query.get("after", ["0"])[0])
      wait_ms = int(query.get("waitMs", ["0"])[0])
    except ValueError:
      self._reply(400, {"error": "Invalid alert cursor"})
      return
    if after < 0 or wait_ms not in range(2001):
      self._reply(400, {"error": "Invalid alert cursor"})
      return

    if self.headers.get("X-RoadTalk-Audio-Ready") == "1":
      self.core.alerts.mark_radio_ready(True)
    event = self.core.alerts.next_event(after, wait_ms / 1000.0)
    self._reply(200, {
      "protocol": 2,
      "event": event.payload(time.monotonic()) if event is not None else None,
    })

  def do_GET(self) -> None:
    path = urlsplit(self.path).path
    if not self._private_peer():
      self._reply(403, {"error": "Private network required"})
    elif (
      path not in (
        "/v1/status",
        "/v1/telemetry",
        "/v1/alerts/next",
        "/v1/capture/photo",
        "/v1/capture/media",
      )
      and not path.startswith("/v1/capture/media/")
    ):
      self._reply(404, {"error": "Not found"})
    elif not self.core.authenticate("GET", self.path, self.headers, b""):
      self._reply(401, {"error": "Authentication failed"})
    elif path == "/v1/alerts/next":
      self._next_alert()
    elif path == "/v1/telemetry":
      self._reply(200, self.core.telemetry())
    elif path == "/v1/capture/photo":
      photo = self.core.capture.photo_bytes()
      if photo is None:
        self._reply(404, {"error": "No road photo is available"})
      else:
        self._reply_jpeg(photo)
    elif path == "/v1/capture/media":
      self._reply(200, {"captures": self.core.capture.media()})
    elif path.startswith("/v1/capture/media/"):
      name = path.removeprefix("/v1/capture/media/")
      path = self.core.capture.capture_path(name)
      if path is None:
        self._reply(404, {"error": "Capture not found"})
      else:
        self._reply_file(path)
    else:
      self._reply(200, self.core.status())

  def do_DELETE(self) -> None:
    if not self._private_peer():
      self._reply(403, {"error": "Private network required"})
    elif not self.path.startswith("/v1/capture/media/"):
      self._reply(404, {"error": "Not found"})
    elif not self.core.authenticate("DELETE", self.path, self.headers, b""):
      self._reply(401, {"error": "Authentication failed"})
    else:
      name = self.path.removeprefix("/v1/capture/media/")
      if self.core.capture.delete_media(name):
        self._reply(200, {"ok": True, "deleted": name})
      else:
        self._reply(404, {"error": "Capture not found"})

  def do_POST(self) -> None:
    body = self._body()
    path = urlsplit(self.path).path
    if not self._private_peer():
      self._reply(403, {"error": "Private network required"})
    elif body is None:
      self._reply(413, {"error": "Invalid request size"})
    elif path == "/v1/pair":
      status, value = self.core.pair()
      self._reply(status, value)
    elif path not in ("/v1/command", "/v1/alerts/ack", "/v1/alerts/heartbeat"):
      self._reply(404, {"error": "Not found"})
    elif not self.core.authenticate("POST", self.path, self.headers, body):
      self._reply(401, {"error": "Authentication failed"})
    else:
      try:
        payload = json.loads(body)
      except (UnicodeDecodeError, json.JSONDecodeError):
        self._reply(400, {"error": "Invalid JSON"})
        return
      if not isinstance(payload, dict):
        self._reply(400, {"error": "Invalid command"})
        return
      if path == "/v1/alerts/ack":
        event_id = payload.get("eventId")
        if not isinstance(event_id, int) or event_id <= 0:
          self._reply(400, {"error": "Invalid alert event"})
        elif not self.core.alerts.acknowledge(event_id):
          self._reply(409, {"error": "Alert expired"})
        else:
          self._reply(200, {"ok": True, "eventId": event_id})
      elif path == "/v1/alerts/heartbeat":
        ready = payload.get("audioReady")
        if not isinstance(ready, bool):
          self._reply(400, {"error": "Invalid audio state"})
        else:
          self.core.alerts.mark_radio_ready(ready)
          self._reply(200, {"ok": True, "audioReady": ready})
      else:
        status, value = self.core.command(payload)
        self._reply(status, value)

  def log_message(self, format_string: str, *args: Any) -> None:
    cloudlog.debug("roadtalkd " + format_string, *args)


class RoadTalkServer(ThreadingHTTPServer):
  daemon_threads = True
  allow_reuse_address = True

  def __init__(self, address: tuple[str, int], core: RoadTalkCore):
    self.core = core
    super().__init__(address, RoadTalkHandler)


def _interface_ipv4(name: str) -> str | None:
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    packed = struct.pack("256s", name[:15].encode())
    return socket.inet_ntoa(struct.unpack("256s", fcntl.ioctl(sock.fileno(), 0x8915, packed))[0][20:24])
  except OSError:
    return None
  finally:
    sock.close()


def _private_wifi_address() -> str | None:
  for _, name in socket.if_nameindex():
    if not name.startswith(("wlan", "wifi")):
      continue
    address = _interface_ipv4(name)
    if address and _is_private_peer(address):
      return address
  return None


def _serve_discovery(address: str, stop_event: threading.Event) -> None:
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, DISCOVERY_PORT))
    sock.settimeout(1)
    while not stop_event.is_set():
      try:
        payload, peer = sock.recvfrom(128)
      except TimeoutError:
        continue
      if payload == DISCOVERY_REQUEST and _is_private_peer(peer[0]):
        sock.sendto(DISCOVERY_RESPONSE, peer)


def _observer_audio_quiet(now: int | None = None) -> bool:
  value = RoadTalkCore._read(QUIET_UNTIL_PATH)
  quiet_until = int(value) if value.isdigit() else 0
  return (int(time.time()) if now is None else now) < quiet_until  # noqa: TID251


def _collect_alerts_and_gps(alerts: AlertBroker) -> None:
  sm = messaging.SubMaster(["roadObserverState", "customReservedRawData0", "gpsLocationExternal"])
  while True:
    sm.update(500)
    if sm.updated["roadObserverState"]:
      value = sm["roadObserverState"]
      if not _observer_audio_quiet():
        alerts.publish(value.eventId, value.prompt.raw, value.confidence)

    if sm.updated["customReservedRawData0"] and not _observer_audio_quiet():
      perception_alert = get_perception_alert(sm["customReservedRawData0"])
      alerts.publish(
        perception_alert.event_id,
        perception_alert.prompt,
        perception_alert.confidence,
      )

    if sm.updated["gpsLocationExternal"]:
      gps = sm["gpsLocationExternal"]
      valid = (
        gps.unixTimestampMillis > 1_500_000_000_000
        and -90.0 <= gps.latitude <= 90.0
        and -180.0 <= gps.longitude <= 180.0
        and gps.horizontalAccuracy <= 100.0
        and (gps.latitude != 0.0 or gps.longitude != 0.0)
      )
      alerts.update_gps({
        "valid": valid,
        "timestampMillis": gps.unixTimestampMillis,
        "latitude": gps.latitude,
        "longitude": gps.longitude,
        "altitude": gps.altitude,
        "speed": gps.speed,
        "bearing": gps.bearingDeg,
        "accuracy": gps.horizontalAccuracy,
      })


def main() -> None:
  alerts = AlertBroker()
  core = RoadTalkCore(alerts=alerts)
  threading.Thread(
    target=_collect_alerts_and_gps,
    args=(alerts,),
    name="roadtalk-events",
    daemon=True,
  ).start()
  while True:
    address = _private_wifi_address()
    if address is None:
      time.sleep(5)
      continue
    discovery_stop = threading.Event()
    try:
      cloudlog.info(f"roadtalkd listening on {address}:{PORT}")
      threading.Thread(
        target=_serve_discovery,
        args=(address, discovery_stop),
        name="roadtalk-discovery",
        daemon=True,
      ).start()
      RoadTalkServer((address, PORT), core).serve_forever(poll_interval=1)
    except OSError:
      cloudlog.exception("roadtalkd server stopped")
      time.sleep(5)
    finally:
      discovery_stop.set()


if __name__ == "__main__":
  main()
