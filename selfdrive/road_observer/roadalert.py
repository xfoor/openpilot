from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any


RUNTIME_DIR = Path("/dev/shm/roadtalk")
RADIO_READY_PATH = RUNTIME_DIR / "radio_ready"
RADIO_ACK_PATH = RUNTIME_DIR / "radio_ack"
RADIO_READY_MAX_AGE_SECONDS = 2.5
ALERT_TTL_SECONDS = 5.0
MAX_ALERTS = 64


PROMPT_PHRASES = {
  1: "attention",
  2: "leadDeparted",
  3: "slowingTraffic",
  4: "leadBraking",
  5: "drowsiness",
  6: "restRecommended",
  7: "restRequired",
  8: "pedestrian",
  9: "cyclist",
  13: "curveAcceleration",
  14: "leadPullAway",
  15: "crossTraffic",
  16: "junctionVehicleLeft",
  17: "junctionVehicleRight",
  18: "speedLimit",
}

PROMPT_PRIORITIES = {
  1: 2,
  2: 1,
  3: 2,
  4: 3,
  5: 3,
  6: 1,
  7: 2,
  8: 3,
  9: 3,
  13: 2,
  14: 1,
  15: 3,
  16: 3,
  17: 3,
  18: 2,
}


def _write_runtime(path: Path, value: str) -> None:
  path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  temporary = path.with_suffix(".tmp")
  temporary.write_text(value, encoding="utf-8")
  temporary.replace(path)


def _read_runtime(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8").strip()
  except OSError:
    return ""


def radio_audio_ready(now: float | None = None, ready_path: Path = RADIO_READY_PATH) -> bool:
  value = _read_runtime(ready_path)
  try:
    updated_at = float(value)
  except ValueError:
    return False
  current = time.monotonic() if now is None else now
  return 0.0 <= current - updated_at <= RADIO_READY_MAX_AGE_SECONDS


def radio_acknowledged(event_id: int, ack_path: Path = RADIO_ACK_PATH) -> bool:
  return event_id > 0 and _read_runtime(ack_path) == str(event_id)


@dataclass(frozen=True)
class AlertEvent:
  event_id: int
  prompt: int
  phrase: str
  priority: int
  confidence: float
  created_at: float
  expires_at: float
  speed_limit_kph: int | None = None
  overspeed: bool = False

  def payload(self, now: float) -> dict[str, Any]:
    payload = {
      "eventId": self.event_id,
      "prompt": self.prompt,
      "phrase": self.phrase,
      "priority": self.priority,
      "confidence": round(self.confidence, 3),
      "expiresInMs": max(0, round((self.expires_at - now) * 1000)),
    }
    if self.speed_limit_kph is not None:
      payload["speedLimitKph"] = self.speed_limit_kph
      payload["overspeed"] = self.overspeed
    return payload


class AlertBroker:
  def __init__(self, runtime_dir: Path = RUNTIME_DIR):
    self.runtime_dir = runtime_dir
    self.ready_path = runtime_dir / RADIO_READY_PATH.name
    self.ack_path = runtime_dir / RADIO_ACK_PATH.name
    self.events: deque[AlertEvent] = deque(maxlen=MAX_ALERTS)
    self.condition = threading.Condition()
    self.latest_gps: dict[str, Any] | None = None

  def publish(self, event_id: int, prompt: int, confidence: float, now: float | None = None,
              speed_limit_kph: int | None = None, overspeed: bool = False) -> bool:
    phrase = PROMPT_PHRASES.get(prompt)
    valid_metadata = (
      (
        speed_limit_kph in (20, 30, 50, 60, 70, 80, 100, 120)
        and isinstance(overspeed, bool)
      )
      if prompt == 18
      else speed_limit_kph is None and overspeed is False
    )
    if event_id <= 0 or phrase is None or not math.isfinite(confidence) or not valid_metadata:
      return False
    created_at = time.monotonic() if now is None else now
    event = AlertEvent(
      event_id=event_id,
      prompt=prompt,
      phrase=phrase,
      priority=PROMPT_PRIORITIES[prompt],
      confidence=float(confidence),
      created_at=created_at,
      expires_at=created_at + ALERT_TTL_SECONDS,
      speed_limit_kph=speed_limit_kph,
      overspeed=overspeed,
    )
    with self.condition:
      if any(existing.event_id == event_id for existing in self.events):
        return False
      self.events.append(event)
      self.condition.notify_all()
    return True

  def next_event(self, after: int, wait_seconds: float, now_fn=time.monotonic) -> AlertEvent | None:
    deadline = now_fn() + max(0.0, wait_seconds)
    with self.condition:
      while True:
        now = now_fn()
        while self.events and self.events[0].expires_at <= now:
          self.events.popleft()
        candidates = [candidate for candidate in self.events if candidate.expires_at > now]
        if after == 0:
          event = candidates[0] if candidates else None
        else:
          cursor = next(
            (index for index, candidate in enumerate(candidates) if candidate.event_id == after),
            None,
          )
          event = (
            candidates[cursor + 1]
            if cursor is not None and cursor + 1 < len(candidates)
            else candidates[0] if cursor is None and candidates
            else None
          )
        if event is not None:
          return event
        remaining = deadline - now
        if remaining <= 0.0:
          return None
        self.condition.wait(remaining)

  def mark_radio_ready(self, ready: bool) -> None:
    _write_runtime(self.ready_path, str(time.monotonic()) if ready else "0")

  def acknowledge(self, event_id: int) -> bool:
    with self.condition:
      known = any(event.event_id == event_id for event in self.events)
    if not known:
      return False
    _write_runtime(self.ack_path, str(event_id))
    return True

  def update_gps(self, payload: dict[str, Any]) -> None:
    with self.condition:
      self.latest_gps = payload

  def telemetry(self) -> dict[str, Any] | None:
    with self.condition:
      return dict(self.latest_gps) if self.latest_gps is not None else None
