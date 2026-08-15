#!/usr/bin/env python3
import enum
import json
import math
import time
from dataclasses import dataclass

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper


class Prompt(enum.IntEnum):
  NONE = 0
  ATTENTION = 1
  LEAD_DEPARTED = 2
  SLOWING_TRAFFIC = 3
  LEAD_BRAKING = 4
  DROWSINESS = 5
  REST_RECOMMENDED = 6
  REST_REQUIRED = 7


PROMPT_PARAM = {
  Prompt.ATTENTION: "RoadObserverAttentionEnabled",
  Prompt.LEAD_DEPARTED: "RoadObserverLeadEnabled",
  Prompt.SLOWING_TRAFFIC: "RoadObserverSlowingTrafficEnabled",
  Prompt.LEAD_BRAKING: "RoadObserverLeadBrakingEnabled",
  Prompt.DROWSINESS: "RoadObserverDriverHealthEnabled",
  Prompt.REST_RECOMMENDED: "RoadObserverDriverHealthEnabled",
  Prompt.REST_REQUIRED: "RoadObserverDriverHealthEnabled",
}


@dataclass
class DriveTimeState:
  health_drive_seconds: float = 0.0
  duty_drive_seconds: float = 0.0
  stationary_seconds: float = 0.0


@dataclass
class ObserverInput:
  now: float
  ego_speed: float
  ego_acceleration: float
  brake_pressed: bool
  lead_status: bool
  lead_distance: float
  lead_relative_speed: float
  lead_speed: float
  lead_acceleration: float
  lead_probability: float
  driver_alert_level: int
  driver_distracted: bool
  driver_awareness_percent: int
  driver_face_probability: float
  driver_sleep_probability: float
  driver_sunglasses_probability: float


class RoadObserver:
  ATTENTION_COOLDOWN = 20.0
  LEAD_DEPARTED_COOLDOWN = 8.0
  LEAD_BRAKING_COOLDOWN = 20.0
  SLOWING_TRAFFIC_COOLDOWN = 15.0
  DROWSINESS_COOLDOWN = 120.0
  REST_RECOMMENDED_COOLDOWN = 30.0 * 60.0
  REST_REQUIRED_COOLDOWN = 15.0 * 60.0
  STOPPED_CONFIRMATION = 2.0
  LEAD_BRAKING_CONFIRMATION = 0.5
  SLOWING_CONFIRMATION = 0.6
  DROWSINESS_CONFIRMATION = 2.0
  DRIVER_HEALTH_MIN_SPEED = 5.0
  REST_RECOMMENDED_AFTER = 2.0 * 60.0 * 60.0
  REST_REQUIRED_AFTER = 4.5 * 60.0 * 60.0
  SHORT_BREAK_SECONDS = 15.0 * 60.0
  FULL_BREAK_SECONDS = 45.0 * 60.0

  def __init__(self, drive_time_state: DriveTimeState | None = None):
    self.last_prompt_at = dict.fromkeys(Prompt, -math.inf)
    self.previous_attention_needed = False
    self.stopped_lead_since: float | None = None
    self.stopped_lead_distance = 0.0
    self.lead_braking_since: float | None = None
    self.slowing_since: float | None = None
    self.drowsiness_since: float | None = None
    self.drive_time = drive_time_state or DriveTimeState()
    self.last_update_at: float | None = None

  def _ready(self, prompt: Prompt, now: float, cooldown: float) -> bool:
    if now - self.last_prompt_at[prompt] < cooldown:
      return False
    self.last_prompt_at[prompt] = now
    return True

  def _confirmed(self, condition: bool, since_name: str, now: float, duration: float) -> bool:
    since = getattr(self, since_name)
    if not condition:
      setattr(self, since_name, None)
      return False
    if since is None:
      setattr(self, since_name, now)
      return False
    return now - since >= duration

  def _update_drive_time(self, state: ObserverInput) -> None:
    if self.last_update_at is None:
      self.last_update_at = state.now
      return

    elapsed = min(max(state.now - self.last_update_at, 0.0), 1.0)
    self.last_update_at = state.now
    if state.ego_speed > 1.0:
      self.drive_time.health_drive_seconds += elapsed
      self.drive_time.duty_drive_seconds += elapsed
      self.drive_time.stationary_seconds = 0.0
      return

    self.drive_time.stationary_seconds += elapsed
    if self.drive_time.stationary_seconds >= self.SHORT_BREAK_SECONDS:
      self.drive_time.health_drive_seconds = 0.0
    if self.drive_time.stationary_seconds >= self.FULL_BREAK_SECONDS:
      self.drive_time.duty_drive_seconds = 0.0

  def update(self, state: ObserverInput) -> tuple[Prompt, float]:
    self._update_drive_time(state)

    drowsiness = self._confirmed(
      state.ego_speed > self.DRIVER_HEALTH_MIN_SPEED
      and state.driver_face_probability >= 0.5
      and state.driver_sleep_probability >= 0.65
      and state.driver_sunglasses_probability < 0.8,
      "drowsiness_since",
      state.now,
      self.DROWSINESS_CONFIRMATION,
    )

    attention_needed = (
      state.driver_alert_level >= 1
      or (state.driver_distracted and state.driver_awareness_percent <= 85)
    )
    attention = attention_needed and not self.previous_attention_needed
    self.previous_attention_needed = attention_needed

    lead_braking = self._confirmed(
      state.lead_status
      and state.ego_speed > 8.0
      and 4.0 < state.lead_distance < max(25.0, state.ego_speed * 3.0)
      and state.lead_acceleration < -1.0
      and state.lead_probability >= 0.5
      and not state.brake_pressed
      and state.ego_acceleration > -1.5,
      "lead_braking_since",
      state.now,
      self.LEAD_BRAKING_CONFIRMATION,
    )

    closing_fast = (
      state.lead_status
      and state.ego_speed > 8.0
      and 3.0 < state.lead_distance < max(20.0, state.ego_speed * 3.0)
      and state.lead_relative_speed < -3.0
      and state.lead_distance / -state.lead_relative_speed < 4.0
      and not state.brake_pressed
      and state.ego_acceleration > -1.5
    )
    slowing_traffic = self._confirmed(
      closing_fast,
      "slowing_since",
      state.now,
      self.SLOWING_CONFIRMATION,
    )

    lead_departed = False
    lead_stopped = (
      state.lead_status
      and state.ego_speed < 0.5
      and 2.0 < state.lead_distance < 35.0
      and state.lead_speed < 0.5
    )
    if lead_stopped:
      if self.stopped_lead_since is None:
        self.stopped_lead_since = state.now
        self.stopped_lead_distance = state.lead_distance
    elif self.stopped_lead_since is not None:
      stopped_duration = state.now - self.stopped_lead_since
      lead_departed = (
        state.lead_status
        and state.ego_speed < 0.5
        and stopped_duration >= self.STOPPED_CONFIRMATION
        and (state.lead_speed > 1.0 or state.lead_distance - self.stopped_lead_distance > 0.75)
      )
      self.stopped_lead_since = None

    rest_required = (
      state.ego_speed > self.DRIVER_HEALTH_MIN_SPEED
      and self.drive_time.duty_drive_seconds >= self.REST_REQUIRED_AFTER
    )
    rest_recommended = (
      state.ego_speed > self.DRIVER_HEALTH_MIN_SPEED
      and not rest_required
      and self.drive_time.health_drive_seconds >= self.REST_RECOMMENDED_AFTER
    )

    if drowsiness and self._ready(Prompt.DROWSINESS, state.now, self.DROWSINESS_COOLDOWN):
      self.drowsiness_since = None
      return Prompt.DROWSINESS, 1.0
    if attention and self._ready(Prompt.ATTENTION, state.now, self.ATTENTION_COOLDOWN):
      return Prompt.ATTENTION, 1.0
    if lead_braking and self._ready(Prompt.LEAD_BRAKING, state.now, self.LEAD_BRAKING_COOLDOWN):
      self.lead_braking_since = None
      return Prompt.LEAD_BRAKING, 0.85
    if slowing_traffic and self._ready(Prompt.SLOWING_TRAFFIC, state.now, self.SLOWING_TRAFFIC_COOLDOWN):
      self.slowing_since = None
      return Prompt.SLOWING_TRAFFIC, 0.8
    if lead_departed and self._ready(Prompt.LEAD_DEPARTED, state.now, self.LEAD_DEPARTED_COOLDOWN):
      return Prompt.LEAD_DEPARTED, 0.9
    if rest_required and self._ready(Prompt.REST_REQUIRED, state.now, self.REST_REQUIRED_COOLDOWN):
      return Prompt.REST_REQUIRED, 1.0
    if rest_recommended and self._ready(Prompt.REST_RECOMMENDED, state.now, self.REST_RECOMMENDED_COOLDOWN):
      return Prompt.REST_RECOMMENDED, 0.8

    return Prompt.NONE, 0.0


def restore_drive_time_state(raw: dict | bytes | str | None, wall_time: float) -> DriveTimeState:
  try:
    payload = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
    state = DriveTimeState(
      health_drive_seconds=max(0.0, float(payload["healthDriveSeconds"])),
      duty_drive_seconds=max(0.0, float(payload["dutyDriveSeconds"])),
      stationary_seconds=max(0.0, float(payload["stationarySeconds"])),
    )
    state.stationary_seconds += max(0.0, wall_time - float(payload["updatedAt"]))
  except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    return DriveTimeState()

  if state.stationary_seconds >= RoadObserver.SHORT_BREAK_SECONDS:
    state.health_drive_seconds = 0.0
  if state.stationary_seconds >= RoadObserver.FULL_BREAK_SECONDS:
    state.duty_drive_seconds = 0.0
  return state


def serialize_drive_time_state(state: DriveTimeState, wall_time: float) -> dict:
  return {
    "version": 1,
    "updatedAt": wall_time,
    "healthDriveSeconds": round(state.health_drive_seconds, 1),
    "dutyDriveSeconds": round(state.duty_drive_seconds, 1),
    "stationarySeconds": round(state.stationary_seconds, 1),
  }


def main() -> None:
  params = Params()
  drive_time_state = restore_drive_time_state(
    params.get("RoadObserverDriveState"),
    time.time(),  # noqa: TID251  # Wall time carries stopped duration across reboots.
  )
  observer = RoadObserver(drive_time_state)
  sm = messaging.SubMaster(["carState", "radarState", "driverMonitoringState", "driverStateV2"])
  pm = messaging.PubMaster(["roadObserverState"])
  rk = Ratekeeper(5)
  last_persist_at = time.monotonic()

  while True:
    sm.update(0)
    prompt, confidence = Prompt.NONE, 0.0
    now = time.monotonic()

    if params.get_bool("RoadObserverEnabled") and sm.all_checks():
      lead = sm["radarState"].leadOne
      monitoring = sm["driverMonitoringState"]
      driver_state = sm["driverStateV2"]
      driver = driver_state.rightDriverData if monitoring.isRHD else driver_state.leftDriverData
      state = ObserverInput(
        now=now,
        ego_speed=sm["carState"].vEgo,
        ego_acceleration=sm["carState"].aEgo,
        brake_pressed=sm["carState"].brakePressed,
        lead_status=lead.status,
        lead_distance=lead.dRel,
        lead_relative_speed=lead.vRel,
        lead_speed=lead.vLeadK,
        lead_acceleration=lead.aLeadK,
        lead_probability=lead.modelProb,
        driver_alert_level=monitoring.alertLevel.raw,
        driver_distracted=monitoring.visionPolicyState.isDistracted,
        driver_awareness_percent=monitoring.visionPolicyState.awarenessPercent,
        driver_face_probability=driver.faceProb,
        driver_sleep_probability=driver.sleepProb,
        driver_sunglasses_probability=driver.sunglassesProb,
      )
      prompt, confidence = observer.update(state)
      if prompt != Prompt.NONE and not params.get_bool(PROMPT_PARAM[prompt]):
        prompt, confidence = Prompt.NONE, 0.0

    if now - last_persist_at >= 300.0:
      params.put(
        "RoadObserverDriveState",
        serialize_drive_time_state(
          observer.drive_time,
          time.time(),  # noqa: TID251  # Monotonic time cannot persist across reboots.
        ),
      )
      last_persist_at = now

    msg = messaging.new_message("roadObserverState")
    msg.valid = sm.all_checks()
    msg.roadObserverState.prompt = int(prompt)
    msg.roadObserverState.confidence = confidence
    pm.send("roadObserverState", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
