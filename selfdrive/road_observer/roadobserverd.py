#!/usr/bin/env python3
import enum
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


@dataclass
class ObserverInput:
  now: float
  ego_speed: float
  lead_status: bool
  lead_distance: float
  lead_relative_speed: float
  lead_speed: float
  driver_alert_level: int
  driver_distracted: bool
  driver_awareness_percent: int


class RoadObserver:
  ATTENTION_COOLDOWN = 20.0
  LEAD_DEPARTED_COOLDOWN = 8.0
  SLOWING_TRAFFIC_COOLDOWN = 15.0
  STOPPED_CONFIRMATION = 2.0
  SLOWING_CONFIRMATION = 0.6

  def __init__(self):
    self.last_prompt_at = dict.fromkeys(Prompt, -math.inf)
    self.previous_attention_needed = False
    self.stopped_lead_since: float | None = None
    self.stopped_lead_distance = 0.0
    self.slowing_since: float | None = None

  def _ready(self, prompt: Prompt, now: float, cooldown: float) -> bool:
    if now - self.last_prompt_at[prompt] < cooldown:
      return False
    self.last_prompt_at[prompt] = now
    return True

  def update(self, state: ObserverInput) -> tuple[Prompt, float]:
    attention_needed = (
      state.driver_alert_level >= 1
      or (state.driver_distracted and state.driver_awareness_percent <= 85)
    )
    if attention_needed and not self.previous_attention_needed:
      self.previous_attention_needed = True
      if self._ready(Prompt.ATTENTION, state.now, self.ATTENTION_COOLDOWN):
        return Prompt.ATTENTION, 1.0
    self.previous_attention_needed = attention_needed

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
      if lead_departed and self._ready(Prompt.LEAD_DEPARTED, state.now, self.LEAD_DEPARTED_COOLDOWN):
        return Prompt.LEAD_DEPARTED, 0.9
    else:
      self.stopped_lead_since = None

    closing_fast = (
      state.lead_status
      and state.ego_speed > 8.0
      and 3.0 < state.lead_distance < max(20.0, state.ego_speed * 3.0)
      and state.lead_relative_speed < -3.0
      and state.lead_distance / -state.lead_relative_speed < 4.0
    )
    if closing_fast:
      if self.slowing_since is None:
        self.slowing_since = state.now
      if (
        state.now - self.slowing_since >= self.SLOWING_CONFIRMATION
        and self._ready(Prompt.SLOWING_TRAFFIC, state.now, self.SLOWING_TRAFFIC_COOLDOWN)
      ):
        self.slowing_since = None
        return Prompt.SLOWING_TRAFFIC, 0.8
    else:
      self.slowing_since = None

    return Prompt.NONE, 0.0


def main() -> None:
  params = Params()
  observer = RoadObserver()
  sm = messaging.SubMaster(["carState", "radarState", "driverMonitoringState"])
  pm = messaging.PubMaster(["roadObserverState"])
  rk = Ratekeeper(5)

  while True:
    sm.update(0)
    prompt, confidence = Prompt.NONE, 0.0

    if params.get_bool("RoadObserverEnabled") and sm.all_checks():
      lead = sm["radarState"].leadOne
      state = ObserverInput(
        now=time.monotonic(),
        ego_speed=sm["carState"].vEgo,
        lead_status=lead.status,
        lead_distance=lead.dRel,
        lead_relative_speed=lead.vRel,
        lead_speed=lead.vLeadK,
        driver_alert_level=sm["driverMonitoringState"].alertLevel.raw,
        driver_distracted=sm["driverMonitoringState"].visionPolicyState.isDistracted,
        driver_awareness_percent=sm["driverMonitoringState"].visionPolicyState.awarenessPercent,
      )
      prompt, confidence = observer.update(state)

    msg = messaging.new_message("roadObserverState")
    msg.valid = sm.all_checks()
    msg.roadObserverState.prompt = int(prompt)
    msg.roadObserverState.confidence = confidence
    pm.send("roadObserverState", msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
