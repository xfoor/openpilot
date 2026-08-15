import enum
import math
from collections import deque


class LanePrompt(enum.IntEnum):
  NONE = 0
  DEPARTURE_LEFT = 13
  DEPARTURE_RIGHT = 14
  REPEATED_DEPARTURE = 15


LANE_PROMPT_PARAM = {
  LanePrompt.DEPARTURE_LEFT: "RoadObserverAttentionEnabled",
  LanePrompt.DEPARTURE_RIGHT: "RoadObserverAttentionEnabled",
  LanePrompt.REPEATED_DEPARTURE: "RoadObserverDriverHealthEnabled",
}


class LaneDepartureAlerter:
  DEPARTURE_COOLDOWN = 4.0
  EVENT_GAP = 2.0
  REPEATED_WINDOW = 15.0 * 60.0
  REPEATED_COUNT = 5
  REPEATED_COOLDOWN = 30.0 * 60.0

  def __init__(self):
    self.departure_active = False
    self.last_departure_at = -math.inf
    self.last_prompt_at = dict.fromkeys(LanePrompt, -math.inf)
    self.departures: deque[float] = deque()

  def update(self, now: float, left: bool, right: bool,
             lateral_active: bool) -> LanePrompt:
    active = not lateral_active and (left or right)
    new_departure = active and not self.departure_active
    self.departure_active = active
    if not new_departure or now - self.last_departure_at < self.EVENT_GAP:
      return LanePrompt.NONE

    self.last_departure_at = now
    self.departures.append(now)
    while self.departures and now - self.departures[0] > self.REPEATED_WINDOW:
      self.departures.popleft()

    if (
      len(self.departures) >= self.REPEATED_COUNT
      and now - self.last_prompt_at[LanePrompt.REPEATED_DEPARTURE] >= self.REPEATED_COOLDOWN
    ):
      self.last_prompt_at[LanePrompt.REPEATED_DEPARTURE] = now
      return LanePrompt.REPEATED_DEPARTURE

    prompt = LanePrompt.DEPARTURE_LEFT if left else LanePrompt.DEPARTURE_RIGHT
    if now - self.last_prompt_at[prompt] < self.DEPARTURE_COOLDOWN:
      return LanePrompt.NONE
    self.last_prompt_at[prompt] = now
    return prompt
