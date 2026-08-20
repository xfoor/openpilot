from cereal import log

from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigMultiParamToggle
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.ui_state import ui_state

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants


class TogglesLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._personality_toggle = BigMultiParamToggle("driving personality", "LongitudinalPersonality", ["aggressive", "standard", "relaxed"])
    self._experimental_btn = BigParamControl("experimental mode", "ExperimentalMode")
    is_metric_toggle = BigParamControl("use metric units", "IsMetric")
    ldw_toggle = BigParamControl("lane departure warnings", "IsLdwEnabled")
    always_on_dm_toggle = BigParamControl("always-on driver monitor", "AlwaysOnDM")
    road_observer_toggle = BigParamControl(tr("Italian road observer"), "RoadObserverEnabled")
    road_observer_attention_toggle = BigParamControl(tr("Driver attention voice alerts"), "RoadObserverAttentionEnabled")
    road_observer_curve_toggle = BigParamControl(tr("Curve acceleration voice alerts"), "RoadObserverCurveEnabled")
    road_observer_lead_toggle = BigParamControl(tr("Lead vehicle voice alerts"), "RoadObserverLeadEnabled")
    road_observer_lead_pull_away_toggle = BigParamControl(tr("Lead pull-away voice alerts"), "RoadObserverLeadPullAwayEnabled")
    road_observer_lead_braking_toggle = BigParamControl(tr("Lead braking voice alerts"), "RoadObserverLeadBrakingEnabled")
    road_observer_slowing_toggle = BigParamControl(tr("Slowing traffic voice alerts"), "RoadObserverSlowingTrafficEnabled")
    road_observer_driver_health_toggle = BigParamControl(tr("Driver health and rest alerts"), "RoadObserverDriverHealthEnabled")
    road_perception_toggle = BigParamControl(tr("Road scene detection (beta)"), "RoadPerceptionEnabled")
    road_perception_voice_toggle = BigParamControl(tr("Pedestrian and cyclist voice alerts (beta)"), "RoadPerceptionVoiceEnabled")
    road_perception_pedestrian_toggle = BigParamControl(tr("Pedestrian voice alerts"), "RoadPerceptionPedestrianEnabled")
    road_perception_cyclist_toggle = BigParamControl(tr("Cyclist voice alerts"), "RoadPerceptionCyclistEnabled")
    record_front = BigParamControl("record & upload driver camera", "RecordFront", toggle_callback=restart_needed_callback)
    record_mic = BigParamControl("record & upload mic audio", "RecordAudio", toggle_callback=restart_needed_callback)
    enable_openpilot = BigParamControl("enable openpilot", "OpenpilotEnabledToggle", toggle_callback=restart_needed_callback)

    self._scroller.add_widgets([
      self._personality_toggle,
      self._experimental_btn,
      is_metric_toggle,
      ldw_toggle,
      always_on_dm_toggle,
      road_observer_toggle,
      road_observer_attention_toggle,
      road_observer_curve_toggle,
      road_observer_lead_toggle,
      road_observer_lead_pull_away_toggle,
      road_observer_lead_braking_toggle,
      road_observer_slowing_toggle,
      road_observer_driver_health_toggle,
      road_perception_toggle,
      road_perception_voice_toggle,
      road_perception_pedestrian_toggle,
      road_perception_cyclist_toggle,
      record_front,
      record_mic,
      enable_openpilot,
    ])

    # Toggle lists
    self._refresh_toggles = (
      ("ExperimentalMode", self._experimental_btn),
      ("IsMetric", is_metric_toggle),
      ("IsLdwEnabled", ldw_toggle),
      ("AlwaysOnDM", always_on_dm_toggle),
      ("RoadObserverEnabled", road_observer_toggle),
      ("RoadObserverAttentionEnabled", road_observer_attention_toggle),
      ("RoadObserverCurveEnabled", road_observer_curve_toggle),
      ("RoadObserverLeadEnabled", road_observer_lead_toggle),
      ("RoadObserverLeadPullAwayEnabled", road_observer_lead_pull_away_toggle),
      ("RoadObserverLeadBrakingEnabled", road_observer_lead_braking_toggle),
      ("RoadObserverSlowingTrafficEnabled", road_observer_slowing_toggle),
      ("RoadObserverDriverHealthEnabled", road_observer_driver_health_toggle),
      ("RoadPerceptionEnabled", road_perception_toggle),
      ("RoadPerceptionVoiceEnabled", road_perception_voice_toggle),
      ("RoadPerceptionPedestrianEnabled", road_perception_pedestrian_toggle),
      ("RoadPerceptionCyclistEnabled", road_perception_cyclist_toggle),
      ("RecordFront", record_front),
      ("RecordAudio", record_mic),
      ("OpenpilotEnabledToggle", enable_openpilot),
    )

    enable_openpilot.set_enabled(lambda: not ui_state.engaged)
    record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    record_mic.set_enabled(lambda: not ui_state.engaged)

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    super()._update_state()

    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._personality_toggle.set_value(self._personality_toggle._options[personality])
      ui_state.personality = personality

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    # CP gating for experimental mode
    if ui_state.CP is not None:
      if ui_state.has_longitudinal_control:
        self._experimental_btn.set_visible(True)
        self._personality_toggle.set_visible(True)
      else:
        # no long for now
        self._experimental_btn.set_visible(False)
        self._experimental_btn.set_checked(False)
        self._personality_toggle.set_visible(False)
        ui_state.params.remove("ExperimentalMode")

    # Refresh toggles from params to mirror external changes
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))
