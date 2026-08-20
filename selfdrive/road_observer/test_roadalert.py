from openpilot.selfdrive.road_observer.roadalert import (
  ALERT_TTL_SECONDS,
  AlertBroker,
  radio_acknowledged,
  radio_audio_ready,
)


def test_broker_delivers_and_acknowledges_event(tmp_path):
  broker = AlertBroker(tmp_path)
  assert broker.publish(100, 2, 0.9, now=10.0)

  event = broker.next_event(after=0, wait_seconds=0.0, now_fn=lambda: 10.1)

  assert event is not None
  assert event.phrase == "leadDeparted"
  assert broker.acknowledge(100)
  assert radio_acknowledged(100, broker.ack_path)


def test_broker_drops_expired_and_duplicate_events(tmp_path):
  broker = AlertBroker(tmp_path)
  assert broker.publish(100, 1, 1.0, now=10.0)
  assert not broker.publish(100, 1, 1.0, now=10.1)

  event = broker.next_event(
    after=0,
    wait_seconds=0.0,
    now_fn=lambda: 10.0 + ALERT_TTL_SECONDS,
  )

  assert event is None
  assert not broker.acknowledge(100)


def test_new_broker_accepts_cursor_from_previous_comma_boot(tmp_path):
  broker = AlertBroker(tmp_path)
  assert broker.publish(10, 1, 1.0, now=10.0)

  event = broker.next_event(after=999_999, wait_seconds=0.0, now_fn=lambda: 10.1)

  assert event is not None
  assert event.event_id == 10


def test_radio_readiness_expires(tmp_path):
  broker = AlertBroker(tmp_path)
  broker.ready_path.parent.mkdir(parents=True, exist_ok=True)
  broker.ready_path.write_text("100.0", encoding="utf-8")

  assert radio_audio_ready(now=101.0, ready_path=broker.ready_path)
  assert not radio_audio_ready(now=103.0, ready_path=broker.ready_path)


def test_directional_junction_alert_is_available_to_radio(tmp_path):
  broker = AlertBroker(tmp_path)

  assert broker.publish(321, 17, 0.88, now=10.0)
  event = broker.next_event(after=0, wait_seconds=0.0, now_fn=lambda: 10.1)

  assert event is not None
  assert event.phrase == "junctionVehicleRight"
  assert event.priority == 3
