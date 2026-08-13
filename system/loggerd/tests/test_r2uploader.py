import datetime
import json
import os
from pathlib import Path

from openpilot.system.loggerd.r2uploader import (
  ConfigError,
  R2Config,
  R2Uploader,
  UPLOAD_ATTR_NAME,
  UPLOAD_ATTR_VALUE,
  _canonical_uri,
  _signed_headers,
  load_config,
)


VALID_CONFIG = {
  "endpoint": "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
  "bucket": "golf",
  "prefix": "comma4/dashcam",
  "access_key_id": "test-access-key",
  "secret_access_key": "test-secret-key",
  "files": ["fcamera.hevc", "qcamera.ts"],
}


class FakeResponse:
  def __init__(self, status_code: int) -> None:
    self.status_code = status_code
    self.closed = False

  def close(self) -> None:
    self.closed = True


class FakeSession:
  def __init__(self, status_code: int = 200) -> None:
    self.status_code = status_code
    self.requests = []

  def put(self, url, data, headers, timeout):
    self.requests.append(
      {
        "url": url,
        "body": data.read(),
        "headers": headers,
        "timeout": timeout,
      }
    )
    return FakeResponse(self.status_code)


def write_config(path: Path, value=VALID_CONFIG, mode: int = 0o600) -> None:
  path.write_text(json.dumps(value), encoding="utf-8")
  path.chmod(mode)


def test_load_config_requires_private_permissions(tmp_path):
  path = tmp_path / "r2.json"
  write_config(path, mode=0o644)

  try:
    load_config(path)
  except ConfigError as error:
    assert "permissions" in str(error)
  else:
    raise AssertionError("group-readable credentials must be rejected")


def test_config_rejects_non_r2_endpoint():
  value = {**VALID_CONFIG, "endpoint": "https://example.com"}
  try:
    R2Config.from_mapping(value)
  except ConfigError as error:
    assert "r2.cloudflarestorage.com" in str(error)
  else:
    raise AssertionError("non-R2 endpoint must be rejected")


def test_signing_is_stable():
  config = R2Config.from_mapping(VALID_CONFIG)
  uri = _canonical_uri(config.bucket, "comma4/dashcam/route 1/fcamera.hevc")
  headers = _signed_headers(
    config,
    uri,
    "0" * 64,
    123,
    datetime.datetime(2026, 8, 13, 12, 0, tzinfo=datetime.UTC),
  )

  assert uri == "/golf/comma4/dashcam/route%201/fcamera.hevc"
  assert headers["Content-Length"] == "123"
  assert headers["x-amz-date"] == "20260813T120000Z"
  assert headers["Authorization"].endswith("Signature=41bcb38e4dfe97256fbd01b6a6a3844b090be5a7ce8d745db9b216927a4c925d")


def test_locked_and_uploaded_recordings_are_skipped(tmp_path):
  config = R2Config.from_mapping(VALID_CONFIG)

  complete = tmp_path / "2026-08-13--10-00-00--0"
  complete.mkdir()
  front = complete / "fcamera.hevc"
  front.write_bytes(b"front")
  quick = complete / "qcamera.ts"
  quick.write_bytes(b"quick")
  os.setxattr(quick, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)

  active = tmp_path / "2026-08-13--10-00-00--1"
  active.mkdir()
  (active / "fcamera.hevc").write_bytes(b"active")
  (active / "fcamera.hevc.lock").touch()

  candidates = list(R2Uploader(config, tmp_path).candidates())
  assert [candidate.path for candidate in candidates] == [front]


def test_successful_upload_marks_original_recording(tmp_path):
  config = R2Config.from_mapping(VALID_CONFIG)
  route = tmp_path / "2026-08-13--10-00-00--0"
  route.mkdir()
  recording = route / "fcamera.hevc"
  recording.write_bytes(b"dashcam data")
  session = FakeSession()
  uploader = R2Uploader(config, tmp_path, session=session)

  assert uploader.step()
  assert os.getxattr(recording, UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE
  assert session.requests[0]["body"] == b"dashcam data"
  assert session.requests[0]["url"].endswith("/golf/comma4/dashcam/2026-08-13--10-00-00--0/fcamera.hevc")
  assert config.secret_access_key not in session.requests[0]["headers"]["Authorization"]


def test_failed_upload_does_not_mark_recording(tmp_path):
  config = R2Config.from_mapping(VALID_CONFIG)
  route = tmp_path / "2026-08-13--10-00-00--0"
  route.mkdir()
  recording = route / "fcamera.hevc"
  recording.write_bytes(b"dashcam data")
  uploader = R2Uploader(config, tmp_path, session=FakeSession(status_code=403))

  assert not uploader.step()
  try:
    os.getxattr(recording, UPLOAD_ATTR_NAME)
  except OSError:
    pass
  else:
    raise AssertionError("failed uploads must remain eligible for retry")
