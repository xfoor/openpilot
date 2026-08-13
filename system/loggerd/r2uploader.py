#!/usr/bin/env python3
from __future__ import annotations

import datetime
import errno
import hashlib
import hmac
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import stat
import threading
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit

import requests


CONFIG_PATH = Path("/data/roadtalk/r2.json")
MAX_CONFIG_BYTES = 16 * 1024
SUPPORTED_FILES = frozenset(("fcamera.hevc", "qcamera.ts"))
DEFAULT_FILES = ("fcamera.hevc", "qcamera.ts")
UPLOAD_ATTR_NAME = "user.r2_upload"
UPLOAD_ATTR_VALUE = b"1"
CONFIG_RETRY_SECONDS = 60
IDLE_SECONDS = 30
MAX_BACKOFF_SECONDS = 300

_BUCKET_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])$")
_ENDPOINT_PATTERN = re.compile(r"^[0-9a-f]{32}(?:\.[a-z0-9-]+)?\.r2\.cloudflarestorage\.com$")
_KEY_PART_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class ConfigError(ValueError):
  pass


@dataclass(frozen=True)
class R2Config:
  endpoint: str
  bucket: str
  prefix: str
  access_key_id: str
  secret_access_key: str
  files: tuple[str, ...]

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any]) -> R2Config:
    allowed_keys = {
      "endpoint",
      "bucket",
      "prefix",
      "access_key_id",
      "secret_access_key",
      "files",
    }
    if unknown_keys := set(value) - allowed_keys:
      raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown_keys))}")

    endpoint = _required_string(value, "endpoint").rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
      raise ConfigError("endpoint must be an HTTPS Cloudflare R2 URL")
    try:
      endpoint_port = parsed.port
    except ValueError:
      raise ConfigError("endpoint port is invalid") from None
    if parsed.path not in ("", "/") or endpoint_port not in (None, 443):
      raise ConfigError("endpoint must not include a path or non-standard port")
    hostname = (parsed.hostname or "").lower()
    if not _ENDPOINT_PATTERN.fullmatch(hostname):
      raise ConfigError("endpoint must use r2.cloudflarestorage.com")

    bucket = _required_string(value, "bucket")
    if not _BUCKET_PATTERN.fullmatch(bucket):
      raise ConfigError("bucket name is invalid")

    prefix = _required_string(value, "prefix").strip("/")
    if not _PREFIX_PATTERN.fullmatch(prefix) or any(part in ("", ".", "..") for part in prefix.split("/")):
      raise ConfigError("prefix is invalid")

    access_key_id = _required_string(value, "access_key_id")
    secret_access_key = _required_string(value, "secret_access_key")
    if len(access_key_id) > 128 or len(secret_access_key) > 256:
      raise ConfigError("R2 credential length is invalid")
    if any(character.isspace() for character in access_key_id + secret_access_key):
      raise ConfigError("R2 credentials must not contain whitespace")

    raw_files = value.get("files", DEFAULT_FILES)
    if not isinstance(raw_files, list) or not raw_files:
      raise ConfigError("files must be a non-empty list")
    if any(not isinstance(name, str) or name not in SUPPORTED_FILES for name in raw_files):
      raise ConfigError("files contains an unsupported recording name")
    files = tuple(dict.fromkeys(raw_files))

    return cls(endpoint, bucket, prefix, access_key_id, secret_access_key, files)


@dataclass(frozen=True)
class UploadCandidate:
  key: str
  path: Path
  size: int
  mtime_ns: int
  device: int
  inode: int


def _required_string(value: Mapping[str, Any], key: str) -> str:
  item = value.get(key)
  if not isinstance(item, str) or not item:
    raise ConfigError(f"{key} is required")
  if any(ord(character) < 0x20 for character in item):
    raise ConfigError(f"{key} contains control characters")
  return item


def load_config(path: Path = CONFIG_PATH) -> R2Config:
  flags = os.O_RDONLY
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW

  try:
    fd = os.open(path, flags)
  except FileNotFoundError:
    raise ConfigError("configuration file is missing") from None
  except OSError as error:
    raise ConfigError("configuration file cannot be opened") from error

  try:
    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode):
      raise ConfigError("configuration path must be a regular file")
    if file_stat.st_uid != os.geteuid():
      raise ConfigError("configuration file must be owned by the uploader user")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
      raise ConfigError("configuration file permissions must be 0600 or stricter")

    with os.fdopen(fd, "r", encoding="utf-8") as stream:
      fd = -1
      raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
      raise ConfigError("configuration file is too large")
    value = json.loads(raw)
  except json.JSONDecodeError as error:
    raise ConfigError("configuration file is not valid JSON") from error
  except UnicodeError as error:
    raise ConfigError("configuration file is not valid UTF-8") from error
  except OSError as error:
    raise ConfigError("configuration file cannot be read") from error
  finally:
    if fd >= 0:
      os.close(fd)

  if not isinstance(value, dict):
    raise ConfigError("configuration root must be an object")
  return R2Config.from_mapping(value)


def _route_sort(name: str) -> list[str]:
  return [part.rjust(10, "0") for part in name.rsplit("--", 1)]


def _get_upload_marker(path: Path) -> bytes | None:
  try:
    return os.getxattr(path, UPLOAD_ATTR_NAME)
  except OSError as error:
    if error.errno == errno.ENODATA or (hasattr(errno, "ENOATTR") and error.errno == errno.ENOATTR):
      return None
    raise


def _canonical_uri(bucket: str, key: str) -> str:
  parts = (bucket, *key.split("/"))
  return "/" + "/".join(quote(part, safe="-_.~") for part in parts)


def _signing_key(secret: str, date_stamp: str, region: str = "auto") -> bytes:
  date_key = hmac.new(("AWS4" + secret).encode(), date_stamp.encode(), hashlib.sha256).digest()
  region_key = hmac.new(date_key, region.encode(), hashlib.sha256).digest()
  service_key = hmac.new(region_key, b"s3", hashlib.sha256).digest()
  return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def _signed_headers(config: R2Config, canonical_uri: str, payload_hash: str, size: int, now: datetime.datetime) -> dict[str, str]:
  parsed = urlsplit(config.endpoint)
  host = parsed.netloc
  amz_date = now.astimezone(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
  date_stamp = amz_date[:8]
  canonical_headers = "\n".join(
    (
      f"host:{host}",
      f"x-amz-content-sha256:{payload_hash}",
      f"x-amz-date:{amz_date}",
      "",
    )
  )
  signed_header_names = "host;x-amz-content-sha256;x-amz-date"
  canonical_request = "\n".join(
    (
      "PUT",
      canonical_uri,
      "",
      canonical_headers,
      signed_header_names,
      payload_hash,
    )
  )
  credential_scope = f"{date_stamp}/auto/s3/aws4_request"
  string_to_sign = "\n".join(
    (
      "AWS4-HMAC-SHA256",
      amz_date,
      credential_scope,
      hashlib.sha256(canonical_request.encode()).hexdigest(),
    )
  )
  signature = hmac.new(
    _signing_key(config.secret_access_key, date_stamp),
    string_to_sign.encode(),
    hashlib.sha256,
  ).hexdigest()
  authorization = (
    "AWS4-HMAC-SHA256 " + f"Credential={config.access_key_id}/{credential_scope}, " + f"SignedHeaders={signed_header_names}, Signature={signature}"
  )
  return {
    "Authorization": authorization,
    "Content-Length": str(size),
    "Host": host,
    "x-amz-content-sha256": payload_hash,
    "x-amz-date": amz_date,
  }


def _sha256(stream: BinaryIO) -> str:
  digest = hashlib.sha256()
  while chunk := stream.read(4 * 1024 * 1024):
    digest.update(chunk)
  stream.seek(0)
  return digest.hexdigest()


class R2Uploader:
  def __init__(self, config: R2Config, root: Path, session: Any = requests) -> None:
    self.config = config
    self.root = root
    self.session = session
    self.last_status_code = 0
    self.last_error = ""

  def candidates(self) -> Iterator[UploadCandidate]:
    try:
      route_names = sorted((path.name for path in self.root.iterdir()), key=_route_sort)
    except OSError:
      return

    for route_name in route_names:
      if not _KEY_PART_PATTERN.fullmatch(route_name):
        continue
      route_path = self.root / route_name
      try:
        route_stat = route_path.lstat()
        names = os.listdir(route_path)
      except OSError:
        continue
      if not stat.S_ISDIR(route_stat.st_mode) or stat.S_ISLNK(route_stat.st_mode):
        continue
      if any(name.endswith(".lock") for name in names):
        continue

      for name in self.config.files:
        path = route_path / name
        try:
          file_stat = path.lstat()
          if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode) or file_stat.st_size == 0:
            continue
          if _get_upload_marker(path) == UPLOAD_ATTR_VALUE:
            continue
        except OSError:
          continue

        yield UploadCandidate(
          key=f"{self.config.prefix}/{route_name}/{name}",
          path=path,
          size=file_stat.st_size,
          mtime_ns=file_stat.st_mtime_ns,
          device=file_stat.st_dev,
          inode=file_stat.st_ino,
        )

  @staticmethod
  def _candidate_unchanged(candidate: UploadCandidate, file_stat: os.stat_result) -> bool:
    return (
      stat.S_ISREG(file_stat.st_mode)
      and file_stat.st_size == candidate.size
      and file_stat.st_mtime_ns == candidate.mtime_ns
      and file_stat.st_dev == candidate.device
      and file_stat.st_ino == candidate.inode
    )

  def upload(self, candidate: UploadCandidate) -> bool:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW

    response = None
    try:
      fd = os.open(candidate.path, flags)
      with os.fdopen(fd, "rb") as stream:
        if not self._candidate_unchanged(candidate, os.fstat(stream.fileno())):
          return False
        payload_hash = _sha256(stream)
        if not self._candidate_unchanged(candidate, os.fstat(stream.fileno())):
          return False

        uri = _canonical_uri(self.config.bucket, candidate.key)
        headers = _signed_headers(
          self.config,
          uri,
          payload_hash,
          candidate.size,
          datetime.datetime.now(datetime.UTC),
        )
        headers["Content-Type"] = "video/hevc" if candidate.path.name == "fcamera.hevc" else "video/mp2t"
        response = self.session.put(
          self.config.endpoint + uri,
          data=stream,
          headers=headers,
          timeout=(10, 180),
        )
        self.last_status_code = response.status_code
        if response.status_code not in (200, 201, 204):
          self.last_error = f"HTTP {response.status_code}"
          return False

      current_stat = candidate.path.lstat()
      if not self._candidate_unchanged(candidate, current_stat):
        self.last_error = "recording changed during upload"
        return False
      os.setxattr(candidate.path, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
      self.last_error = ""
      return True
    except (OSError, requests.RequestException) as error:
      self.last_error = type(error).__name__
      return False
    finally:
      if response is not None:
        response.close()

  def step(self) -> bool | None:
    candidate = next(self.candidates(), None)
    if candidate is None:
      return None
    return self.upload(candidate)


def main(exit_event: threading.Event | None = None) -> None:
  from openpilot.common.swaglog import cloudlog
  from openpilot.system.hardware.hw import Paths

  if exit_event is None:
    exit_event = threading.Event()

  config_path = Path(os.getenv("R2_UPLOAD_CONFIG", str(CONFIG_PATH)))
  root = Path(Paths.log_root())
  backoff = 5.0
  last_config_error = ""

  while not exit_event.is_set():
    try:
      config = load_config(config_path)
      last_config_error = ""
    except ConfigError as error:
      reason = str(error)
      if reason != last_config_error:
        cloudlog.warning("r2uploader disabled: %s", reason)
        last_config_error = reason
      exit_event.wait(CONFIG_RETRY_SECONDS)
      continue

    uploader = R2Uploader(config, root)
    success = uploader.step()
    if success is None:
      backoff = 5.0
      exit_event.wait(IDLE_SECONDS)
    elif success:
      backoff = 5.0
      cloudlog.info("r2uploader upload complete")
      exit_event.wait(0.1)
    else:
      cloudlog.warning(
        "r2uploader upload failed: status=%d error=%s",
        uploader.last_status_code,
        uploader.last_error,
      )
      exit_event.wait(backoff)
      backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


if __name__ == "__main__":
  main()
