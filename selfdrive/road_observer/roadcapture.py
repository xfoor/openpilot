import io
from pathlib import Path
import threading
import time
from typing import Any


CAPTURE_DIR = Path("/data/media/0/roadtalk")
LOG_ROOT = Path("/data/media/0/realdata")
TS_PACKET_BYTES = 188
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_PHOTOS = 20
MAX_VIDEOS = 8
PHOTO_COOLDOWN_SECONDS = 5


class RoadCapture:
  def __init__(self) -> None:
    self.lock = threading.Lock()
    self.video_active = False
    self.video_minutes = 0
    self.video_started_at = 0
    self.last_photo_at = 0.0
    self.latest_photo: Path | None = None
    self.latest_video: Path | None = None
    self.last_error = ""

  @staticmethod
  def _prepare_dir() -> None:
    CAPTURE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

  @staticmethod
  def _cleanup(pattern: str, keep: int) -> None:
    files = sorted(CAPTURE_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in files[keep:]:
      try:
        path.unlink()
      except OSError:
        pass

  def status(self) -> dict[str, Any]:
    with self.lock:
      return {
        "videoActive": self.video_active,
        "videoMinutes": self.video_minutes,
        "videoStartedAt": self.video_started_at,
        "latestPhoto": self.latest_photo.name if self.latest_photo else "",
        "latestVideo": self.latest_video.name if self.latest_video else "",
        "captureError": self.last_error,
      }

  def take_photo(self) -> tuple[int, dict[str, Any]]:
    with self.lock:
      if time.monotonic() - self.last_photo_at < PHOTO_COOLDOWN_SECONDS:
        return 429, {"error": "Wait a few seconds before taking another road photo"}
      self.last_photo_at = time.monotonic()

    try:
      # Keep camera/numpy/Pillow imports off roadtalkd's boot path.
      from PIL import Image
      from msgq.visionipc import VisionIpcClient, VisionStreamType
      from openpilot.system.camerad.snapshot import extract_image

      client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
      client.connect(True)
      frame = client.recv()
      if frame is None:
        raise RuntimeError("road camera did not return a frame")

      image = Image.fromarray(extract_image(frame))
      image.thumbnail((1280, 1280))
      encoded = io.BytesIO()
      image.save(encoded, "JPEG", quality=82, optimize=True)

      self._prepare_dir()
      timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
      destination = CAPTURE_DIR / f"road-{timestamp}.jpg"
      temporary = destination.with_suffix(".tmp")
      temporary.write_bytes(encoded.getvalue())
      temporary.replace(destination)
      self._cleanup("road-*.jpg", MAX_PHOTOS)
      with self.lock:
        self.latest_photo = destination
        self.last_error = ""
      return 200, {
        "ok": True,
        "message": "Road photo captured",
        "capture": destination.name,
      }
    except Exception as error:
      with self.lock:
        self.last_error = str(error)[:160]
      return 503, {"error": "The forward road camera is not ready"}

  def photo_bytes(self) -> bytes | None:
    with self.lock:
      path = self.latest_photo
    if path is None:
      photos = sorted(CAPTURE_DIR.glob("road-*.jpg"), key=lambda item: item.stat().st_mtime)
      path = photos[-1] if photos else None
    try:
      return path.read_bytes() if path and path.parent == CAPTURE_DIR else None
    except OSError:
      return None

  def start_video(self, minutes: int) -> tuple[int, dict[str, Any]]:
    if minutes not in (1, 2):
      return 400, {"error": "Road video must be one or two minutes"}
    with self.lock:
      if self.video_active:
        return 409, {"error": "A road video is already recording"}
      self.video_active = True
      self.video_minutes = minutes
      self.video_started_at = int(time.time())
      self.last_error = ""
    threading.Thread(
      target=self._record_video,
      args=(minutes,),
      name="roadtalk-video",
      daemon=True,
    ).start()
    return 202, {
      "ok": True,
      "message": f"Recording a {minutes} minute road video on Comma 4",
      "minutes": minutes,
    }

  @staticmethod
  def _qcamera_files() -> list[Path]:
    try:
      return [path for path in LOG_ROOT.glob("*--*/qcamera.ts") if path.is_file()]
    except OSError:
      return []

  @staticmethod
  def _append_new_packets(source: Path, destination: Any, offset: int) -> int:
    size = source.stat().st_size
    complete_size = size - (size % TS_PACKET_BYTES)
    if complete_size <= offset:
      return offset
    with source.open("rb") as stream:
      stream.seek(offset)
      remaining = complete_size - offset
      while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
          break
        destination.write(chunk)
        remaining -= len(chunk)
    return complete_size - remaining

  def _record_video(self, minutes: int) -> None:
    destination: Path | None = None
    try:
      self._prepare_dir()
      existing = self._qcamera_files()
      if not existing:
        raise RuntimeError("qcamera stream is unavailable")
      current = max(existing, key=lambda path: path.stat().st_mtime_ns)
      start_epoch = time.time()
      tracked = [current]
      offsets = {
        current: ((current.stat().st_size + TS_PACKET_BYTES - 1) // TS_PACKET_BYTES) * TS_PACKET_BYTES,
      }
      timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
      destination = CAPTURE_DIR / f"road-{timestamp}-{minutes}m.ts"
      deadline = time.monotonic() + minutes * 60

      with destination.open("wb") as output:
        while time.monotonic() < deadline:
          candidates = []
          for path in self._qcamera_files():
            if path in offsets:
              continue
            try:
              if path.stat().st_mtime >= start_epoch - 2:
                candidates.append(path)
            except OSError:
              continue
          for path in sorted(candidates, key=lambda item: item.stat().st_ctime_ns):
            tracked.append(path)
            offsets[path] = 0

          for path in tracked:
            try:
              offsets[path] = self._append_new_packets(path, output, offsets[path])
            except OSError:
              continue
          if output.tell() >= MAX_VIDEO_BYTES:
            raise RuntimeError("road video reached its storage limit")
          time.sleep(0.5)

        for path in tracked:
          try:
            offsets[path] = self._append_new_packets(path, output, offsets[path])
          except OSError:
            continue

      if destination.stat().st_size < TS_PACKET_BYTES * 100:
        raise RuntimeError("qcamera stream did not produce enough video")
      self._cleanup("road-*.ts", MAX_VIDEOS)
      with self.lock:
        self.latest_video = destination
        self.last_error = ""
    except Exception as error:
      if destination is not None:
        try:
          destination.unlink()
        except OSError:
          pass
      with self.lock:
        self.last_error = str(error)[:160]
    finally:
      with self.lock:
        self.video_active = False
        self.video_minutes = 0
        self.video_started_at = 0
