# Italian Road Observer

This fork adds an advisory-only Italian voice observer for comma 4. It uses
signals already produced locally by openpilot and does not require an Internet
connection.

## Spoken observations

- `Guarda la strada.` when driver monitoring sees sustained distraction, before
  or at the first stock attention alert.
- `Il veicolo davanti è partito.` after a confirmed stop when the tracked lead
  vehicle moves away.
- `Attenzione, traffico in rallentamento.` after sustained rapid closing on a
  tracked lead vehicle.
- `Attenzione, pedone sulla traiettoria.` for a repeatedly detected pedestrian
  inside the estimated driving corridor.
- `Attenzione, ciclista sulla traiettoria.` for a repeatedly detected cyclist
  inside the estimated driving corridor.
- `Semaforo rosso/giallo/verde.` for a repeatedly detected, centered traffic
  light. Green is only announced after a confirmed red light while stopped.

Stock openpilot alerts always have audio priority. The observer is enabled by
default on this branch and can be disabled in Settings under
`Italian road observer`.

## Local road perception

`roadperceptionmodeld` samples the road camera at 2 Hz and runs the official
YOLOX-Nano 416x416 COCO model locally on the comma 4 Qualcomm backend. It keeps
only person, bicycle, and traffic-light detections, applies class-aware NMS,
requires three matching observations, and estimates whether pedestrians or
cyclists are inside a conservative image-space driving corridor.

Traffic-light color is estimated only inside a confirmed traffic-light crop.
It does not infer which traffic light legally controls the current lane, so
these messages are observations, not driving instructions.

`Road scene detection (beta)` and `Road hazard voice alerts (beta)` are disabled
by default. Enable scene detection while parked for device validation, then
enable voice only after reviewing local drives for false positives, thermal
load, and model latency. When enabled, structured shadow-mode observations are
published in `customReservedRawData0`.

Settings also provide individual switches for driver-attention, lead-vehicle,
slowing-traffic, pedestrian, cyclist, and traffic-light announcements. Turning
off the road-hazard voice master prevents further local perception
announcements without disabling stock openpilot safety sounds. Turning off
Italian road observer separately prevents attention, lead, and slowing-traffic
announcements.

## Build and release

This branch changes parameter keys, Cap'n Proto schemas, messaging services,
and UI code. It must not contain the stock `prebuilt` marker unless all affected
comma 4 native artifacts have been rebuilt from this exact commit. Without that
guarantee, `launch_chffrplus.sh` skips the build and can start the modified
Python and UI code against stale stock binaries.

Before publishing an installer commit:

1. Complete a clean `scons` build in the repository's supported build
   environment.
2. Install while parked with a reliable power source and keep SSH available.
3. Confirm manager, UI, pandad, `roadobserverd`, and
   `roadperceptionmodeld` remain healthy before driving.
4. Keep road-perception voice alerts disabled until latency, thermals, and
   false positives have been reviewed.

## RoadTalk companion

The optional `roadtalkd` service lets the RoadTalk Android head-unit app query
basic status, control only the custom observer audio, capture a forward-road
photo, or record a one- or two-minute road video. It binds to the comma four's
private Wi-Fi interface and never exposes CAN, shell, process control,
arbitrary parameter writes, steering, braking, acceleration, or engagement.

Pair while parked:

1. Connect the comma four and Android radio to the same private Wi-Fi network.
2. Enable ADB in comma settings.
3. Open RoadTalk and tap `Pair Comma 4`, then disable ADB again.

The app discovers comma over UDP port 7767 and uses HTTP port 7766. Pairing
creates `/persist/roadtalk/shared_secret`; subsequent requests require an HMAC
signature, timestamp, and one-time nonce. Pair only on a trusted private
hotspot because the one-time bootstrap exchange is not encrypted.

To revoke an old Android installation, run this while parked over SSH, then
pair the new installation:

```bash
rm -f /persist/roadtalk/shared_secret
```

Supported commands are status, observer mute, observer unmute, quiet mode for
15, 30, 60, or 120 minutes, a road photo, and a road video limited to one or
two minutes. Photos and videos remain in `/data/media/0/roadtalk`; the service
keeps the newest 20 photos and 8 videos. Video capture copies the existing
low-bitrate `qcamera.ts` stream instead of starting another encoder.

The app's local command path works without Internet. GPT Realtime maps natural
requests to the same fixed allowlist and receives a road photo only when the
user explicitly asks it to analyze the road ahead.

## Safety boundary

The observer only publishes advisory messages to `soundd`. It never writes CAN,
changes steering, applies braking, or changes openpilot engagement.

The detector is not safety-certified, does not see outside the camera field of
view, can miss or misclassify objects, and can select a traffic light belonging
to another lane. Network inference is intentionally not in the alert path:
connectivity and cloud latency are unsuitable for time-critical road warnings.
Do not act on a spoken traffic-light color without verifying it visually, and
do not treat this observer as a substitute for an attentive driver.
