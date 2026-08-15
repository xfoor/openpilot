# Italian Road Observer

This fork adds an advisory-only Italian voice observer for comma 4. It uses
signals already produced locally by openpilot and does not require an Internet
connection.

## Spoken observations

- `Guarda la strada.` when driver monitoring sees sustained distraction, before
  or at the first stock attention alert.
- `Il veicolo davanti è partito.` after a confirmed stop when the tracked lead
  vehicle moves away.
- `Attenzione, il veicolo davanti sta frenando.` when a close, confidently
  tracked lead decelerates for at least half a second and the driver has not
  started braking.
- `Attenzione, traffico in rallentamento.` after sustained rapid closing on a
  tracked lead vehicle.
- `Sembri stanco. Fermati appena possibile.` after the driver model reports
  sustained sleep probability while the car is moving.
- `Guida da due ore. Programma una pausa.` after two hours of driving without a
  15-minute break.
- `È necessaria una pausa. Fermati appena possibile.` after 4.5 hours of driving
  without a 45-minute break.
- `Attenzione, stai uscendo dalla corsia a sinistra.` for a detected left lane
  departure while openpilot is not steering.
- `Attenzione, stai uscendo dalla corsia a destra.` for a detected right lane
  departure while openpilot is not steering.
- `Stai uscendo spesso dalla corsia. Fermati e fai una pausa.` after five
  separate lane departures within 15 minutes. This reminder repeats no more
  than once every 30 minutes.
- `Attenzione, pedone sulla traiettoria.` for a tracked pedestrian occupying
  the calibrated future path or moving toward it.
- `Attenzione, ciclista sulla traiettoria.` for a tracked cyclist occupying the
  calibrated future path or moving toward it.

Stock openpilot alerts always have audio priority. The observer is enabled by
default on this branch and can be disabled in Settings under
`Italian road observer`.

## Lane departure assistance

Directional lane speech uses openpilot's existing lane-departure signal. The
stock detector requires a speed above 50 km/h, a reliable lane estimate, no
recent turn signal, and openpilot lateral control to be inactive. A continuous
departure is treated as one event. Separate directional warnings are debounced
for four seconds.

The directional voice replaces the stock generic lane chime only when it can
actually start. If the observer, attention-and-lane setting, or lane-departure
setting is disabled, or observer audio is temporarily muted, the stock chime
remains unchanged. Repeated-drift speech uses the existing driver-health
setting; when that setting is disabled, the normal directional warning is
still spoken.

This feature is warning-only while the driver is steering. Comma can actively
center the car only after the driver deliberately engages openpilot on a
supported road. It never silently takes steering control in response to a lane
departure.

## Local road perception

`roadperceptionmodeld` samples the road camera at 2 Hz and runs a compiled
YOLOX-Nano 320x320 COCO model locally on the comma 4 Qualcomm backend. It keeps
only person, bicycle, and traffic-light detections and applies class-aware NMS.
Pedestrian and cyclist boxes are tracked across frames. Their ground contact
points are projected through the comma's live camera calibration and compared
with the future path already produced by openpilot's driving model. A warning
requires either repeated occupancy of that path or tracked lateral motion that
would enter it within 2.5 seconds and inside the speed-dependent warning
distance. If calibration, camera identity, or the future path is unavailable,
no pedestrian or cyclist voice prompt is eligible.

Traffic-light color is estimated only inside a confirmed traffic-light crop.
It does not infer which traffic light legally controls the current lane.
Traffic-light observations are therefore hard-blocked from `soundd`, even if a
raw observation incorrectly carries a voice flag. They remain shadow
diagnostics for developing a dedicated signal model and lane association.

`Road scene detection (beta)` and `Pedestrian and cyclist voice alerts (beta)`
are disabled by default. Enable scene detection while parked for device
validation, then enable voice only after reviewing local drives for false
positives and missed crossings. When enabled, structured shadow-mode
observations are published in `customReservedRawData0`, including track ID,
projected distance, path offset, lateral speed, and the reason a risk qualified.

An offline screen of 191 retained Slovenia urban frames produced 33 compiled
person detections and two traffic-light candidates. The compiled model's median
inference time was 27 ms and its 95th percentile was 29 ms on the comma 4. A
focused 207-frame, 2 Hz replay around every person candidate produced 240
detections but no path-occupancy or predicted-crossing alert: the people were
parked or beside the driven path. This is useful negative coverage, not a
pedestrian-crossing recall measurement. Traffic-light speech remains disabled
until a signal-specific model and lane-association strategy pass broader
positive and negative replay.

Settings also provide individual switches for driver-attention and lane,
lead-vehicle, lead-braking, slowing-traffic, driver-health, pedestrian, and
cyclist announcements. Turning off the perception voice master prevents
pedestrian and cyclist announcements without disabling stock openpilot safety
sounds. Turning off Italian road observer separately prevents driver, lane,
lead, and traffic announcements from this observer.

The driving clocks count time above 1 m/s and persist across ignition cycles.
A stationary period of 15 minutes resets the two-hour health reminder. A
stationary period of 45 minutes also resets the 4.5-hour driving-limit warning.
Rest and drowsiness speech waits until the car is above 5 m/s. These are
wellness reminders, not a certified tachograph or legal compliance system.

## Build and release

This branch changes parameter keys, Cap'n Proto schemas, messaging services,
and UI code. It must not contain the stock `prebuilt` marker unless all affected
comma 4 native artifacts have been rebuilt from this exact commit. Without that
guarantee, `launch_chffrplus.sh` skips the build and can start the modified
Python and UI code against stale stock binaries.

Before publishing an installer commit:

1. Rebuild every affected native artifact from the exact installer commit in
   the comma 4 build environment. This change requires at least
   `common/params_pyx.so`, `cereal/messaging/bridge`, and
   `system/loggerd/loggerd`.
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

On the tested Volkswagen Golf Mk7, openpilot longitudinal control is
unsupported. The observer cannot apply a gentle brake or regulate road speed
while openpilot is disengaged; braking remains the responsibility of the driver
and the vehicle's stock ACC/AEB systems.

The detector is not safety-certified, does not see outside the camera field of
view, can miss or misclassify objects, and can select a traffic light belonging
to another lane. Network inference is intentionally not in the alert path:
connectivity and cloud latency are unsuitable for time-critical road warnings.
Do not act on a spoken traffic-light color without verifying it visually, and
do not treat this observer as a substitute for an attentive driver.

The COCO perception model does not detect or read European speed-limit signs.
Speed-limit voice warnings require a separately validated sign-recognition or
offline map-matching source. No speed-limit value was present in the archived
navigation, map, or Golf CAN signals.
