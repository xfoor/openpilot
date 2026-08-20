# Italian Road Observer

This fork adds an advisory-only Italian voice observer for comma 4. It uses
signals already produced locally by openpilot and does not require an Internet
connection.

## Spoken observations

- `Guarda la strada.` when driver monitoring sees sustained distraction, before
  or at the first stock attention alert.
- `Il veicolo davanti è ripartito.` after a confirmed stop when the tracked lead
  vehicle moves away.
- `Attenzione, il veicolo davanti frena.` when a close, confidently
  tracked lead decelerates for at least half a second and the driver has not
  started braking.
- `Attenzione, il traffico rallenta.` after sustained rapid closing on a
  tracked lead vehicle.
- `Sembri stanco. Fermati appena possibile.` after the driver model reports
  sustained sleep probability while the car is moving.
- `Guida da due ore. Programma una pausa.` after two hours of driving without a
  15-minute break.
- `È necessaria una pausa. Fermati appena possibile.` after 4.5 hours of driving
  without a 45-minute break.
- `Attenzione, pedone sulla traiettoria.` for a tracked pedestrian occupying
  the calibrated future path or moving toward it.
- `Attenzione, ciclista sulla traiettoria.` for a tracked cyclist occupying the
  calibrated future path or moving toward it.
- `Attenzione, l'auto sta accelerando in curva.` when factory ACC accelerates
  for at least half a second in a tight, low-speed curve without a pedal input.
- `Il veicolo davanti si sta allontanando.` when a confidently tracked lead
  pulls away for at least one second but factory ACC is not matching it.
- `Controlla a sinistra: c'è un veicolo.` or `Controlla a destra: c'è un
  veicolo.` for a confirmed side-vehicle risk during a low-speed turn.
- `Controlla il limite di velocità.` on Comma after a speed-limit sign is
  confirmed. When the radio is connected, its neural voice announces the
  detected number and tells the driver to reduce speed if the current speed or
  factory-ACC set speed is more than 3 km/h above it.

Stock openpilot alerts always have audio priority. The observer is enabled by
default on this branch and can be disabled in Settings under
`Italian road observer`.

## Local road perception

`roadperceptionmodeld` samples a camera at 2 Hz and runs a compiled YOLOX-Nano
320x320 COCO model locally on the comma 4 Qualcomm backend. It normally uses
the road camera, then switches to the wide road camera below 15 m/s while a
turn signal is active or steering exceeds 25 degrees. It keeps person, bicycle,
vehicle, and traffic-light detections and applies class-aware NMS. Pedestrian,
cyclist, car, motorcycle, bus, and truck boxes are tracked across frames. Their
ground contact points are projected through the selected camera's live
calibration and compared with the future path already produced by openpilot's
driving model.

A pedestrian or cyclist warning requires either repeated occupancy of that
path or tracked lateral motion that would enter it within 2.5 seconds and
inside the speed-dependent warning distance. A moving side vehicle must be
tracked for three frames, move toward the path by at least 0.8 m/s, and reach
the path within three seconds. A stationary side-vehicle reminder is more
restricted: wide camera, active turn signal, at most 6 m/s, four frames,
confidence of at least 0.72, within 18 metres and 7 metres of the path, no
recent driver check of that side, and a confident look toward the opposite
side within 1.5 seconds. Moving path conflicts remain eligible regardless of
gaze. If calibration, camera identity, or the future path is unavailable, no
road-user or junction voice prompt is eligible.

Junction speech reports the side where a vehicle was observed. It does not
claim that the vehicle is waiting, approaching, or legally has priority. The
driver must verify the scene and right-of-way.

Traffic-light color is estimated only inside a confirmed traffic-light crop.
It does not infer which traffic light legally controls the current lane.
Traffic-light observations are therefore hard-blocked from `soundd`, even if a
raw observation incorrectly carries a voice flag. They remain shadow
diagnostics for developing a dedicated signal model and lane association.

Speed-limit recognition runs independently on the road camera as a two-stage
pipeline: a 256x256 nano detector scans a direct high-resolution crop of the
right roadside and a small GTSRB classifier reads each candidate sign. These
two models run on a two-thread CPU backend so they do not compete with the
Qualcomm accelerator used by openpilot's driving model. A reading must stay on
the same spatial track for at least two sampled frames over 0.4 seconds,
include a strong classifier result, have a sufficiently large and growing sign
box, and win a confidence-weighted vote before it can be announced.
Single-frame readings are discarded. Supported limits are 20, 30, 50, 60, 70,
80, 100, and 120 km/h; end-of-limit and other sign classes produce no limit.

This feature is advisory-only. It does not write a cruise set speed, press
virtual steering-wheel buttons, accelerate, brake, or send CAN messages. A
driver may use the spoken value to change the factory ACC setting with the
physical controls after verifying the sign.

`Road scene detection (beta)` and `Road scene voice alerts (beta)` are disabled
by default. Junction, pedestrian, and cyclist voice switches are independent
under that master. Enable scene detection while parked for device validation,
then enable only the required event classes. When enabled, structured
observations are published in `customReservedRawData0`, including a unique
voice event ID, selected camera, side, track ID, projected distance, path
offset, lateral speed, recent-side-check result, and qualification reason.
`Speed limit sign detection (beta)` is a separate switch and is disabled by
default. Its voice switch is independent of the road-scene voice master.

An offline screen of 191 retained Slovenia urban frames produced 33 compiled
person detections and two traffic-light candidates. The compiled model's median
inference time was 27 ms and its 95th percentile was 29 ms on the comma 4. A
focused 207-frame, 2 Hz replay around every person candidate produced 240
detections but no path-occupancy or predicted-crossing alert: the people were
parked or beside the driven path. This is useful negative coverage, not a
pedestrian-crossing recall measurement. Traffic-light speech remains disabled
until a signal-specific model and lane-association strategy pass broader
positive and negative replay.

For the 18 August junction reviewed after road testing, the narrow road camera
lost side coverage while the wide camera retained both approaches. The exact
compiled QCOM model detected vehicles in 14 of 16 retained one-second
wide-camera frames around the stop line. Steady inference took 32–40 ms after a
2.5–3.8-second first-load warmup. This validates camera coverage and detector
visibility for that event, but not universal junction recall.

Settings also provide individual switches for driver-attention, lead-vehicle,
lead-braking, slowing-traffic, driver-health, curve-acceleration,
lead-pull-away, junction-vehicle, pedestrian, and cyclist announcements.
Turning off the perception voice master prevents all three perception
announcement classes without disabling stock openpilot safety sounds. Turning
off Italian road observer separately prevents driver and lead announcements
from this observer.

The curve and lead-pull-away rules only run when the vehicle uses factory ACC;
they do not request acceleration or braking. An 18 August 2026 replay covering
about 80 minutes produced three curve advisories, including both reviewed
roundabout accelerations, and two lead-pull-away advisories.

A 974-frame, 1 Hz screen of the retained 19 August drive showed that individual
single-stage speed-sign frames can be wrong. Focused 4 Hz full-resolution
replay of three visible sign sequences showed that separating detection from
classification read the sustained signs as 50, 70, and 50 km/h. That evidence
is why automatic cruise-speed changes are hard-blocked and multi-frame
confirmation is mandatory. At the production 2 Hz sampling rate, a sign that is
detectable for less than half a second can be missed.

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
basic status and GPS, receive observer speech events, control only the custom
observer audio, capture a forward-road photo, or record a one- or two-minute
road video. It binds to the comma four's private Wi-Fi interface and never
exposes CAN, shell, process control, arbitrary parameter writes, steering,
braking, acceleration, or engagement.

RoadTalk protocol 2 makes the radio's packaged neural Italian voice the primary
speech channel. Every event is authenticated and expires after five seconds.
The packaged prompts use Azure AI Speech's Italian Isabella Dragon HD neural
voice as 48 kHz mono PCM and require no cloud connection while driving.
When the radio reports that audio is ready, `soundd` waits up to 300 ms for the
radio's playback acknowledgement. A missing acknowledgement immediately falls
back to the matching packaged neural voice on Comma; Android system TTS is a
last fallback on the radio. Stock openpilot safety alerts always remain local
and always take priority. The companion also records journeys locally from
Comma GPS, falls back to recent radio GPS, and can export a GeoJSON trace.

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

On the tested Volkswagen Golf Mk7.5 configuration, the vehicle's factory radar
ACC supplies longitudinal control while openpilot supplies lateral control.
The observer does not replace or override factory ACC and cannot apply a gentle
brake or regulate road speed; braking and speed selection remain the
responsibility of the driver and the vehicle's stock ACC/AEB systems.

The detector is not safety-certified, does not see outside the camera field of
view, can miss or misclassify objects, and can select a traffic light belonging
to another lane. Network inference is intentionally not in the alert path:
connectivity and cloud latency are unsuitable for time-critical road warnings.
Do not act on a spoken traffic-light color without verifying it visually, and
do not treat this observer as a substitute for an attentive driver.

Speed-limit recognition can miss, misread, or announce a sign for a side road.
It can also miss signs shown only on the far left. It is not a legal-speed
authority and has no map-based confirmation in the current build. The driver
must verify every announced limit.
