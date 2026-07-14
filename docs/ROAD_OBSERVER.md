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

`Road scene detection (beta)` is enabled by default and records structured
shadow-mode observations in `customReservedRawData0`. `Road hazard voice alerts
(beta)` is disabled by default. Enable voice only after reviewing local drives
for false positives, thermal load, and model latency.

## Safety boundary

The observer only publishes advisory messages to `soundd`. It never writes CAN,
changes steering, applies braking, or changes openpilot engagement.

The detector is not safety-certified, does not see outside the camera field of
view, can miss or misclassify objects, and can select a traffic light belonging
to another lane. Network inference is intentionally not in the alert path:
connectivity and cloud latency are unsuitable for time-critical road warnings.
Do not act on a spoken traffic-light color without verifying it visually, and
do not treat this observer as a substitute for an attentive driver.
