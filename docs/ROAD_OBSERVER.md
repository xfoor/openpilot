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

Stock openpilot alerts always have audio priority. The observer is enabled by
default on this branch and can be disabled in Settings under
`Italian road observer`.

## Safety boundary

The observer only publishes advisory messages to `soundd`. It never writes CAN,
changes steering, applies braking, or changes openpilot engagement.

Pedestrian recognition and traffic-light color recognition are intentionally
not included. The stock driving model does not expose validated semantic
classes for those objects, and an added detector must first be tested on comma
4 for accuracy, latency, thermal load, and camera-frame impact. Do not treat
this observer as a substitute for an attentive driver.
