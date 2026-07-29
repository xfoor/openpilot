# RoadTalk bridge

This release keeps the stock openpilot driving stack and adds only the
authenticated RoadTalk companion service. It does not include the custom road
observer, perception model, or any vehicle-control endpoint.

`roadtalkd` binds HTTP port 7766 and UDP discovery port 7767 to the comma
device's private Wi-Fi address. It exposes:

- basic openpilot status;
- forward-road photo capture;
- one- or two-minute copies of the existing low-bitrate `qcamera.ts` stream;
- authenticated media listing, download, and deletion.

The service never exposes CAN, shell, process control, arbitrary Params writes,
steering, braking, acceleration, or engagement. Observer audio commands remain
in protocol version 1 for Android compatibility, but return an explicit
unavailable response on this stock release.

## Pairing

Pair only while parked on a trusted private hotspot:

1. Connect the comma device and Android head unit to the same Wi-Fi network.
2. Enable ADB in comma settings.
3. In RoadTalk, select `Pair Comma 4`.
4. Disable ADB after pairing.

The bootstrap exchange is not encrypted. Pairing is allowed only while
offroad with ADB enabled and creates `/persist/roadtalk/shared_secret`.
Subsequent requests require an HMAC signature, a current timestamp, and a
one-time nonce.

To revoke a pairing, remove `/persist/roadtalk/shared_secret` while parked and
pair again. Captures are stored in `/data/media/0/roadtalk`; the service retains
at most 20 photos and 8 videos.
