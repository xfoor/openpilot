# Cloudflare R2 dashcam archive

The comma 4 can copy completed forward-camera segments to a private Cloudflare
R2 bucket. This archive is independent of comma's stock uploader and RoadTalk's
manual photo/video captures.

`r2uploader` has no vehicle-control access and runs only while openpilot is
offroad. It never reads a segment that still has a logger lock, never deletes a
local recording, and marks a file only after R2 accepts the complete object.
The stock logger's normal storage-pressure deletion policy remains unchanged.

## Recordings

- `fcamera.hevc` is the full-quality forward camera, normally about 75 MB per
  minute. One hour is roughly 4.5 GB before any R2 lifecycle deletion.
- `qcamera.ts` is the low-bitrate forward copy, normally about 2-3 MB per
  minute and easier to preview.

The configured `files` list controls which copies are uploaded. Objects use:

```text
<prefix>/<route>/<recording>
```

For example:

```text
comma4/golf/dashcam/2026-08-13--10-00-00--4/fcamera.hevc
```

## Private configuration

Credentials are deliberately excluded from git. Create
`/persist/roadtalk/r2.json` as the `comma` user with mode `0600`:

```json
{
  "endpoint": "https://ACCOUNT_ID.r2.cloudflarestorage.com",
  "bucket": "golf",
  "prefix": "comma4/golf/dashcam",
  "access_key_id": "R2_ACCESS_KEY_ID",
  "secret_access_key": "R2_SECRET_ACCESS_KEY",
  "files": ["fcamera.hevc", "qcamera.ts"]
}
```

Use an R2 token scoped to object read/write for only the selected bucket. Do
not reuse a Cloudflare account API token. The uploader rejects symlinked,
non-owner, group-readable, non-HTTPS, and non-R2 configurations.

After changing the file, restart openpilot while parked. Uploads resume
idempotently: if an object was accepted by R2 but the local marker was not
written, the same object key is overwritten on the next attempt.
