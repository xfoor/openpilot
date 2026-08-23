# Parking dashcam

The `Record road + cabin while parked` toggle keeps the forward road and cabin
driver cameras recording after ignition turns off while the Panda harness
reports a vehicle connection. The driver camera records in parking mode even
when normal driver-camera uploads are disabled. Its infrared illumination
continues to follow the camera exposure for recording in darkness. The screen
stays off, and vehicle-control, CAN, driver-monitoring, and road-model
processes remain stopped.

Each ignition-off session is limited to one hour. Recording also stops if the
filtered vehicle supply drops below 11.8 V or the device reaches its critical
thermal state. Turning ignition on starts a normal route with the full onroad
camera configuration.
