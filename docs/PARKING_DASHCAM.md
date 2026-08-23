# Parking dashcam

The `Record while parked` toggle keeps only the forward road camera recording
after ignition turns off while the Panda harness reports a vehicle connection.
The screen stays off, and vehicle-control, CAN, driver-monitoring, and
road-model processes remain stopped.

Each ignition-off session is limited to one hour. Recording also stops if the
filtered vehicle supply drops below 11.8 V or the device reaches its critical
thermal state. Turning ignition on starts a normal route with the full onroad
camera configuration.
