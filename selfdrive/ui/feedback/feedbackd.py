#!/usr/bin/env python3
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER

FEEDBACK_MAX_DURATION = 10.0


def main():
  pm = messaging.PubMaster(['userBookmark', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton'])
  should_record_audio = False
  block_num = 0
  early_stop_triggered = False

  while True:
    sm.update()
    should_send_bookmark = False

    if should_record_audio and sm.updated['rawAudioData']:
      raw_audio = sm['rawAudioData']
      msg = messaging.new_message('audioFeedback', valid=True)
      msg.audioFeedback.audio.data = raw_audio.data
      msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
      msg.audioFeedback.blockNum = block_num
      block_num += 1
      if (block_num * SAMPLE_BUFFER / SAMPLE_RATE) >= FEEDBACK_MAX_DURATION or early_stop_triggered:  # Check for timeout or early stop
        should_send_bookmark = True  # send bookmark at end of audio segment
        should_record_audio = False
        early_stop_triggered = False
        cloudlog.info("10-second recording completed or second button press - stopping audio feedback")
      pm.send('audioFeedback', msg)

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      should_send_bookmark = True

    if should_send_bookmark:
      msg = messaging.new_message('userBookmark', valid=True)
      pm.send('userBookmark', msg)


if __name__ == '__main__':
  main()
