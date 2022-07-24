from vad import VoiceActivityDetector
import argparse
import json
from pydub import AudioSegment
from pydub.playback import play
import sys

def save_to_file(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Analyze input wave-file and save detected speech interval to json file.')
#     parser.add_argument('inputfile', metavar='INPUTWAVE',
#                         help='the full path to input wave file')
#     parser.add_argument('outputfile', metavar='OUTPUTFILE',
#                         help='the full path to output json file to save detected speech intervals')
#     args = parser.parse_args()

inputfile = "wav-sample.wav"
outputfile = "results.json"
#
v = VoiceActivityDetector(inputfile)
raw_detection = v.detect_speech()
speech_labels = v.convert_windows_to_readible_labels(raw_detection)

save_to_file(speech_labels, outputfile)


v = VoiceActivityDetector(inputfile)

v.plot_detected_speech_regions()



