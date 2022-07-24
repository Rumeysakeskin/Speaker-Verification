#!/usr/bin/env python
# coding: utf-8
import collections
import contextlib
import sys
import wave
import webrtcvad


VAD_ROBUSTNESS = 3



def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
        
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

path = 'vad_test_file.wav'
audio, sample_rate = read_wave(path)
frames = frame_generator(30, audio, sample_rate)
frames = list(frames)




import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [30, 10]





vad = webrtcvad.Vad(VAD_ROBUSTNESS)

y = []
x = []
for i,frame in enumerate(frames):
    
    y.append(int(vad.is_speech(frame.bytes, sample_rate)))
    x.append(30*i/1000)

plt.plot(x,y,marker = 'o',ms = 3,ls = ':')
plt.yticks([0,1])
plt.xticks(range(0,25))
plt.savefig(f'vad({VAD_ROBUSTNESS}).png')
plt.show()