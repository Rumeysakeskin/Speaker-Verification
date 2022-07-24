import collections
import contextlib
import sys
import wave
import webrtcvad
import numpy as np
import pyaudio
from time import monotonic
import malaya_speech
import numpy as np
from malaya_speech import Pipeline

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

    return audio



def has_speech_activity(frames,number_of_frames = 25,percentage=0.85):
	# print("len_frames: ",len(frames))
	if len(frames) < number_of_frames:
		return False

	num_voiced = len([1 for frame in frames[-number_of_frames:] if vad.is_speech(frame, RATE)])
	print("num_voiced: ",num_voiced,end='\r')
	return num_voiced > percentage * number_of_frames

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

VAD_ROBUSTNESS = 1
vad = webrtcvad.Vad(VAD_ROBUSTNESS)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print(p.get_device_info_by_index(1))


MIN_CONSECUTIVE_FRAMES = 25
MAX_WAIT = 2


for i in range(1):
	consecutive_frames_reached = 0
	frames = []
	int_frames = []
	last_frames = []
	has_spoken = False

	silent_time = None


	while True:
	    raw = stream.read(CHUNK, exception_on_overflow=False)
	    _last_frame = np.fromstring(raw, dtype=np.int16)
	    
	    pcm = _last_frame.tostring()

	    if has_speech_activity(last_frames):
	        silent_time = None
	        print("Consecutive frames detected!!!")
	        has_spoken = True
	    else:
	        if silent_time is None:
	            silent_time = monotonic()
	        elif monotonic() - silent_time > MAX_WAIT and has_spoken:
	            print("Voice command ended!!!")
	            y = write_wave(f"test_{i}.wav",b''.join([f for f in frames]),RATE)
	            break
	    last_frames.append(_last_frame[0:480].tostring())

	    frames.append(pcm)

#
#
# y_int = malaya_speech.astype.float_to_int(y)
# vad = malaya_speech.vad.webrtc(minimum_amplitude=int(np.quantile(np.abs(y_int), 0.2)))
# malaya_speech.vad.available_model()
# model_factor1 = malaya_speech.vad.deep_model(model='marblenet-factor1')
# frames_int = list(malaya_speech.utils.generator.frames(y_int, 30, RATE))
#
#
# frames = list(malaya_speech.utils.generator.frames(last_frames, 30, RATE))
#
# frames_deep_factor1 = [(frame, model_factor1(frame)) for frame in frames]
#
# p = Pipeline()
# pipeline = (
# 	p.batching(5)
# 		.foreach_map(model_factor1.predict)
# 		.flatten()
# )
# p.visualize()
#
# result = p.emit(frames)
# result.keys()
#
# frames_deep_factor1_batch = [(frame, result['flatten'][no]) for no, frame in enumerate(frames)]
# len(frames_deep_factor1) == len(frames_deep_factor1_batch)
#
# malaya_speech.extra.visualization.visualize_vad(y, frames_deep_factor1, RATE)








