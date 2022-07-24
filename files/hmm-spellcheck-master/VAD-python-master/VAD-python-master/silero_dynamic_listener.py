import io
import numpy as np
import torch

torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt

torchaudio.set_audio_backend("soundfile")
import pyaudio
import contextlib
import wave

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True, onnx=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


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


# Taken from utils_vad.py
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound


FORMAT = pyaudio.paInt16
CHANNELS = 1
# SAMPLE_RATE = 16000
CHUNK = 4096
# CHUNK = 1024
audio = pyaudio.PyAudio()

num_samples = 1536

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=48000,
                    input=True,
                    frames_per_buffer=CHUNK)

import soxr



voiced_confidences = []
frames = []
print("Started Recording")
while True:
    try:
        audio_chunk = stream.read(num_samples)

        audio_int16 = np.fromstring(audio_chunk, dtype=np.int16)

        pcm = audio_int16.tostring()
        frames.append(pcm)

        audio_float32 = int2float(audio_int16)

        rs = soxr.ResampleStream(
            48000,  # input samplerate
            16000,  # target samplerate
            1,  # channel(s)
            dtype='float32'  # data type (default = 'float32')
        )

        # eof = True
        # while not eof:
        #     # Get chunk
        #     ...
        #
        #     audio_float32 = rs.resample_chunk(
        #         audio_float32,  # 1D(mono) or 2D(frames, channels) array input
        #         last=eof  # Set True at end of input
        #     )
        audio_float32 = soxr.resample(audio_float32, 48000, 16000, quality=soxr.VHQ)

        # get the confidences and add them to the list to plot them later
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)
    except KeyboardInterrupt:
        break

print("Stopped the recording")

write_wave(f"silero_recording.wav", b''.join([f for f in frames]), 16000)

# plot the confidences for the speech
plt.figure(figsize=(20, 6))
plt.plot(voiced_confidences)
plt.savefig("silero_audio.jpg")
plt.show()


