import speech_recognition as sr
import contextlib
import webrtcvad
import numpy as np
import pyaudio
from time import monotonic
import numpy as np
import soundfile as sf
import torch
import soxr
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

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound



RATE = 48000
CHUNK = int(RATE / 10)
FORMAT = pyaudio.paInt16
CHANNELS = 1
import wave

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# print(p.get_device_info_by_index(1))

MIN_CONSECUTIVE_FRAMES = 5
MAX_WAIT = 5

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,onnx=True)


def has_speech_activity(frames, number_of_frames=MIN_CONSECUTIVE_FRAMES, percentage=0.80):

    if len(frames) < number_of_frames:
        return False

    num_voiced = len([1 for frame in frames[-number_of_frames:] if model(torch.from_numpy(frame), 16000).item() > 0.5])
    print("num_voiced: ", num_voiced)
    return num_voiced > percentage * number_of_frames




consecutive_frames_reached = 0
concat_last_frame = []
int_last_frames = []
last_frames = []
has_spoken = False
wake_word_frames = []
float_frames = []
silent_time = None

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)

    audio_int16 = np.frombuffer(data, np.int16)
    audio_float32 = int2float(audio_int16)

    audio_float32 = soxr.resample(audio_float32, 48000, 16000, quality=soxr.VHQ)

    float_frames.append(audio_float32)

    if has_speech_activity(float_frames):
        silent_time = None
        print("Consecutive frames detected!!!")
        has_spoken = True
        wake_word_frames.append(data)

    else:
        if silent_time is None:
            silent_time = monotonic()
        elif monotonic() - silent_time > MAX_WAIT and has_spoken:
            print("Voice command ended!!!")

            break

r = sr.Recognizer()
y = write_wave(f"test_.wav", b''.join([f for f in wake_word_frames]), 48000)
# audio_source = b''
#
# for chunk in wake_word_frames:
#     audio_source += chunk

# audio_source = sr.AudioData(audio_bytes, 48000, 1)

audio_file_ = sr.AudioFile("test_.wav")
with audio_file_ as source:
  audio_file = r.record(source)
try:
    # recognised_text = r.recognize_google(text, language='en-US')
    recognised_text = r.recognize_google(audio_file, language='en-US')
    print(recognised_text)
    if recognised_text == "hey archie" or "hey archi" or "archie":
        print("Wake word detected")

except:
    print("i didn't get that...")