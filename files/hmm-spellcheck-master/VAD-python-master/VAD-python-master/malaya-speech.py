import malaya_speech
import numpy as np
from malaya_speech import Pipeline

y, sr = malaya_speech.load('test.wav')

len(y), sr

y_int = malaya_speech.astype.float_to_int(y)
vad = malaya_speech.vad.webrtc(minimum_amplitude=int(np.quantile(np.abs(y_int), 0.2)))

malaya_speech.vad.available_model()

model_factor1 = malaya_speech.vad.deep_model(model = 'marblenet-factor1')
model_factor3 = malaya_speech.vad.deep_model(model = 'marblenet-factor3')

frames_int = list(malaya_speech.utils.generator.frames(y_int, 30, sr))
frames = list(malaya_speech.utils.generator.frames(y, 30, sr))

# frames_webrtc = [(frame, vad(frame)) for frame in frames_int]
frames = list(malaya_speech.utils.generator.frames(y, 50, sr))
frames_deep_factor1 = [(frame, model_factor1(frame)) for frame in frames]

frames_deep_factor3 = [(frame, model_factor3(frame)) for frame in frames]

p = Pipeline()
pipeline = (
    p.batching(5)
    .foreach_map(model_factor1.predict)
    .flatten()
)
p.visualize()

result = p.emit(frames)
result.keys()

frames_deep_factor1_batch = [(frame, result['flatten'][no]) for no, frame in enumerate(frames)]
len(frames_deep_factor1) == len(frames_deep_factor1_batch)



probs = [(frame, model_factor1.predict_proba([frame])) for frame in frames]
probs[:5]


malaya_speech.extra.visualization.visualize_vad(y, frames_webrtc, sr)