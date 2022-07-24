import numpy as np
import onnxruntime
import torch
import soxr
class VadInferencer():

    def __init__(self, path):

        self.session = onnxruntime.InferenceSession(path)
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1

        self.reset_states()

    def reset_states(self):
        self._h = np.zeros((2, 1, 64)).astype('float32')
        self._c = np.zeros((2, 1, 64)).astype('float32')

    def predict(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[::step]
            sr = 16000

        if x.shape[0] > 1:
            raise ValueError("Onnx model does not support batching")

        if sr not in [16000]:
            raise ValueError(f"Supported sample rates: {[16000]}")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        ort_inputs = {'input': x.numpy(), 'h0': self._h, 'c0': self._c}
        ort_outs = self.session.run(None, ort_inputs)
        out, self._h, self._c = ort_outs

        out = torch.tensor(out).squeeze(2)[:, 1]  # make output type match JIT analog

        return out

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def has_speech_activity(self, audio_chunk):

        audio_int16 = np.fromstring(audio_chunk, dtype=np.int16)
        # pcm = audio_int16.tostring()

        audio_float32 = self.int2float(audio_int16)

        audio_float32 = soxr.resample(audio_float32, 48000, 16000, quality=soxr.VHQ)

        confidence = self.predict(torch.from_numpy(audio_float32), 16000).item()

        return confidence > 0.5




