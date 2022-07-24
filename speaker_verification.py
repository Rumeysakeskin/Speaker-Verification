import time
import torch
import librosa
import pickle
import nemo
import nemo.collections.asr as nemo_asr
from sklearn.metrics.pairwise import cosine_similarity
from models import *
import numpy as np

class NemoSpeakerIdentifier():
    def __init__(self):

        self.verification_model = MODEL.SPEAKERNET
        self._init_torch_model()

        # cosine similarity threshold: values between (-1,1)
        self.SIMILARITY_THRESHOLD = 0.4
        self.new_speaker_audio = None



        self.load_speakers_vectors()

    def _init_torch_model(self):

        if self.verification_model == MODEL.SPEAKERNET:
            self.speakers_vectors_file = 'files/speakers_vectors_speakernet.pkl'
            self.verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path="files/speakerverification_speakernet.nemo")
        if self.verification_model == MODEL.TITANET_L:
            self.speakers_vectors_file = 'files/speakers_vectors_titanet-l.pkl'
            self.verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path="files/titanet-l.nemo")
        if self.verification_model == MODEL.ECAPA_TDNN:
            self.speakers_vectors_file = 'files/speakers_vectors_ecapa_tdnn.pkl'
            self.verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path="files/ecapa_tdnn.nemo")

        self.device = 'cpu'
        self.verification_model = self.verification_model.cpu()
        self.verification_model.eval()

        self.consumed_time = []

    def predict_speaker_vector(self, audio_file):

        return self._torch_predict_speaker_vector(audio_file)

    def _torch_predict_speaker_vector(self, audio_file):

        # default input sample rate is 16K
        audio, sr = librosa.load(audio_file, sr=16000)

        audio_length = audio.shape[0]
        audio_signal, audio_signal_len = (
            torch.tensor([audio], device=self.device),
            torch.tensor([audio_length], device=self.device),
        )

        _, embs = self.verification_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        emb_shape = embs.shape[-1]
        embs = embs.view(-1, emb_shape)

        return embs.cpu().detach().numpy()

    def load_speakers_vectors(self):
        try:
            with open(self.speakers_vectors_file, 'rb') as f:
                self.speakers_vectors = pickle.load(f)

        except FileNotFoundError:
            self.speakers_vectors = {}

    def save_speakers_vectors(self):
        with open(self.speakers_vectors_file, 'wb') as f:
            pickle.dump(self.speakers_vectors, f)

    def add_speaker_vector(self, user_id, audio_file):
        vector = self.predict_speaker_vector(audio_file)
        self.speakers_vectors[user_id] = vector
        self.save_speakers_vectors()

    def get_most_similar_speaker(self, file):

        filepath = os.path.join("test_voices", file)

        if len(self.speakers_vectors) == 0:
            print("no speaker vector")

        start = time.time()
        vector = self.predict_speaker_vector(filepath)

        speakers_vectors = [v[0] for v in self.speakers_vectors.values()]
        speakers_names = list(self.speakers_vectors.keys())

        sims = cosine_similarity([vector[0]], speakers_vectors)[0]
        end = time.time()
        print("Cosine similarity for {}: {}".format(file, sims))
        self.consumed_time.append(end-start)


        return self.consumed_time


import os
wav_files = os.listdir("./test_voices")

speaker_indentifier = NemoSpeakerIdentifier()

user_id = ["USER_1", "USER_2", "USER_3", "USER_4"]
user_ref_voices = ["./ref_voices/user1.wav", "./ref_voices/user2.wav", "./ref_voices/user3.wav", "./ref_voices/user4.wav"]

for id, voice in zip(user_id, user_ref_voices):
    speaker_indentifier.add_speaker_vector(id, voice)

for file in wav_files:
    speaker_indentifier.get_most_similar_speaker(file)

time_vector = speaker_indentifier.get_most_similar_speaker(file)
total_time = 0
for i in time_vector:
    total_time += i

print("Average consuming time: {}".format(total_time / len(time_vector)))
