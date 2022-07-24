from collections import defaultdict
import string
import readline

import dill
import pybktree
import editdistance
from nltk import sent_tokenize, wordpunct_tokenize

import config
from probabilistic_distance import probabilistic_distance
from learn import tokenize_sentence, generalize_tokens
from Viterbi import Viterbi

def rebuild(tokens, correct_tokens):
  rebuilt_tokens = correct_tokens.copy()
  for i, corr_token in enumerate(correct_tokens):
    if corr_token == "PERSON_NAME":
      rebuilt_tokens[i] = tokens[i]
  return rebuilt_tokens

class Corrector():
  def load_model(self):
    print("Loading model")
    model = dill.load(open(f"{config.MODEL}/model.dill", 'rb'))
    words = model['words']
    words_inverse = model['words_inverse']
    tree = model['tree']
    viterbi = Viterbi(words, words_inverse, tree)
    print("Ready.")
    self.viterbi = viterbi
    self.words = words
    self.words_inverse = words_inverse
    self.tree = tree

  def correct(self, text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sentence in sentences:
      corrected_sentences.append(" ".join(self.correct_sentence(sentence)))
    return corrected_sentences

  def correct_sentence(self, sentence):
    tokens = tokenize_sentence(sentence)
    generalized_tokens = generalize_tokens(tokens)
    corrected_tokens = self.viterbi.run(generalized_tokens)
    rebuilt_tokens = rebuild(tokens, corrected_tokens)
    return rebuilt_tokens

if __name__ == "__main__":
  corrector = Corrector()
  corrector.load_model()
  while True:
    text = input(">>> ")
    corrected_sentences = corrector.correct(text)
    [print(sent) for sent in corrected_sentences]
