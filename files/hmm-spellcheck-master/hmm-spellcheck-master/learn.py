from collections import defaultdict
from itertools import repeat
from itertools import islice
import subprocess
import string
import re
import os

from nltk import sent_tokenize, wordpunct_tokenize
import dill
import numpy as np
import tqdm

import config

pattern = re.compile(r'[\W_]+')
punct = set(string.punctuation)
def clean_token(token):
  if token in punct: return None
  token = pattern.sub('', token).strip()
  return token or None

names = set([line.strip() for line in open("data/names.txt")])
def replace_names(token, keep_both=False):
  if token in names:
    if keep_both:
      return f"PERSON_NAME|{token}"
    else:
      return "PERSON_NAME"
  else:
    return token

def tokenize_sentence(sentence, end_sentence=False):
  sentence = sentence.lower()
  sentence = tag_regex.sub('', sentence)
  tokens = wordpunct_tokenize(sentence)
  tokens = map(clean_token, tokens)
  tokens = filter(None, tokens)
  tokens = list(tokens)
  if end_sentence:
    tokens.append("END_SENTENCE")
  return tokens

def generalize_tokens(tokens, keep_both=False):
  tokens = list(map(lambda x: replace_names(x, keep_both=keep_both), tokens))
  return tokens

words = defaultdict(int)
url_regex = re.compile(r'http\S+')
tag_regex = re.compile(r'@\S+')
skip_lines = ["<text", "</text", "## ", "# ", "warning:", "facezie", "pubblicato da", "temi:", "pubblicato", "postato ", "from:", "by", "__", "directory", "appunti:"]
skip_sents = ["http", "www."]
def parse(line):
  line = url_regex.sub('', line)
  lower_line = line.lower()
  if any([lower_line.startswith(skip_form) for skip_form in skip_lines]): return
  sentences = sent_tokenize(line)
  for sentence in sentences:
      sent_lower = sentence.lower()
      if any([skip in sent_lower for skip in skip_sents]): continue
      tokens = tokenize_sentence(sentence, end_sentence=True)
      if len(tokens) < 3: continue
      tokens = generalize_tokens(tokens, keep_both=True)
      for i, token in enumerate(tokens):
        if i == 0:
          if token.startswith("PERSON_NAME"):
            tokens = token.split("|")
            words[("START_SENTENCE", tokens[0])] += 1
            words[("START_SENTENCE", tokens[1])] += 1
          else:
            words[("START_SENTENCE", token)] += 1
        else:
          if token.startswith("PERSON_NAME"):
            tokens = token.split("|")
            words[(prev_token, tokens[0])] += 1
            words[(prev_token, tokens[1])] += 1
            token = tokens[0]
          else:
            words[(prev_token, token)] += 1
        prev_token = token

MAX_LINES = 100_000
if __name__ == "__main__":
  for FILE in os.listdir(config.MODEL):
    if not FILE.endswith('.txt'): continue
    with open(f"{config.MODEL}/{FILE}") as data_file:
      n_lines = min(MAX_LINES, int(subprocess.check_output(f'wc -l "{config.MODEL}/{FILE}"', shell=True).split()[0]))
      for line in tqdm.tqdm(islice(data_file, n_lines), total=n_lines):
        parse(line)

  words_split = defaultdict(lambda: defaultdict(int))
  for (prev_token, token), value in words.items():
      words_split[prev_token][token] = value
  words = words_split

  dill.dump(words, open(f"{config.MODEL}/words.dill", 'wb'))
