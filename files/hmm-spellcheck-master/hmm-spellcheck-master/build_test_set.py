from collections import defaultdict
from itertools import repeat
from itertools import islice
import subprocess
import string
import re
import os
import config
import collections
from nltk import sent_tokenize
import random
from learn import skip_lines, skip_sents
from learn import MAX_LINES
def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

if __name__ == "__main__":
  lines = 0
  for FILE in os.listdir(config.MODEL):
    if not FILE.endswith('.txt'): continue
    with open(f"{config.MODEL}/{FILE}") as data_file:
      consume(data_file, MAX_LINES)
      for line in data_file:
        lower_line = line.lower()
        if any([line.startswith(form) for form in skip_lines]): continue
        if random.random() > 0.8:
          sentences = sent_tokenize(line)
          sentence = sentences[0]
          sent_lower = sentence.lower()
          if any([skip in sent_lower for skip in skip_sents]): continue
          if len(sentence) > 30:
            print(sentence)
            lines+=1
            if lines > 1_000:
              quit()
