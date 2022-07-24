from collections import defaultdict

import dill
import pybktree
import editdistance

import config

print("Loading words...")
words = dill.load(open(f"{config.MODEL}/words.dill", 'rb'))

# remove words that only appear one time
i = 0
keys = list(words.keys())
for word in keys:
  if sum(words[word].values()) <= 2:
    i += 1
    del words[word]



words_inverse = defaultdict(lambda: defaultdict(float))

print("Building inverse words...")
# build inverse lookup

for predecessor in words.keys():
  for successor in words[predecessor].keys():
    words_inverse[successor][predecessor] += words[predecessor][successor]

for successor in words_inverse.values():
  predecessors = successor.keys()
  occurrences = successor.values()

  prob_factor = 1/sum(occurrences)
  for pred in predecessors:
    successor[pred] *= prob_factor
# sort inverse lookup
for successor in words_inverse.keys():
  pred_and_probs = words_inverse[successor].items()
  pred_and_probs = sorted(pred_and_probs, key=lambda x: x[1], reverse=True)
  words_inverse[successor] = dict()
  for (pred, probability) in pred_and_probs:
    words_inverse[successor][pred] = probability

print("Normalizing word frequencies...")
for word in words.values():
  successors = word.keys()
  occurrences = word.values()
  prob_factor = 1/sum(occurrences)
  for successor in successors:
    word[successor] *= prob_factor


print("Building BKTree...")
tree = pybktree.BKTree(editdistance.eval)
[tree.add(word) for word in words.keys()]

print("Dumping to file...")
model = dict()
model['words'] = words
model['words_inverse'] = words_inverse
model['tree'] = tree

dill.dump(model, open(f"{config.MODEL}/model.dill", 'wb'))
