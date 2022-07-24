from correct import Corrector
from learn import tokenize_sentence, names
from flask import Flask, render_template, request, jsonify, current_app
from nltk import sent_tokenize
from probabilistic_distance import probabilistic_distance
from collections import defaultdict
from test import noise_maker

app = Flask(__name__, static_folder="web_interface", template_folder="web_interface")

def bktree_to_set(results):
  return set(map(lambda x: x[1], results))

corrector = Corrector()
corrector.load_model()

@app.route("/")
def hello():
  return render_template('index.html')

@app.route("/api/successors")
def successors():
  RETURN_AMOUNT = 10
  text = request.args.get("text","")

  has_space = False
  if text.endswith(" "): has_space = True
  #text = "ciao a tutti. come va?"
  sentences = sent_tokenize(text)
  if len(sentences):
    last_sentence = sentences[-1]
    tokens = tokenize_sentence(last_sentence)
  else:
    tokens = []

  if len(tokens) == 0:
    return jsonify(
      sorted(corrector.words["START_SENTENCE"].keys(), key=lambda x: corrector.words["START_SENTENCE"].get(x, 1e-15), reverse=True)[:RETURN_AMOUNT]
    )

  if len(tokens) == 1 and not has_space:
    input_word = tokens[0]
    similar_words = bktree_to_set(corrector.tree.find(input_word, 3))
    weighted_similar_words = [
      (word, probabilistic_distance(input_word, word) * corrector.words["START_SENTENCE"].get(word, 1e-15)) for word in similar_words
    ]
    weighted_similar_words = sorted(weighted_similar_words, key=lambda x: x[1], reverse=True)[:RETURN_AMOUNT]
    return jsonify(
      [x[0] for x in weighted_similar_words]
    )

  if len(tokens) > 1 and not has_space:
    token = tokens[-1]
    prev_token = tokens[-2]
    similar_words = bktree_to_set(corrector.tree.find(token, 3))
    weighted_similar_words = [
      (word, probabilistic_distance(token, word) * corrector.words[prev_token].get(word, 1e-15)) for word in similar_words
    ]
    weighted_similar_words = sorted(weighted_similar_words, key=lambda x: x[1], reverse=True)[:RETURN_AMOUNT]
    return jsonify(
      [x[0] for x in weighted_similar_words]
    )

  if len(tokens) >= 1 and has_space:
    # prendi l'ultimo token e fai vedere quelli più probabili successivi
    last_token = tokens[-1]
    if last_token in names:
      last_token = "PERSON_NAME"

    return jsonify(
      sorted(corrector.words[last_token].keys(), key=lambda x: corrector.words[last_token].get(x, 1e-15), reverse=True)[:RETURN_AMOUNT]
    )

@app.route("/api/viterbi")
def viterbi():
  text = request.args.get("text","")
  #text = "ciao a tutti. come va?"
  sentences = sent_tokenize(text)
  if not len(sentences):
    return ""

  last_sentence = sentences[-1]
  return corrector.correct(last_sentence)[0]

@app.route("/api/noise")
def noise():
  text = request.args.get("text","")
  return noise_maker(text, 0.9)

app.run(debug=True)
