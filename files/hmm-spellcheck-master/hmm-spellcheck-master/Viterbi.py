from probabilistic_distance import probabilistic_distance
from collections import defaultdict
from pprint import pprint
from math import log
import numpy as np
from itertools import chain

def bktree_to_set(results):
  return set(map(lambda x: x[1], results))

def get_max_inverse(state, previous_states, model, model_inverse):
  try:
    #max_prob = next(iter(model_inverse[state].values()))
    # return the max inverse probability (first since it is ordered dict)
    max_prob = max([model[prev].get(state, 1e-15) for prev in previous_states])
    #a = sorted([(prev, model[prev].get(state, 1e-15)) for prev in previous_states], reverse=True, key=lambda x: x[1])
    #print(a[0])
    # la probabilità più alta che tra le possibili parole precedenti si vada nella parola corrente
  except:
    max_prob = 1e-15
  return max_prob

def filter_possible_states(observation, possible_states, previous_states=None, model_inverse=None, model=None, AMOUNT=20, ADVANCE_FILTERING=True):
  if model_inverse is None:
    weighted_states = [(state, probabilistic_distance(observation, state)) for state in possible_states]
  else:
    #[model_inverse[state] for state in possible_states]
    weighted_states = []
    for state in possible_states:
      # estrai solo i previous_state di state
      """max_prob = 1e-15
      for predecessor in model_inverse[state].keys():
        if predecessor in previous_states:
          max_prob = model_inverse[state][predecessor]
          break"""
      if ADVANCE_FILTERING:
        multiplier = get_max_inverse(state, previous_states, model, model_inverse)
      else:
        multiplier = 1
      weighted_states.append(
        (
          state,
          probabilistic_distance(observation, state) * multiplier
        )
      )
  weighted_states = sorted(weighted_states, key=lambda p: p[1], reverse=True)
  possible_states = [state for (state, distance) in weighted_states]
  return possible_states[:AMOUNT]

class Viterbi():
    def __init__(self, model,model_inverse, bktree):
      self.model = model
      self.model_inverse = model_inverse
      self.bktree = bktree

    def run(self, observations, SEARCH_DEPTH=2, ADVANCE_FILTERING=True, AMOUNT=20):
      if not len(observations): return []

      T1 = defaultdict(lambda: defaultdict(float))
      T2 = defaultdict(lambda: defaultdict(str))

      starting_states = bktree_to_set(self.bktree.find(observations[0], 3))
      if not len(starting_states):
        starting_states = [observations[0]]

      for state in starting_states:
        # self.model['START_SENTENCE'].get(state, 1e-15) *
        T1[0][state] = self.model['START_SENTENCE'].get(state, 1e-15) * probabilistic_distance(state, observations[0])
        T2[0][state] = ''

      states = defaultdict(set)
      states[0] = filter_possible_states(observations[0], starting_states, ADVANCE_FILTERING=ADVANCE_FILTERING, AMOUNT=AMOUNT)
      if not len(states[0]):
        states[0] = [observations[0]]
      #print(len(states[0]))
      #print(states[0])
      #print("\n")
      for j, observation in enumerate(observations):
        if j == 0:
          continue

        similar_states = []
        curr_depth = SEARCH_DEPTH
        while not similar_states:
          similar_states =  bktree_to_set(self.bktree.find(observation, curr_depth))
          curr_depth += 1
        possible_successor_states_and_probs = set(chain.from_iterable([ list(self.model[state].items()) for state in states[j-1] ]))
        possible_successor_states = [pair[0] for pair in sorted(possible_successor_states_and_probs, key=lambda x: x[1], reverse=True)][:100]
        states[j] = similar_states | set(possible_successor_states)
        states[j] = filter_possible_states(observation, states[j], states[j-1], model_inverse=self.model_inverse, model=self.model, ADVANCE_FILTERING=ADVANCE_FILTERING, AMOUNT=AMOUNT)
        if not len(states[j]):
          states[j] = [observation]
        #print(states[j])
        for state in states[j]:
          prev_states = states[j-1]
          probs = [T1[j-1][prev_state] * self.model[prev_state].get(state, 1e-15) * probabilistic_distance(state, observation) for prev_state in prev_states]
          T1[j][state] = max(probs)

          probs2 = [T1[j-1][prev_state] * self.model[prev_state].get(state, 1e-15) for prev_state in prev_states]
          T2[j][state] = prev_states[np.argmax(probs2)]

      # delete last level, not needed
      if len(observations) in states: del states[len(observations)]

      # reconstruct
      T = len(observations) - 1
      X = ['' for i in range(len(observations))]
      X[T] = states[T][np.argmax([T1[T][state] for state in states[T]])]
      for j in range(T, 0, -1):
        X[j-1] = T2[j][X[j]]
      #pprint(T2)
      return X
