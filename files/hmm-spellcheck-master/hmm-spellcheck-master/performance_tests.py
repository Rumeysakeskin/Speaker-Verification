from correct import Corrector
from learn import tokenize_sentence
from collections import defaultdict
import random
random.seed(a=1)

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]

def noise_maker(sentence, threshold=0.9):
  noisy_sentence = []
  i = 0
  while i < len(sentence):
    do_error = random.random()
    if do_error < threshold:
      noisy_sentence.append(sentence[i])
    else:
      error_type = random.choice([1,2,3])
      if error_type == 1:
        # inversion
        if i == (len(sentence) - 1):
          continue
        else:
          if sentence[i] != ' ' and sentence[i+1] != ' ':
            #print("INVERSION ", (sentence[i], sentence[i+1]))
            noisy_sentence.append(sentence[i+1])
            noisy_sentence.append(sentence[i])
            i += 1
          else:
            i -= 1 # riconsidero la stessa lettera nell'iterazione successiva
      elif error_type == 2:
        #aggiunta

        random_letter = random.choice(letters)
        noisy_sentence.append(random_letter)
        noisy_sentence.append(sentence[i])
        #print("AGGIUNTA ", random_letter)
      elif error_type == 3:
        # remove a letter
        if sentence[i] == ' ':
          noisy_sentence.append(sentence[i])
        else:
          prev_char = sentence[i-1] if i != 0 else ' '
          next_char = sentence[i+1] if i != len(sentence) - 1 else ' '
          if prev_char == ' ' and next_char == ' ':
            noisy_sentence.append(sentence[i])
          else:
            pass # remove letter
      else:
          noisy_sentence.append(sentence[i])
    i += 1
  noisy_sentence = "".join(noisy_sentence)
  if noisy_sentence == sentence:
    return noise_maker(sentence)
  return noisy_sentence

def list_diff(a, b):
  assert len(a) == len(b)
  cnt = 0
  for i in range(len(a)):
    if a[i] != b[i]: cnt += 1
  return cnt

FILE = "data/spelling_correction_corpus.txt"
if __name__ == "__main__":
  corrector = Corrector()
  corrector.load_model()

  total_lines = 0
  ok_lines = 0

  total_tokens = 0
  same_tokens = 0

  totale_token_non_perturbati_non_modificati = 0
  totale_token_non_perturbati_sballati = 0
  totale_token_corretti = 0
  totale_token_sballati = 0
  totale_token_ok = 0
  totale_token_not_ok = 0
  totale_token_noised = 0
  with open(FILE, "r") as test_file:
    for n_line, correct_line in enumerate(test_file):

      correct_tokens = tokenize_sentence(correct_line)
      if len(correct_tokens) < 3: continue
      correct_line = " ".join(correct_tokens)

      noised_sentence = noise_maker(correct_line, 0.9)
      #noised_sentence = "".join(noised_sentence_list)
      #print("\t\t", noised_sentence)
      noised_tokens = tokenize_sentence(noised_sentence)
      if len(noised_tokens) != len(correct_tokens): continue

      corrected_tokens = corrector.viterbi.run(noised_tokens)
      corrected_line = " ".join(corrected_tokens)

      token_non_perturbati_non_modificati = 0
      token_non_perturbati_sballati = 0
      token_corretti = 0
      token_sballati = 0
      token_noised = 0
      for i in range(len(correct_tokens)):
        if noised_tokens[i] != correct_tokens[i]:
          token_noised += 1
        if correct_tokens[i] == noised_tokens[i]:
          if corrected_tokens[i] == correct_tokens[i]:
            token_non_perturbati_non_modificati += 1
          else:
            token_non_perturbati_sballati += 1
          pass
        elif corrected_tokens[i] == correct_tokens[i] and noised_tokens[i] != correct_tokens[i]:
          token_corretti += 1
          pass
        elif correct_tokens[i] != corrected_tokens[i] and noised_tokens[i] != correct_tokens[i]:
          token_sballati += 1
          pass
        else:
          print(correct_tokens[i])
          print(corrected_tokens[i])
          print(noised_tokens[i])
          assert False

      totale_token_corretti += token_corretti
      totale_token_sballati += token_sballati
      totale_token_non_perturbati_non_modificati += token_non_perturbati_non_modificati
      totale_token_non_perturbati_sballati += token_non_perturbati_sballati
      totale_token_noised += token_noised
      token_ok = token_corretti + token_non_perturbati_non_modificati
      token_not_ok = token_sballati + token_non_perturbati_sballati

      if corrected_line == correct_line:
        print("OK")
        print("\tNOISED:", " ".join(noised_tokens), "\n\tCORRECTED:", correct_line)
        ok_lines += 1
      else:
        print("NO")
        print("\tNOISED:", " ".join(noised_tokens), "\n\tATTEMPT:", corrected_line,"\n\tCORRECT:", correct_line)
        print(f"\t{token_corretti} parole correttamente corrette")
        print(f"\t{token_sballati} parole modificate sbagliate")
        print(f"\t{token_non_perturbati_non_modificati} parole non perturbate non modificate")
        print(f"\t{token_non_perturbati_sballati} parole non perturbate ma sbagliate")
        print(f"\t{token_ok} totale parole azzeccate")
        print(f"\t{token_not_ok} totale parole sballate")

      total_tokens += len(corrected_tokens)
      totale_token_ok += token_ok
      totale_token_not_ok += token_not_ok

      total_lines += 1

      print("Accuracy: ", ok_lines/total_lines)
      print("Token ok:", (totale_token_ok)/total_tokens)
      print("Token not ok:", (totale_token_not_ok)/total_tokens)
      print("Token noised:", totale_token_noised/total_tokens)

  print("Sentence accuracy: ")
  print(total_lines)
  print(ok_lines)
  print(ok_lines/total_lines)
  print()
  print("Token accuracy:")
  print(total_tokens)
  print(same_tokens)
  print(same_tokens/total_tokens)
  print()
  print("Other:")
  print("Totale token:", total_tokens)
  print("Corretti:", totale_token_corretti)
  print("Sballati:", totale_token_sballati)
  print("NonPerturbatiNonModificati:", totale_token_non_perturbati_non_modificati)
  print("NonPerturbatiSballati:", totale_token_non_perturbati_sballati)
  print("TokenAzzeccati:", totale_token_ok)
  print("TokenSballati:", totale_token_not_ok)
