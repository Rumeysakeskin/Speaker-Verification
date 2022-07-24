# Importing all required libraries for this task.
import nltk
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
from itertools import chain
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import *
from nltk.corpus import wordnet as wn
import time
from tqdm import tqdm
from nltk.metrics.distance import edit_distance
from difflib import SequenceMatcher
lemmatizer = WordNetLemmatizer()


def parsing(sent):
    """Parsing the sentence to corrected and original and storing in the dictionary."""
    loriginal = []
    lcorrected = []
    lcorr = []
    indexes = []
    cnt = 0

    for i in sent:
        if '|' in i:
            # Splitting the sentence on '|'
            str1 = i.split('|')
            # Previous word to '|' is storing in loriginal list.
            loriginal.append(str1[0])
            # Next word to '|' is storing in lcorrected list.
            lcorrected.append(str1[1])
            # Noting down the index of error.
            indexes.append(cnt)

        else:
            # If there is no '|' in sentence, sentence is stored in loriginal and lcorrected as it is.
            loriginal.append(i)
            lcorrected.append(i)
        cnt = cnt + 1

    # Loading to loriginal, lcorrected and index list to dictionary.
    dictionary = {'original': loriginal, 'corrected': lcorrected, 'indexes': indexes}

    return dictionary


def preprocessing():
    """Loading the data from 'holbrook.txt' and passing to parsing function to get parssed sentences.
    Returning the whole dictionary as data."""
    data = []

    # Reading the txt file
    text_file = open("spelling_correction_corpus.txt", "r", encoding="utf-8")
    lines = []
    for i in text_file:
        lines.append(i.strip())

    # Word tokenizing the sentences
    sentences = [nltk.word_tokenize(sent) for sent in lines]

    # Calling a parse function to get corrected, original sentences.
    for sent in sentences:
        data.append(parsing(sent))

    return data


# Calling preprocessing function
data = preprocessing()

# Testing
# Splitting the data to test 100 lines and remaining training lines
test = data[:100]
train = data[100:]


# Splitting the data to test - first 100 lines and remaining training lines
def test_train_split():
    """Splitting the data to test - first 100 lines and remaining training lines."""
    test = data[:100]
    train = data[100:]

    # Seperating the train original, test original, test corrected and train corrected from dictionary to list.
    train_corrected = [elem['corrected'] for elem in train]
    tokenizer = RegexpTokenizer(r'\w+')
    test_corrected = [elem['corrected'] for elem in test]
    test_original = [elem['original'] for elem in test]

    # Removing all special characters from the list.
    test_original = [tokenizer.tokenize(" ".join(elem)) for elem in test_original]
    test_corrected = [tokenizer.tokenize(" ".join(elem)) for elem in test_corrected]
    train_corrected = [tokenizer.tokenize(" ".join(elem)) for elem in train_corrected]

    return test_corrected, test_original, train_corrected


# Test and Training data.
test_corrected, test_original, train_corrected = test_train_split()


# def unigram(words):
#     """This function returns a unigram frequency for a given word."""
#     doc = []
#     words = words.lower()
#     for i in train_corrected:
#         doc.append(" ".join(i).lower())
#
#     doc = " ".join(doc)
#     doc = nltk.word_tokenize(doc)
#
#     # Calculates frequency distribution.
#     unig_freq = nltk.FreqDist(doc)
#
#     # This gives word count - which is not used (for future modification)
#     tnum_unig = sum(unig_freq.values())
#
#     return unig_freq[words], tnum_unig


def bigram(words):
    """This function returns a bigram frequency for a given words."""
    doc = []

    # This function get words in string, hence converting string of 2 words to tuple.
    words = words.split(" ")
    words[0] = words[0].lower()
    words[1] = words[1].lower()
    words = tuple(words)

    for i in train_corrected:
        doc.append(" ".join(i))

    doc = " ".join(doc)
    doc = doc.lower()

    # Calculating Bigrams for given words.
    tokens = nltk.wordpunct_tokenize(doc)
    bigrams = nltk.collocations.BigramCollocationFinder.from_words(tokens)
    biag_freq = dict(bigrams.ngram_fd)

    # This gives totla bigram count - which is not used (for future modification)
    tnum_bg = sum(biag_freq.values())

    # If there is no such bigram return 0
    try:
        return biag_freq[words], tnum_bg
    except KeyError:
        return 0, 0


# Note: I cannot execute below assert code, my function unigram and bigram returns two values for my future work.
# However, count is same and printed when executed



# Edit distance returns the number of changes to transform one word to another

def get_candidates(token):
    """Get nearest word for a given incorrect word."""
    doc = []

    for i in train_corrected:
        doc.append(" ".join(i))

    doc = " ".join(doc)
    doc = nltk.word_tokenize(doc)
    unig_freq = nltk.FreqDist(doc)
    unique_words = list(unig_freq.keys())

    # Calculate distance between two words
    s = []
    for i in unique_words:
        t = edit_distance(i, token)
        s.append(t)

    # Store the neares words in ordered dictionary
    dist = dict(zip(unique_words, s))
    dist_sorted = dict(sorted(dist.items(), key=lambda x: x[1]))
    minimal_dist = list(dist_sorted.values())[0]

    keys_min = list(filter(lambda k: dist_sorted[k] == minimal_dist, dist_sorted.keys()))

    return keys_min

# This is to culculate unigram and bigram probabilities in correct function
doc = []

for i in train_corrected:
    doc.append(" ".join(i).lower())

doc = " ".join(doc)
doc = nltk.word_tokenize(doc)
unig_freq = nltk.FreqDist(doc)
unique_words = list(unig_freq.keys())

cf_biag = nltk.ConditionalFreqDist(nltk.bigrams(doc))
cf_biag = nltk.ConditionalProbDist(cf_biag, nltk.MLEProbDist)


def correct(sentence):
    "This function returns the corrected sentence based on bigram probability."
    corrected = []
    cnt = 0
    indexes = []

    for i in sentence:
        if len(i.split()) == 1:
            continue
        # If word not in unique word the calculate suggested words with minimal distance
        if i.lower() not in unique_words:
            indexes.append(cnt)
            if len(get_candidates(i)) > 1:

                suggestion = get_candidates(i)
                prob = []

                # For each suggested word calculate bigram probability
                for sug in suggestion:

                    # Check the misspelled word is first or last word of the sentence
                    if ((cnt != 0) and (cnt != len(sentence) - 1)):

                        p1 = cf_biag[sug.lower()].prob(sentence[cnt + 1].lower())
                        p2 = cf_biag[corrected[len(corrected) - 1].lower()].prob(sug.lower())
                        p = p1 * p2
                        prob.append(p)


                    else:
                        # If mispelled word is last word of a sencence take probaility of previous word
                        # if cnt == len(sentence) and len(sentence) != 1:
                        if cnt == len(sentence) - 1:
                            try:

                                p2 = cf_biag[corrected[len(corrected) - 1].lower()].prob(sug.lower())
                                prob.append(p2)
                            except:
                                pass

                        # If mispelled word is first word of a sencence take probaility of next word
                        # elif cnt == 0 and len(sentence) != 1:
                        elif cnt == 0:
                            p1 = cf_biag[sug.lower()].prob(sentence[cnt + 1].lower())
                            prob.append(p1)

                        # elif cnt == 0 and len(sentence) == 1:
                        #     p3 = cf_biag[corrected[len(corrected)].lower()]
                        #     prob.append(p3)


                # Take the suggested word with maximum priobability.
                if len(suggestion[prob.index(max(prob))]) > 1:
                    corrected.append(suggestion[prob.index(max(prob))])
                else:
                    corrected.append(suggestion[prob.index(max(prob))])
            # If only 1 suggested word take that word - no need to calculate probabilities
            else:
                corrected.append(get_candidates(i)[0])

        else:
            corrected.append(i)
        # Return the corrected sentence
        cnt = cnt + 1
    return corrected



start_time = time.time()


def accuracy(actual_sent, sent_pred):
    """This is based on word to word accuracy calculation. Compares each word with the actual word and calculate accuracy"""
    actual = actual_sent
    predict = correct(sent_pred)
    # If the blank sentence i.e for a blank line predicted is also blank take accuracy as 1
    if len(actual) == 0 and len(predict) == 0:
        accuracy = 1.0
    else:
        # Take all predicted words same as actual word and divide by lenght of sentence
        accuracy = len(set(predict) & set(actual)) / len(set(actual))


    return accuracy


acc = []
for i in tqdm(range(len(test_corrected))):
    acc.append(accuracy(test_corrected[i], test_original[i]))


print(accuracy(test_corrected[i], test_original[i]))

print(
    "************************************************EVALUATION**********************************************************")
print("Average Accuracy of words in each sentence: ", round(sum(acc) / len(acc) * 100, 4), "%")
print(acc.count(1), "out of 100 sentences predicted correctly without any error.")
elapsed_time = time.time() - start_time
print("Elapsed Time is: ", elapsed_time)


def tense(suggestion, sentence):
    """Tense Detection"""
    tag = dict(nltk.pos_tag(sentence)).values()
    past_tense = ['VBN', 'VBD']
    conti_tense = ['VBG']

    # If sentence is past tense append ed and check if it is valid word
    if any(x in tag for x in past_tense):
        sug = []
        for a in suggestion:
            if a.lower() + 'ed' in unique_words:
                sug.append(a + 'ed')
        for aelem in sug:
            suggestion.append(aelem)

    # If sentence is past tense append ed and check if it is valid word
    if any(x in tag for x in conti_tense):
        sug = []
        for b in suggestion:
            if b.lower() + 'ing' in unique_words:
                sug.append(b + 'ing')
        for belem in sug:
            suggestion.append(belem)

    return suggestion


def named_entity(sentence):
    """Named Entity Detection using nltk.pos_tag and nltk.ne_chunk"""
    l = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(sentence)):
        # If any named tag like PERSON, ORGANIZATION or GEOLOCATION append to list.
        if hasattr(chunk, 'label'):
            l.append(' '.join(c[0] for c in chunk))

    if len(l) != 0:
        l = " ".join(l)
        l = l.split(" ")

    return l


# print(named_entity(['I', 'live', 'at', 'Boar', 'Parva', 'it', 'is', 'near', 'Melton', 'and', 'Bridgebrook', 'and', 'Smallerden']))


def word_forms_new(suggest):
    """Taking different forms of words using derivationally related forms"""
    sug_form = []
    for w in suggest:
        forms = set()
        for i in wn.lemmas(w):
            forms.add(i.name())
            for j in i.derivationally_related_forms():
                forms.add(j.name())

        for a in list(forms):
            sug_form.append(a)

    for q in sug_form:
        suggest.append(q)

    word_forms = []
    [word_forms.append(i) for i in suggest if not i in word_forms]
    return word_forms


def conditions(corrected, cr_ind):
    """Common word - Oclock is not detecting. Hence handling manually but not necessary"""

    if 'oclock' in corrected:
        ind = corrected.index('oclock')
        corrected = list(map(lambda x: str.replace(x, "oclock", "clock"), corrected))
        corrected.insert(ind, 'o')
        return corrected

    return corrected


# word_forms_new(['wait', 'said', 'laid', 'paid', 'wad', 'waited'])

def sentence_sentence_similarity(sentence1):
    """Sentence - Sentence similarity using sequence matcher. We can also use cosine similarity but not implemented"""
    correc = []
    for d in train_corrected:
        ratio = SequenceMatcher(None, " ".join(d), " ".join(sentence1)).ratio()
        if ratio > 98:
            correc.append(d)

    if len(correc) == 1:
        return correc[0]
    else:
        return []



def correct_mod(sentence):
    sts = ['oclock']
    corrected = []
    cnt = 0
    indexes = []
    # To check stemmed word in dictonary or not
    stemmer = PorterStemmer()
    status = 0
    # This will extract all named entities of a sentence
    n_en = named_entity(sentence)

    for i in sentence:
        # Check for sentence similarity
        corr = sentence_sentence_similarity(i)
        if len(corr) == 1:
            return corr
        # Ignoring digits like page number and lemmatizing the word and check if it is present in dictionary and use words.words() for word validation.
        elif i.lower() not in unique_words and not i.isdigit() and lemmatizer.lemmatize(
                i.lower()) not in unique_words and i not in n_en and i not in sts and i not in wn.words() and stemmer.stem(
                i) not in wn.words():
            indexes.append(cnt)
            if len(get_candidates(i)) > 1:
                # Get words forms, tense detection for suggested sentence
                suggestion = get_candidates(i)
                suggestion = tense(suggestion, sentence)
                wd_fms = word_forms_new(suggestion)
                suggestion = wd_fms

                prob = []

                # Bigram probabilities
                for sug in suggestion:

                    # Check the misspelled word is first or last word of the sentence
                    if ((cnt != 0) and (cnt != len(sentence) - 1)):

                        try:
                            p1 = cf_biag[sug.lower()].prob(sentence[cnt + 1].lower())
                            p2 = cf_biag[corrected[len(corrected) - 1].lower()].prob(sug.lower())
                            p = p1 * p2
                            prob.append(p)
                        except:
                            prob.append(0)

                    else:
                        # If mispelled word is last word of a sencence take probaility of previous word
                        if cnt == len(sentence) - 1:
                            try:
                                p2 = cf_biag[corrected[len(corrected) - 1].lower()].prob(sug.lower())
                                prob.append(p2)
                            except:
                                prob.append(0)


                        elif cnt == 0:
                            # If mispelled word is first word of a sencence take probaility of next word
                            try:
                                p1 = cf_biag[sug.lower()].prob(sentence[cnt + 1].lower())
                                prob.append(p1)
                            except:
                                prob.append(0)

                if len(suggestion[prob.index(max(prob))]) > 1:
                    corrected.append(suggestion[prob.index(max(prob))])
                else:
                    corrected.append(suggestion[prob.index(max(prob))])

            else:
                corrected.append(get_candidates(i)[0])

        else:
            corrected.append(i)

        cnt = cnt + 1
        # Manula hadling 'Oclock'
        corrected = conditions(corrected, indexes)

    fin = sentence_sentence_similarity(corrected)
    if len(fin) != 0:
        return fin
    else:
        return corrected


start_time = time.time()


def accuracy(actual_sent, sent_pred):
    """This is based on word to word accuracy calculation. Compares each word with the actual word and calculate accuracy"""
    actual = actual_sent
    predict = correct_mod(sent_pred)
    # If the blank sentence i.e for a blank line predicted is also blank take accuracy as 1
    if len(actual) == 0 and len(predict) == 0:
        accuracy = 1.0
    else:
        # Take all predicted words same as actual word and divide by lenght of sentence
        accuracy = len(set(predict) & set(actual)) / len(set(actual))

    print("Actual Sentence: ", actual)
    print("Sentence to predict: ", sent_pred)
    print("Predicted Sentence: ", predict)
    print("Accuracy: ", accuracy)

    return accuracy


acc = []
for i in tqdm(range(len(test_corrected))):
    acc.append(accuracy(test_corrected[i], test_original[i]))

test = ['konuşma', 'kızını', 'artar']
test_corrected = ['konuşma', 'hızını', 'artır']
print(accuracy(test_corrected, test))



print(
    "************************************************EVALUATION**********************************************************")
print("Average Accuracy of words in each sentence: ", round(sum(acc) / len(acc) * 100, 4), "%")
print(acc.count(1), "out of 100 sentences predicted correctly without any error")
elapsed_time = time.time() - start_time
print("Elapsed Time is: ", elapsed_time)