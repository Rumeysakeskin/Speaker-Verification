import numpy as np
import random

class NGramModel:
    def __init__(self, dictionary, rawWordCounts, isSmoothed):
        self.isSmoothed = isSmoothed
        self.dictionary = dictionary
        self.dictionaryLength = len(dictionary);
        self.rawWordCounts = rawWordCounts;
    def getDictionary(self):
        return self.dictionary
    def getRawWordCounts(self):
        return self.rawWordCounts
    def toString(self):
        print("\n--------------toString()-start-------------")
        print("Dictionary length is : ", len(self.dictionary))
        print("nGram Model is ", self.__class__.__name__)
        print("--------------toString()-end---------------\n")
        
class UniGramModel(NGramModel):
    def __init__(self, dictionary, rawWordCounts, isSmoothed):
        super(UniGramModel,self).__init__(dictionary, rawWordCounts, isSmoothed);
        self.countTable = self.rawWordCounts;
    def calculateProbabilityTable(self):
        N = self.dictionaryLength;
        # TODO : isSmoothed
        self.probabilyTable = (self.countTable+1)/(self.countTable.sum()+ N);
        return self.probabilyTable;
    def getLogProbabilityOfSentence(self, sentence):
        tokens = sentence.split(" ");
        logProbability = 0.0;
        logProbabilityTable = np.log2(self.probabilyTable);
        for token in tokens:
            if token in self.dictionary:
                logProbability = logProbability + logProbabilityTable[self.dictionary.index(token)]
        return logProbability;
    def generateEmail(self):
        sentence = "<s>";
        
        for i in range(30):
            dictionaryIndex = self.getRandom();
            sentence = sentence + " " + self.dictionary[dictionaryIndex ];
            if self.dictionary[dictionaryIndex] == "</s>":
                break;
        return sentence;
                      
    def getRandom(self):
        randomFlaot =  random.uniform(0.0, 1.0);
        probabilityCounter = 0.0;
        for i in range(self.dictionaryLength):
            probabilityCounter  = probabilityCounter   + self.probabilyTable[i];
            if probabilityCounter > randomFlaot :
                return i;
        return -1;
class BiGramModel(NGramModel):
    def __init__(self, dictionary, rawWordCounts, isSmoothed):
        super(BiGramModel,self).__init__(dictionary, rawWordCounts, isSmoothed)
        #self.probabilities = self._BiGramModel__calculateProbabilities()
    def generateEmail(self, uniDictionary):
        sentence = "<s>";        
        prefix = sentence;
        for i in range(30):
            dictionaryIndex = self.getRandom(len(uniDictionary), uniDictionary.index(prefix));
            if dictionaryIndex != -1:
                sentence = sentence + " " + uniDictionary[dictionaryIndex];
                prefix = uniDictionary[dictionaryIndex];
                if uniDictionary[dictionaryIndex] == "</s>":
                    break;
        return sentence;
    def getRandom(self, uniDictionaryLength, prefixIndex):
        randomFlaot =  random.uniform(0.0, 1.0);
        temp = self.probabilityTable.sum(axis=1);
        tempCoefficient = 1/temp[prefixIndex];
        probabilityCounter = 0.0;
        for i in range(uniDictionaryLength):
            probabilityCounter  = probabilityCounter   + (self.probabilityTable[prefixIndex][i] * tempCoefficient);
            if probabilityCounter > randomFlaot :
                return i;
        return -1;
    def getLogProbabilityOfSentence(self, sentence, uniGramDictionary, prefix):
        tokens = sentence.split(" ");
        logProbability = 0.0;
        logProbabilityTable = np.log2(self.probabilityTable);
        for token in tokens:
            if (token in uniGramDictionary) and (token in uniGramDictionary):
                logProbability = logProbability + logProbabilityTable[uniGramDictionary.index(prefix) ,uniGramDictionary.index(token)]
                prefix = token;
        return logProbability;
    def calculateProbabilityTable(self, uniDictionary, uniCountTable):
        dictionaryLength = len(uniDictionary);
        biGramWordCounts = np.zeros((dictionaryLength, dictionaryLength)).astype(np.float64);
        biGramWordProbabilities = np.zeros((dictionaryLength, dictionaryLength)).astype(np.float64);     
#        print("Calculating biGram Probabilities...");
        
        for i in range(self.dictionaryLength):
            tokens = self.dictionary[i].split(" ");
            isTokensTrue = True;
            isTokensTrue =  [(isTokensTrue and (token in uniDictionary)) for token in tokens];
            if isTokensTrue :
                index1 = uniDictionary.index(tokens[0]);
                index2 = uniDictionary.index(tokens[1]);
                biGramWordCounts[index1, index2] = self.rawWordCounts[i];
        if self.isSmoothed:
            temp = uniCountTable+dictionaryLength;
            biGramWordProbabilities = (biGramWordCounts+1) / temp[:,None];
        else:
            biGramWordProbabilities = (biGramWordCounts) / uniCountTable[:,None];
            
        self.countTable = biGramWordCounts;
        self.probabilityTable = biGramWordProbabilities;
        return self.probabilityTable;

class CorpusModel:
    def __init__(self, datasetFileName):
        self.datasetFileName = datasetFileName
        with open(self.datasetFileName, encoding='utf-8') as file:
            lines = file.readlines()
        self.corpus = ["<s> " + line.strip() + " </s>" for line in lines]

    #def getUnigramCountsAndDictionary():
    def getDataBetween(self, startingIndex, endingIndex):
        return self.corpus[startingIndex:endingIndex];
    def getCorpus(self):
        return self.corpus
    
    def getTrainData(self):
        maxIndex = 6*len(self.corpus)/10
        return self.corpus[0:int(round(maxIndex))];
    def getTestData(self):
        maxIndex = 6*len(self.corpus)/10
        return self.corpus[int(round(maxIndex)):];
        
def tokenize(s):
    illegalTokens = ['!', '"', '$', '&', '(', ')', ',', '.', '-' ,'--', '/']
    tempTokens = s.split(" ")
    tokens = []
    for token in tempTokens:
        if token.strip() not in illegalTokens:
            tokens.append(token.strip())
            
    return tokens

class CountVectorizer:
    def __init__(self, analyzer, ngram):
        self.ngram = ngram;
        self.analyzer = analyzer;

    def fit_transform(self, data):
        counts = [];
        dictionary = [];
        
        for sentence in data:
            sentence = sentence.lower();
            if "." in sentence:
                sentence = sentence.replace(".", "");
            if "," in sentence:
                sentence = sentence.replace(".", "");
            if "\"" in sentence:
                sentence = sentence.replace(".", "");
            
            tokens = self.analyzer(sentence);
            for i in range(len(tokens)):
                if i == (len(tokens)-(self.ngram-1)): 
                    break;
                gramWord = "";
                if self.ngram == 1: 
                    gramWord = tokens[i];
                elif self.ngram == 2:    
                    gramWord = tokens[i] + " " + tokens[i+1];
                elif self.ngram == 3:
                    gramWord = tokens[i] + " " + tokens[i+1] + " " + tokens[i+2];
                else:
                    #unhandled
                    exit(-1);
                if gramWord not in dictionary:
                    dictionary.append(gramWord);
                    counts.append(1.0);
                else:
                    gramWordIndex = dictionary.index(gramWord);
                    counts[gramWordIndex ] = counts[gramWordIndex ] + 1.0;
        self.dictionary = dictionary;
        self.counts = counts;
        return counts;

    def get_feature_names(self):
        return self.dictionary;
    def getCount(self):
        return self.counts;

def getUnigramCountsAndDictionary(data):
    counts = [];
    dictionary = [];

    for sentence in data:
        tokens = tokenize(sentence);
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token);
                counts.append(1.0);
            else:
                tokenIndex = dictionary.index(token);
                counts[tokenIndex] = counts[tokenIndex]  + 1.0;

    return {"dictionary":dictionary, "counts":counts};

def getBigramCountsAndDictionary(data):
    counts = [];
    dictionary = [];

    for sentence in data:
        tokens = tokenize(sentence);
        for i in range(len(tokens)):
            if i == len(tokens)-1 : 
                break;
            bigramWord = tokens[i] + " " + tokens[i+1];
            if bigramWord not in dictionary:
                dictionary.append(bigramWord);
                counts.append(1.0);
            else:
                bigramWordIndex = dictionary.index(bigramWord);
                counts[bigramWordIndex] = counts[bigramWordIndex] + 1.0;
    return {"dictionary":dictionary, "counts":counts};

