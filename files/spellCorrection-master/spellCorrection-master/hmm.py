import utils as utls
import numpy as np
import ngram as ng
import gc


class HMM:
    def __init__(self, dataFile):
        self.dataFile = dataFile;
        self.initializeDatasets();
        self.initializeCorruptedWordDictionary();
        self.initializeNGramModels();
        self.initializeTransactionProbabilities();
        self.initializeEmissionProbabilities();
        
    # gets dataset with error tags and returns the dataset without errors(with correct words)
    def initializeDatasets(self):
        self.errorFullDataSet = utls.readLinesFromFile(self.dataFile);
        self.errorFreeDataset = [utls.escapeErrorsInSentence(sentence) for sentence in self.errorFullDataSet];
    def initializeCorruptedWordDictionary(self):
        corruptedWordDictionary = dict();
        for errorFullSentence in self.errorFullDataSet:
            (corruptedWords, correctWords )= utls.getCorruptedWordWithCorrectWordInSentence(errorFullSentence);
            for i in range(len(correctWords)):
                if corruptedWords[i] in corruptedWordDictionary:
                    correspondingCorrectWords = corruptedWordDictionary.get(corruptedWords[i])
                    if (correctWords[i] not in correspondingCorrectWords):
                        correspondingCorrectWords.append(correctWords[i])
                        corruptedWordDictionary[corruptedWords[i]] = correspondingCorrectWords
                else:
                    correspondingCorrectWords = []
                    correspondingCorrectWords.append(correctWords[i])
                    corruptedWordDictionary[corruptedWords[i]] = correspondingCorrectWords
        self.corruptedWordDictionary = corruptedWordDictionary;
        self.corruptedWordList = list(self.corruptedWordDictionary.keys());
        
    def initializeNGramModels(self):
        uniGramCV = ng.CountVectorizer(analyzer=ng.tokenize, ngram=1);
        biGramCV = ng.CountVectorizer(analyzer=ng.tokenize, ngram=2);
        uniTempCounts = np.asarray(uniGramCV.fit_transform(self.errorFreeDataset))
        biTempCounts = np.asarray(biGramCV.fit_transform(self.errorFreeDataset))
        self.uniGramDictionary = uniGramCV.get_feature_names()
        self.biGramDictionary = biGramCV.get_feature_names()
        self.uniGramModel = ng.UniGramModel(self.uniGramDictionary, uniTempCounts, True)
        self.biGramModel = ng.BiGramModel(self.biGramDictionary, biTempCounts, True)
        
    def initializeTransactionProbabilities(self):
        print("-->Calculating initial Probabilities...");
        self.uniGramProbabilities = self.uniGramModel.calculateProbabilityTable();
        print("-->Calculating transaction Probabilities...");
        self.transactionProbabilities = self.biGramModel.calculateProbabilityTable(self.uniGramModel.dictionary, self.uniGramModel.countTable);

    def initializeEmissionProbabilities(self):
        print("-->Calculating emission Probabilities...");
        possibleCorrectWordsDictionary = dict();
        
        substitutionDictionary = dict();
        deletionDictionary = dict();
        insertionDictionary = dict();
        for i in range(len(self.corruptedWordList)):
            corruptedWord= self.corruptedWordList[i];
            #getting correct words with edit distance 1
            possibleCorrectWords = []

            for word in self.uniGramDictionary:
                word = word.strip();
                word = word.lower();
                corruptedWord = corruptedWord.strip();
                corruptedWord = corruptedWord.lower();
                if (utls.editDistance(word, corruptedWord) == 1):
                    possibleCorrectWords.append(word);
                    lengthDifference = len(word) - len(corruptedWord)
                    #SUBSTITION
                    if lengthDifference == 0: #subs
                        key = utls.getSubstitutedCharacterKey(corruptedWord, word);
                        if key in substitutionDictionary:
                            substitutionDictionary[key] = substitutionDictionary[key]+1;
                        else:
                            substitutionDictionary[key] = 1;
                    #DELETION --> can be changed
                    elif lengthDifference == -1:
                        key = utls.getDeletedCharacterKey(corruptedWord, word);
                        if key in deletionDictionary:
                            deletionDictionary[key] = deletionDictionary[key] + 1;
                        else:
                            deletionDictionary[key] = 1;
                    #INSERTION --> can be changed
                    else:
                        key = utls.getInsertedCharacterKey(corruptedWord, word);
                        if key in insertionDictionary:
                            insertionDictionary[key] = insertionDictionary[key] + 1;
                        else:
                            insertionDictionary[key] = 1;
                    
            possibleCorrectWordsDictionary[corruptedWord] = possibleCorrectWords;
                    
                    
        self.possibleCorrectWordsDictionary = possibleCorrectWordsDictionary;
        self.deletionDictionary = deletionDictionary;
        self.insertionDictionary = insertionDictionary;
        self.substitutionDictionary = substitutionDictionary;
        
        
        #calculating emission probs
        emissonProbabilityDictionary = dict();
        
        for key in self.possibleCorrectWordsDictionary:
            value = list();
            possibleCorrectWords = self.possibleCorrectWordsDictionary[key];
            for correctWord in possibleCorrectWords:
                prob = 0;
                if (len(key) - len(correctWord)) == 0: # subs
                    editedCharacter = utls.getSubstitutedCharacterKey(key, correctWord);
                    prob = self.calculateEmissionProbability(self.substitutionDictionary, editedCharacter);
                elif (len(key) - len(correctWord)) == -1: # deletion
                    editedCharacter = utls.getDeletedCharacterKey(key, correctWord);
                    prob = self.calculateEmissionProbability(self.deletionDictionary, editedCharacter);               
                else:
                    editedCharacter = utls.getInsertedCharacterKey(key, correctWord);
                    prob = self.calculateEmissionProbability(self.insertionDictionary, editedCharacter);               
                
                if(prob == 0):
                    print("huston we got a problem")
                value.append((correctWord, prob));
            emissonProbabilityDictionary[key] = value;
        self.emissonProbabilityDictionary = emissonProbabilityDictionary;
    def calculateEmissionProbability(self, editDictionary, editedChar):
        totalCount = 0;
        for key in editDictionary:
            totalCount = editDictionary[key] + totalCount;
        if editedChar in editDictionary:
            return float(editDictionary[editedChar]/totalCount);
        else:
            return float(1/totalCount);
    
    def viterbi(self, errorFullSentence):
        errors = utls.getErrors(errorFullSentence);
        possibleSentences = list();        
        newPossibleSentences = list();
        for error in errors:
            wrongWord = utls.getWrongWordFromErrorTags(error);
            if wrongWord.strip() in self.emissonProbabilityDictionary:    
                possibleWords = self.possibleCorrectWordsDictionary[wrongWord.strip()]
                if len(possibleSentences) == 0:
                    l = len(possibleWords)
                    for i in range(l):
                        possibleSentences.append(errorFullSentence.replace(error, ("ERROR:*"+wrongWord+"*+CORRECTION:*"+possibleWords[i])));
                else:
                    del newPossibleSentences[:];
                    l = len(possibleWords)
                    for i in range(l):
                        t = len(possibleSentences)
                        for j in range(t):
                            newPossibleSentences.append(possibleSentences[j].replace(error, "ERROR:*"+wrongWord+"*+CORRECTION:*"+possibleWords[i]));
                    del possibleSentences[:];
                    possibleSentences = list(newPossibleSentences);
            else:
                if len(possibleSentences) == 0:
                    possibleSentences.append(errorFullSentence.replace(error, ("ERROR:*"+wrongWord+"*+CORRECTION:*NOTFOUND!")));
                else:
                    del newPossibleSentences[:];
                    l = len(possibleSentences);
                    for i in range(l):
                        newPossibleSentences.append(possibleSentences[i].replace(error, ("ERROR:*"+wrongWord+"*+CORRECTION:*NOTFOUND!")));
                    del possibleSentences[:];
                    possibleSentences = list(newPossibleSentences);
        #now going to calculate probs.
        probs = [self.getProbabilityOfSentence(sentence) for sentence in possibleSentences];
        index = -1;
        if len(probs) > 0 :
            maxProb = probs[0];        
            index = 0;
            for prob in probs:
                if prob > maxProb:
                    maxProb = prob;
            index = probs.index(maxProb)
        gc.collect()
        if index == -1:
            return errorFullSentence;
        else:    
            return possibleSentences[index];
    

    def getProbabilityOfSentence(self, sentence):
        sentence = sentence.replace(".", " ");
        sentence = sentence.replace(",", " ");
        sentence.strip();
        tokens = sentence.split(" ");
        prob = 0;
        for i in range(len(tokens)):
            token = tokens[i].strip();
            if i == 0: # initial prob
                if token.startswith("ERROR:"):
                    temp = token.split("*");
                    wrongWord = temp[1];
                    correctWord = temp[3];
                    prob = prob + self.getEmissionProbability(wrongWord, correctWord);
                    token = correctWord;
                prob = prob + self.getInitialProbability(token);
            else:
                if token.startswith("ERROR:"):
                    temp = token.split("*");
                    wrongWord = temp[1];
                    correctWord = temp[3];
                    prob = prob + self.getEmissionProbability(wrongWord, correctWord);
                    token = correctWord;
                prob = prob + self.getTransitionProbability(tokens[i-1], token);
        return prob;                                                            

    def getInitialProbability(self, word):
        prob = 0.000000000001;
        if word in self.uniGramDictionary:
            index = self.uniGramDictionary.index(word);
            prob = self.uniGramProbabilities[index];
        return prob;
    def getTransitionProbability(self, word1, word2):
        prob = 0.000000000001;
        if word1 in self.uniGramDictionary and word2 in self.uniGramDictionary:
            index1 = self.uniGramDictionary.index(word1);
            index2 = self.uniGramDictionary.index(word2);
            prob = self.transactionProbabilities[index1, index2];
        return prob;
        
    def getEmissionProbability(self, wrongWord, correctWord):
        prob = 0.000000000001;
        if wrongWord in self.emissonProbabilityDictionary:
            wordProbList = self.emissonProbabilityDictionary[wrongWord];
            for temp in wordProbList:
                if temp[0] == correctWord:
                    prob = temp[1];
        return prob;