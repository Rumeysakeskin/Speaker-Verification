import numpy as np
import math
import string
import sys
class Viterbi:
    def __init__(self, probEmission, probTransition, testSet):
        self.delta = []
        self.states = list(string.ascii_lowercase)
        self.symbols = list(string.ascii_lowercase)
        self.corruptedTestSet = testSet
        self.emissionProbabilities = probEmission
        self.transitionProbabilities = probTransition
        self.initialProbabilities = float(1)/len(self.states)      # Same initial probabilities for every states

        self.FN = 0
        self.TP = 0
        self.totalWords = 0
        self.FP = 0

    def calculateInitialDelta(self, symbolChar):
        self.delta = []
        for i in range(0, len(self.states)):
            init = self.initialProbabilities
            emission = self.emissionProbabilities[i][self.symbols.index(symbolChar)]
            self.delta.append(math.log(emission) + math.log(init))

    def calculateDelta(self, symChar):
        backTrack = [None] * len(self.states)
        deltaTemp = [None] * len(self.states)
        for j in range(0, len(self.states)):
            maxValue = None
            for i in range(0, len(self.states)):
                transition = self.transitionProbabilities[i][j]
                mul = self.delta[i] + math.log(transition)
                if (maxValue is None or mul > maxValue ):
                    maxValue = mul
                    backTrack[j] = self.symbols[i]
            deltaCalc = maxValue + math.log((self.emissionProbabilities[j][self.symbols.index(symChar)]))
            deltaTemp[j] = deltaCalc
        self.delta = deltaTemp
        return backTrack

    def correctedWord(self, backTrack):
        temp = self.delta.index(max(self.delta))
        word = [self.symbols[temp]]
        if (backTrack):
            for l in backTrack:
                word.append(l[temp])
                temp = self.states.index(l[temp])
            word.reverse()
        return ''.join(word)

    def process(self, testSet):
        # TP: wrong-->correct
        # FN: wrong-->wrong
        # TP: correct-->wrong
        #Precision = TP/(TP + FN)
        #Recall = TP/(TP + FP)

        testSet = [x for x in testSet if x.isalpha()]
        counter = 0
        for word in self.corruptedTestSet:
            # print "==============================="
            # print "Actual:", testSet[counter]
            # print "Corrupted:", word
            backtrack = []
            for i in range(0, len(word)):
                symChar = word[i]
                if (i is 0):
                    self.calculateInitialDelta(symChar)
                else:
                    backtrack.insert(0, self.calculateDelta(symChar))

            corrected = self.correctedWord(backtrack)
            # print "Corrected:", corrected
            self.totalWords += 1
            if (testSet[counter] != word):
                # self.totalCorrectionsNeeded += 1
                # TP: True Positive (wrong->right)
                if (testSet[counter] == corrected):
                    self.TP += 1
                # FN: False Negative (wrong->wrong)
                if (word != corrected):
                    self.FN += 1

            if (testSet[counter] == word):
                # FP: False Positive (correct->wrong)
                if (word != corrected):
                    self.FP += 1


            counter += 1
            # print "==============================="

        # print "Total Correct Corrections:", self.TP
        # print "Total Corrections Needed:", self.FP
        # print "Total Corrected: ", self.FN
        print("==================================")
        print("Total Words          :", self.totalWords)
        print("True Positive  (W->C):", self.TP)
        print("False Negative (W->W):", self.FN)
        print("False Positive (C->W):", self.FP)
        print("----------------------------------")
        print("Recall: ", float(self.TP) / (self.TP + self.FN) * 100)
        print("Precision: ", float(self.TP) / (self.TP + self.FP) * 100)
        print("==================================")