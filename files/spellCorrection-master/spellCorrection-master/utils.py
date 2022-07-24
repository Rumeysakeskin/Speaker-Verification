import re
import numpy as np




def readLinesFromFile( fileDirectory ) :
    with open(fileDirectory) as file:
        lines = file.readlines();
    return lines;

def writeLinesToFile( fileDirectory , lines):
    with open( fileDirectory ) as file:
        file.writelines(lines)

def getCorruptedWordWithCorrectWordInSentence(sentence):
    errors = re.findall(r'<ERR targ=[\?\'\-\w\s,]+>[\?\'\-\w\s,]+</ERR>', sentence, re.I)
    corruptedWords = []
    correctWords = []
    for error in errors:
        correctWords.append(getCorrectWordFromErrorTags(error))
        corruptedWords.append(getWrongWordFromErrorTags(error))
    return (corruptedWords, correctWords)

def getErrors(sentence):
    return re.findall(r'<ERR targ=[\?\'\-\w\s,]+>[\?\'\-\w\s,]+</ERR>', sentence, re.I)
def getCorrections(sentence):
    tokens = sentence.split(" ");
    corrections = list();
    for token in tokens:
        if token.startswith("ERROR:"):
            corrections.append(token)
    return corrections;
#gets sentence with error tags and return 
def escapeErrorsInSentence(sentence):
    correctedSentence = sentence;
    errors = re.findall(r'<ERR targ=[\?\'\-\w\s,]+>[\?\'\-\w\s,]+</ERR>', correctedSentence, re.I)
    
    for error in errors:
        correctedSentence = getCorrectedSentence(correctedSentence, error);   
    correctedSentence = correctedSentence.replace(".", " ");
    correctedSentence = correctedSentence.replace(",", " ");
    correctedSentence.lower();
    return correctedSentence.strip();

    
def  getCorrectedSentence(sentence, error):

    sentenceParts = sentence.split(error);
        
    if len(sentenceParts) == 2:
        return sentenceParts[0] + getCorrectWordFromErrorTags(error) + sentenceParts[1];     
    elif len(sentenceParts) > 2:
        correntSentence = sentenceParts[0] + getCorrectWordFromErrorTags(error) + sentenceParts[1];
        for i in range(2, len(sentenceParts)):
            correntSentence  = correntSentence + error + sentenceParts[i];
        return correntSentence;
    else:
        return sentenceParts[0] + getCorrectWordFromErrorTags(error) + sentenceParts[1];     

def getEstimatedWordFromCorrectionTags(correctionString):
    word = " ";
    if correctionString.startswith("ERROR:"):
        temp = correctionString.split("*");
        word= temp[3];
    return word;
        # gets correct word from error tags
def getCorrectWordFromErrorTags(errorString):
    matchObj = re.search(r'targ=[\?\'\-\w\s,]+>', errorString, re.I);
    matchString = matchObj.group()
    ##HERE and THERE
    temp = matchString[5:-1]
    tempList = temp.split(" ");
    if len(tempList) > 1:
        temp2 = ""
        for temp3 in tempList:
            temp2 = temp2 + "-" + temp3
        return temp2;
    else:
        return temp

def getWrongWordFromErrorTags(errorString):
    matchObj = re.search(r'>[\?\'\-\w\s,]+<', errorString, re.I);
    matchString = matchObj.group()
    temp = matchString[1:-1]
    tempList = temp.strip().split(" ");
    if len(tempList) > 1:
        temp2 = ""
        for temp3 in tempList:
            temp2 = temp2 + "-" + temp3
        return temp2.strip();
    else:
        return temp.strip();

def editDistance(word1, word2):
    word1 = word1.strip();
    word1 = word1.lower();
    word2 = word2.strip();
    word2 = word2.lower();

    lengthWord1 = len(word1);
    lengthWord2 = len(word2);

    #init. edit distance array
    distanceTable = np.zeros((lengthWord1+1, lengthWord2+1));
    for i in range(lengthWord1+1):
        distanceTable[i,0] = i;
    for i in range(lengthWord2+1):
        distanceTable[0,i] = i;
    #generating distances
    for i in range (1, lengthWord1+1, 1):
        for j in range(1, lengthWord2+1, 1):
            if word1[i-1] == word2[j-1]:
                distanceTable[i][j] = distanceTable[i-1][j-1]
            else:
                minDistance = distanceTable[i-1][j-1]
                if minDistance > distanceTable[i][j-1]:
                    minDistance = distanceTable[i][j-1]
                if minDistance > distanceTable[i-1][j]:
                    minDistance = distanceTable[i-1][j]
                distanceTable[i][j] = minDistance + 1;
    return distanceTable[lengthWord1][lengthWord2]
def getDeletedCharacterKey(corruptedWord, word):
    deletedCharacter = "";
    for i in range(len(word)):
        if i == len(corruptedWord):
            deletedCharacter = word[i];
            break;
        elif corruptedWord[i] != word[i]:
            deletedCharacter = word[i];
            break;
    return deletedCharacter;
    
def getSubstitutedCharacterKey(corruptedWord, word):
    wrongCharacter = "";
    correctCharacter = "";
    for i in range(len(corruptedWord)):
        if corruptedWord[i] != word[i]:
            wrongCharacter = corruptedWord[i];
            correctCharacter = word[i];
            
            break;
    key =  correctCharacter.strip() + " " + wrongCharacter.strip();
    return key;
def getInsertedCharacterKey(corruptedWord, word):
    insertedCharacter = "";
    for i in range(len(corruptedWord)):
        if i == len(word):
            insertedCharacter = corruptedWord[i];
            break;
        elif corruptedWord[i] != word[i]:
            insertedCharacter = corruptedWord[i];
            break;
    key = insertedCharacter;
    return key;
    
    
def evaluateSentence(rawSentence, estimatedSentece):
    errorsWithTags = getErrors(rawSentence);
    correctionsWithTags = getCorrections(estimatedSentece);
    
    errorCount = len(errorsWithTags);
    trueEstimationCount = 0;
    
    correctWords = list();
    estimatedWords = list();
    
    for errorWithTag in errorsWithTags:
        correctWords.append(getCorrectWordFromErrorTags(errorWithTag).lower().strip());

    for correctionWithTag in correctionsWithTags:
        estimatedWords.append(getEstimatedWordFromCorrectionTags(correctionWithTag));

    for word in estimatedWords:
        if word in correctWords:
            trueEstimationCount = trueEstimationCount + 1;
    return (errorCount, trueEstimationCount);