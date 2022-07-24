import hmm as HiddenMarkov
import gc 
import utils as utls
import sys


datasetFile = "dataset.txt" 
outFile = "out.txt" 
testDataSize = 200 

# datasetFile = sys.argv[1]
# outFile = sys.argv[2]

print("initializing hmm...") 
hiddenMarkovModel = HiddenMarkov.HMM(datasetFile) 


print("Correcting the sentences...")
results = list() 
data = hiddenMarkovModel.errorFullDataSet[:testDataSize] 
dataLength = len(data) 
for i in range(dataLength):
    temp = hiddenMarkovModel.viterbi(data[i]) 
    results.append(temp) 
    if not (i%100):
        gc.collect()

#evaluation
correctEstimatedWordCount = 0 
wrongTypedWordCount = 0 
for i in range(dataLength):
    counts = utls.evaluateSentence(data[i], results[i]) 
    correctEstimatedWordCount = correctEstimatedWordCount + counts[1] 
    wrongTypedWordCount = wrongTypedWordCount + counts[0] 


print(wrongTypedWordCount) 
print(correctEstimatedWordCount) 
acc = correctEstimatedWordCount/wrongTypedWordCount 

out = open(outFile,'w') 
out.write("------- Evaluation -------\n") 
out.write("Accuracy : " + str(acc)) 
out.write("\nTotal miss typed word count is : "+ str(wrongTypedWordCount)) 
out.write("\nTotal correct estimated word count is : "+ str(correctEstimatedWordCount)) 
out.write("\n\nSentences after viterbi:") 
for result in results:
    out.write("\n" + result) 
    

out.write("\n----EOF----") 
out.close()


