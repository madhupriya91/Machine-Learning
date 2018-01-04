import os
import re
import sys
import math
from stop_words import get_stop_words
stop_words = get_stop_words('english')

directory=str(sys.argv[1])
data = []

for root, dirs, files in os.walk(directory, topdown=False):
   for name in files:
      fileName = os.path.join(root, name)
      with open(str(fileName)) as fileName:
          header = 1
          for words in fileName:
              if header is 1:
                  matchObj = re.match(r'Lines:',words)
                  if matchObj:
                      header=0
                  continue
              words = re.sub('[^a-zA-Z0-9\n\']',' ', words)
              word=words.strip().split()
              for word1 in word :
                  word1=word1.lower().strip(' \\"')
                  #word1 = word1.strip(' \\"')
                  if word1 not in stop_words: #removing stop words
                      data.append(word1) # appending all words in the documents

def getVocab(dataSet):
    vocabDict = {}
    for word in dataSet:
        if word not in vocabDict:
            vocabDict.update({word: 1})
        else:
            vocabDict[word] += 1
    return vocabDict # getting vocabulory

def countDocs(directory): # counting number of docs in each directory
    count=0
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            count+=1
    return count

def concatText(classPath): # getting words from each folder
    textcDict={}
    for files in os.listdir(classPath):
        fileName = os.path.join(classPath, files)
        with open(fileName) as fileName:
            header = 1
            for words in fileName:
                if header is 1:
                    matchObj = re.match(r'Lines:', words)
                    if matchObj:
                        header = 0
                    continue
                x = re.sub('[^a-zA-Z0-9\n\']', ' ', words).split()
                for word in x:
                    word = word.lower()
                    if word not in stop_words:
                        if word not in textcDict:
                            textcDict.update({word:1})
                        else:
                            textcDict[word]+=1
    return textcDict

def countForAll(vocabulory,allWordsClassCount,textC):
    for word in vocabulory:
        if word in textC:
            value = textC[word]
        else:
            value = 0
        allWordsClassCount.update({word: value})
    return allWordsClassCount

priorDict={}
condProbDict={}
vocabulory=getVocab(data)

# creating training model
def trainModel():
    docCount=countDocs(directory)
    docCountDict={}

    for files in os.listdir(directory):
        denom=len(vocabulory)
        folderPath=str(directory)+"\\"+str(files)
        countThis = 0
        className = str(files)
        for docs in os.listdir(folderPath):
            countThis+=1
        docCountDict.update({className:countThis})
        priorProb=countThis/docCount # finding prior probabilty
        priorDict.update({className:priorProb})
        textC=concatText(folderPath)
        innerDict={}
        for word in vocabulory:
            if word in textC:
                denom+=textC[word]
            else:
                denom+=0
        for word in vocabulory:
            numer=0
            if word in textC:
                numer=textC[word]+1
            else:
                numer=1
            condProb=numer/denom
            innerDict.update({word:condProb})
        condProbDict.update({className:innerDict}) #updating conditionbal probability

#creating testmodel
def testModel(filePath):
    testWords = []
    with open(filePath) as fileName:
        header = 1
        for words in fileName:
            if header is 1:
                matchObj = re.match(r'Lines:', words)
                if matchObj:
                    header = 0
                continue
            x = re.sub('[^a-zA-Z0-9]', ' ', words).split()
            for word in x:
                word = word.lower().strip()
                if word not in stop_words:
                    testWords.append(word)
    scoreValue = {}
    for thisClass in condProbDict:
        if thisClass not in scoreValue:
            scoreValue.update({thisClass:math.log(priorDict[thisClass])})
        for word in testWords:
            if word not in vocabulory:
                continue
            scoreValue[thisClass]+=math.log(condProbDict[thisClass][word]) # calculating likelihood
    key_max = max(scoreValue.keys(), key=(lambda k: scoreValue[k]))
    return key_max

trainModel()
testDirect=str(sys.argv[2])
testDocCount=0
correctPred=0
for root, dirs, files in os.walk(testDirect, topdown=False):
   for name in files:
      testDocCount+=1
      filePath = os.path.join(root, name)
      predictedClass=testModel(filePath)
      actualClass=str(root).split("\\")[-1]
      if predictedClass==actualClass:
          correctPred+=1
accuracy=(correctPred/testDocCount)*100
print("Accuracy : "+str(accuracy)+"%")




