'''
    Alogrithm to classify emails as spam/ham using Naive Bayes and Logistic Regression
    @author: Chandni Shankar
'''
import os, time
from os.path import join
import sys
import re
from collections import Counter
import math

class Instances:
    """
        Class to create instances with features in array and actual label of each instance
    """
    def __init__(self, label, data, idNum):
        self.label = label
        self.featuresCounter = data
        self.id = idNum
        self.features = []
        
    def addFeatures(self, data):
        self.features.append(data)
        
class Vocabulary:
    """
        Class to create vocabulary and initial prior associated 
    """
    def __init__(self, vocab, prior):
        self.vocab = vocab
        self.prior = prior  

def makeFeatureVector(file, vocab, stopWords):
    fileWords = Counter()
    for word in open(join(file)).read().split():
        if word not in stopWords and re.match("^[a-zA-Z]*$", word):
            fileWords[word] += 1
    featureVector = [1]
    for word in vocab:
        if word in fileWords:
            featureVector.append(fileWords[word]) 
        else:
            featureVector.append(0) 
    return featureVector
        
def trainLogisticReg(allInstances, vocab):
    """
        Function to train Logistic Regression
    """
    for instance in allInstances:
        instance.addFeatures(1)
        for word in vocab:
            if word in instance.featuresCounter:
                instance.addFeatures(instance.featuresCounter[word])
            else:
                instance.addFeatures(0)  
    
    weightVector = []
    for i in range(len(vocab) + 1):
        weightVector.append(0) # initializing all weights to 0
        
    #ham = 1, spam = 0
    eeta = 0.005
    lamb = 2
    #pyone = exp(sum (xiwi))/(1 + exp(sum (xiwi)))
    for i in range(1, 100):
        error = []
        for instance in allInstances:
            if instance.label == 'ham':
                actualY = 1
            else:
                actualY = 0
            wixi = 0.0
            for x in range(len(vocab) + 1):
                wixi += weightVector[x] * instance.features[x]               
            predY = math.exp(wixi)/ (1 + math.exp(wixi))
            error.append(actualY-predY)
        for index, w in enumerate(weightVector):
            allError = 0.0
            for insIndex, instance in enumerate(allInstances):
                allError += instance.features[index] * error[insIndex]
            weightVector[index] = (weightVector[index] + (eeta * allError) - (eeta * lamb * weightVector[index]))
    return weightVector 

def predictLabelLR(weights, featureVector):
    """
        Function to predict label of a test file for Logistic Regression
    """

    wixi = 0.0
    for index, w in enumerate(weights):
        wixi += (weights[index] * featureVector[index])
        
    hamProb = float(math.exp(wixi)) / float(1 + math.exp(wixi))
    spamProb = 1 - hamProb
    if hamProb >= spamProb:
        predictedLabel = 'ham'
    else:
        predictedLabel = 'spam'
    return predictedLabel

def calcAccuracyLR(testDir, weights, vocab, stopWords):
    """
        Function to calculate accuracy of Logistic Regression classifier
    """
    numFiles , accuracy = 0, 0
    for root, dirs, files in os.walk(testDir):
        if dirs:
            classes = dirs
        elif files:
            for file in files:
                numFiles += 1
                actualLabel = root.split('\\')[-1]
                featureVector = makeFeatureVector(join(root,file), vocab , stopWords)
                predlabel = predictLabelLR(weights, featureVector)
                if predlabel == actualLabel:
                    accuracy += 1
    return float(accuracy)/float(numFiles) * 100    

def trainNaiveBayes(classDocs, path, stopWords, classes):
    """
        Function to train Naive Bayes
    """
    vocab = set() #set of all unique words in the documents 
    idNum = 0
    allInstances = [] #training instance objects containing the actual label and feature vector
    totalWords = Counter() #Contains a counter for no.of words in ham and spam
    spamProb = Counter()
    hamProb = Counter()
    for files in classDocs:
        label = classes[classDocs.index(files)]
        root = path[classDocs.index(files)]
        for file in files:
            fileWords = Counter()
            for word in open(join(root,file)).read().split():
                if word not in stopWords and re.match("^[a-zA-Z]*$", word):
                    vocab.add(word)
                    fileWords[word] += 1
                    if label == 'ham':
                        index = 0
                        hamProb[word] += 1 #incrementing the ham counter for a word
                    else:
                        index = 1
                        spamProb[word] += 1 #incrementing the spam counter for the word
                    totalWords[index] += 1
            allInstances.append(Instances(label, fileWords, idNum))
            idNum = idNum + 1 
    return allInstances, vocab, spamProb, hamProb, totalWords
        
def naiveBayes1Laplace(vocab, hamProb, spamProb, totalWords):
    """
        Function to get ham prob and spam prob of all vocab words and adding 1 laplace
    """
    wordStats = {} #Dict of all vocab words and associated ham and spam probability
    vocabLen = len(vocab)
    for word in vocab:
        hamSpam = [0, 0] #index 0 for ham, index1 for spam
        if word in hamProb:
            hamSpam[0] = float(hamProb[word] + 1) / float(totalWords[0] + vocabLen)
        else:
            hamSpam[0] = float(1) / float(totalWords[0] + vocabLen)
        if word in spamProb:
            hamSpam[1] = float(spamProb[word] + 1) / float(totalWords[0] + vocabLen)
        else:
            hamSpam[1] =  float(1) / float(totalWords[1] + vocabLen)
        wordStats[word] = hamSpam
    return wordStats

def predictLabel(file, stopWords, wordProb, prior):
    """
        Function to predict label of a test file for Naive Bayes
    """
    hamProb = 0.0
    spamProb = 0.0
    for word in open(file).read().split():
        if word not in stopWords and re.match("^[a-zA-Z]*$", word) and word in wordProb:
            probList = wordProb[word]
            hamProb += math.log(probList[0]) #Using summation rule of log
            spamProb += math.log(probList[1])
    hamProb +=  math.log(prior['ham'])
    spamProb +=  math.log(prior['spam'])
    if hamProb >= spamProb:
        predictedLabel = 'ham'
    else:
        predictedLabel = 'spam'
    return predictedLabel

def calcAccuracy(testDir, stopWords, wordProb, prior):
    """
        Function to calculate accuracy of Naive Bayes classifier
    """
    numFiles , accuracy = 0, 0
    for root, dirs, files in os.walk(testDir):
        if dirs:
            classes = dirs
        elif files:
            for file in files:
                numFiles += 1
                actualLabel = root.split('\\')[-1]
                predlabel = predictLabel(join(root,file), stopWords, wordProb, prior)
                if predlabel == actualLabel:
                    accuracy += 1
    return float(accuracy)/float(numFiles) * 100
    
def main(argv):
    trainDir = argv[0]
    testDir = argv[1]
    stopWordsFile = argv[2]
    classDocs = []
    prior = {}
    total = 0
    i = 0
    path = []
    
    for root, dirs, files in os.walk(trainDir):
        if dirs:
            classes = dirs
        elif files:
            classDocs.append(files)
            path.append(root)
            total = (total + len(files))
                    
    for classLabel in classes:
        prior[classLabel] = float(len(classDocs[classes.index(classLabel)]))/total
        
    stopWords = ['subject', 're:', 'from' , 'to' , 'cc', 'ect', 'the']
    
    allInstances, vocab, spamProb, hamProb, totalWords = trainNaiveBayes(classDocs, path, stopWords, classes)

    wordProb = naiveBayes1Laplace(vocab, hamProb, spamProb, totalWords)
    accuracyNB = calcAccuracy(testDir, stopWords, wordProb, prior)

    print "Accuracy of Naive Bayes without Stop Words: ", accuracyNB
    
    weightVector = trainLogisticReg(allInstances, vocab)
    accuracyLR = calcAccuracyLR(testDir, weightVector, vocab, stopWords)
    print "Accuracy of Logistic Regression without Stop Words: ", accuracyLR
    
    
    for line in open(stopWordsFile):
        curLine = line.rstrip('\n')
        stopWords.append(curLine)
        
    allInstances2, vocab, spamProb, hamProb, totalWords = trainNaiveBayes(classDocs, path, stopWords, classes)
    wordProb = naiveBayes1Laplace(vocab, hamProb, spamProb, totalWords)
    accuracyNB2 = calcAccuracy(testDir, stopWords, wordProb, prior)
    print "Accuracy of Naive Bayes with Stop Words: ", accuracyNB2
    
    weightVector = trainLogisticReg(allInstances2, vocab)
    accuracyLR2 = calcAccuracyLR(testDir, weightVector, vocab, stopWords)
    print "Accuracy of Logistic Regression with Stop Words: ", accuracyLR2
   

if __name__=="__main__":
    main(sys.argv[1:])
