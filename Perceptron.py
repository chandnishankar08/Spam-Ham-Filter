'''
    Alogrithm to classify emails as spam/ham using Perceptron
    @author: Chandni Shankar
'''
import os, time
from os.path import join
import sys
import re
from collections import Counter

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

def makeFeatureVector(allInstances, vocab):
    for instance in allInstances:
        instance.addFeatures(1)
        for word in vocab:
            if word in instance.featuresCounter:
                instance.addFeatures(instance.featuresCounter[word])
            else:
                instance.addFeatures(0)
        
def trainPerceptron(allInstances, vocab):
    """
        Function to train perceptron
    """    
    weightVector = []
    for i in range(len(vocab) + 1):
        weightVector.append(0) # initializing all weights to 0
        
    eeta = 0.2
    #wi = wi + dwi
    #dwi = eeta * (t - o) * xi
    for i in range(1, 10):
        for instance in allInstances:
            if instance.label == 'ham':
                t = 1
            else:
                t = 0
            totalSum = 0.0
            for index, w in enumerate(weightVector):
                totalSum += (w * instance.features[index])
            if totalSum >= 0 :
                o = 1
            else:
                o = 0
            if ((t-o) != 0):
                for index, w in enumerate(weightVector):
                        dw = eeta * (t - o) * instance.features[index]
                        weightVector[index] = (weightVector[index] + dw)               
    return weightVector 

def predictLabel(weights, featureVector):
    """
        Function to predict label of a test file for Logistic Regression
    """

    wixi = 0.0
    for index, w in enumerate(weights):
        wixi += (weights[index] * featureVector[index])
        
    if wixi >= 0:
        predictedLabel = 'ham'
    else:
        predictedLabel = 'spam'
    return predictedLabel

def calcAccuracy(testDir, weights, vocab, stopWords):
    """
        Function to calculate accuracy of Logistic Regression classifier
    """
    numFiles , accuracy = 0, 0
    testInstances, testVocab = makeInstances(testDir, stopWords)
    for instance in testInstances:
        actualLabel = instance.label
        predlabel = predictLabel(weights, instance.features)
        if predlabel == actualLabel:
            accuracy += 1
    numFiles = len(testInstances)
    print numFiles
    return float(accuracy)/float(numFiles) * 100    

def makeInstances(trainDir, stopWords):
    """
        Function to make instances from each data to get actual label, word count of repeating words
    """
    vocab = set() #set of all unique words in the documents 
    idNum = 0
    allInstances = [] #training instance objects containing the actual label and feature vector
    index = 0
    classes = []
    for root, dirs, files in os.walk(trainDir):
        if dirs:
            classes = dirs
        elif files:
            label = classes[index]
            for file in files:
                fileWords = Counter()
                for word in open(join(root,file)).read().split():
                    if word not in stopWords and re.match("^[a-zA-Z]*$", word):
                        vocab.add(word)
                        fileWords[word] += 1
                allInstances.append(Instances(label, fileWords, idNum))
                idNum = idNum + 1 
            index = index + 1
    makeFeatureVector(allInstances, vocab)
    return allInstances, vocab
            
def main(argv):
    trainDir = argv[0]
    testDir = argv[1]
    stopWordsFile = argv[2]
    stopWords = ['subject', 're:', 'from' , 'to' , 'cc', 'ect', 'the']
    
    trainInstances, vocab = makeInstances(trainDir, stopWords)
    weightVector = trainPerceptron(trainInstances, vocab)
    accuracy = calcAccuracy(testDir, weightVector, vocab, stopWords)
    print "Accuracy of perceptron: ", accuracy
   

if __name__=="__main__":
    main(sys.argv[1:])
