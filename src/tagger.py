import pickle
from collections import defaultdict
import numpy as np
import sys

class Tagger:

    def __init__(self):
        self.transitionProbabilities = defaultdict(dict)
        self.emissionProbabilities = defaultdict(dict)
        self.tagCounts = defaultdict(int)

    def _normalize(self, data):
        for previousTag in data:
            currentDict = data[previousTag]
            values = np.array(list(currentDict.values()))
            total = np.sum(values)
            for tag in currentDict:
                currentDict[tag] = currentDict[tag] / total
        return data

    def calculateCounts(self, data):
        transitionCounts = defaultdict(dict)
        emissionCounts = defaultdict(dict)
        previousTag = "START"

        for line in data:
            line = line.strip()
            if line:
                (word, tag) = line.strip().split("\t")
                self.tagCounts[tag] += 1
                currentDict = transitionCounts[previousTag]
                currentDict[tag] = currentDict.get(tag, 0) + 1
                currentDict = emissionCounts[tag]
                currentDict[word] = currentDict.get(word, 0) + 1
                previousTag = tag
            else:
                previousTag = "START"

        return (transitionCounts, emissionCounts)

    def train(self, filename):
        try:
            with open(filename, "r", encoding="utf-8") as file:
                (transitionCounts, emissionCounts) = self.calculateCounts(file)
                self.transitionProbabilities = self._normalize(transitionCounts)
                self.emissionProbabilities = self._normalize(emissionCounts)

        except IOError:
            print("File Error")
            sys.exit(1)

    def save(self, filename):
        try:
            with open(filename, "wb") as file:
                model = {
                    "transitionProbabilities": self.transitionProbabilities,
                    "emissionProbabilities": self.emissionProbabilities,
                    "tagCounts": self.tagCounts,
                }
                pickle.dump(model, file)

        except IOError:
            print("File Error")
            sys.exit(1)

    def load(self, filename):
        try:
            with open(filename, "rb") as file:
                model = pickle.load(file)
                self.transitionProbabilities = model["transitionProbabilities"]
                self.emissionProbabilities = model["emissionProbabilities"]
                self.tagCounts = model["tagCounts"]
        except IOError:
            print("File Error")
            sys.exit(1)


    def predictAll(self, data): 
        predictions = []
        currSentence = []

        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as file:
                data = file.readlines()

        for line in data:
            if line.strip():
                currSentence.append(line.strip())
            else:
                if currSentence:
                    sentencePredictions = self.viterbiAlgorithm(currSentence)
                    predictions.extend(sentencePredictions)
                    currSentence = []

                predictions.append("")

        if currSentence:
            sentencePredictions = self.viterbiAlgorithm(currSentence)
            predictions.extend(sentencePredictions)

        return predictions

    
    def _countCorrect(self, actualTags, predictedTags):

        total = 0
        correct = 0

        for actualTag, predictedTag in zip(actualTags, predictedTags):
            if actualTag:
                total += 1
                if predictedTag == actualTag:
                    correct += 1

        return (total, correct)

    def eval(self, filename):
        testData = []
        actualTags = []

        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    word, tag = line.strip().split("\t")
                    testData.append(word)
                    actualTags.append(tag)
                else:
                    testData.append("")
                    actualTags.append("")

        predictedTags = self.predictAll(testData)
        correct, total = self._countCorrect(actualTags, predictedTags)
        return round(correct / len(testData) * 100, 2) if total > 0 else 0.0

    def viterbiAlgorithm(self, sentence):
        states = list(self.tagCounts.keys())
        numStates = len(states)
        lenSentence = len(sentence)

        V = np.zeros((numStates, lenSentence))
        path = np.zeros((numStates, lenSentence), dtype=int)

        stateIdx = {state: i for i, state in enumerate(states)}
        for state in states:
            stateNum = stateIdx[state]
            trans = self.transitionProbabilities["START"].get(state, 0)
            emiss = self.emissionProbabilities[state].get(sentence[0], 0)
            V[stateNum, 0] = trans * emiss

        for t in range(1, lenSentence):

            for state in states:
                stateNum = stateIdx[state]
                previousProbabilities = V[:, t - 1]

                transition = np.array(
                    [self.transitionProbabilities[ps].get(state, 0) for ps in states]
                )
                emission = self.emissionProbabilities[state].get(sentence[t], 0)
                probs = previousProbabilities * transition * emission
                maxProb = np.max(probs)  # max probability
                maxState = np.argmax(probs)  # max state index
                V[stateNum, t] = maxProb
                path[stateNum, t] = maxState

        optimalPath = []
        currState = np.argmax(V[:, lenSentence - 1])
        optimalPath.append(states[currState])

        for t in range(lenSentence - 1, 0, -1):
            currState = path[currState, t]
            optimalPath.insert(0, states[currState])

        return optimalPath
