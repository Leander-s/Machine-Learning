import numpy as np
import pandas as pd
from math import log2


def load_data(path):
    """
    Load data from an external file.

    Args:
        path (str): The file path

    Returns:
        X, y (tuple): A data matrix and a label vector
    """
    data = pd.read_csv(path, header=None)
    X = data.values
    y = data.iloc[:, 0].values
    return X, y


def entropy(y):
    entropy = 0
    bins = np.bincount(y)
    for bin in bins:
        try:
            entropy -= bin / len(y) * log2(bin / len(y))
        except:
            entropy -= 0
    return entropy


def information_gain(X, y, idx):
    posCases = []
    negCases = []
    for entry in X:
        if entry[idx] == 0:
            negCases.append(entry[0])
        else:
            posCases.append(entry[0])

    return (
        entropy(y)
        - len(posCases) / len(y) * entropy(posCases)
        - len(negCases) / len(y) * entropy(negCases)
    )


class ID3Tree:
    def __init__(self, X, y) -> None:
        self.root = Node(Node.select_best_attribute(X, y))

    def train(self, X, y) -> None:
        # recursively run node train
        pass

    def predict(self, X) -> None:
        # recursively run node predict
        pass

    def print(self):
        # recursively run node print
        pass


class Node:
    def __init__(self, attribute, label) -> None:
        self.leaf = False
        self.label = label
        self.finalLabel = None
        self.attribute = attribute

    def train(self, X, y) -> None:
        # check if data is "pure"
        # select best attribute to split upon
        # initialize the new nodes
        # create new subsets of X
        # check if some of the new nodes are leaves
        # train the new nodes on each subset
        if entropy(y) == 0:
            self.leaf = True
            self.finalLabel = y[0]
            return
        bestAttribute = Node.select_best_attribute(X, y)
        leftData = []
        rightData = []
        for i in range(len(y)):
            if X[i][bestAttribute] == 1:
                rightData.append(X[i])
            else:
                leftData.append(X[i])
        leftData = np.array(leftData)
        rightData = np.array(rightData)

        right_y = rightData[:, 0]
        left_y = leftData[:, 0]

        self.rightChild = Node(bestAttribute, 1)
        self.leftChild = Node(bestAttribute, 0)
        self.rightChild.train(rightData, right_y)
        self.leftChild.train(leftData, left_y)

    def predict(self, X) -> None:
        pass

    def print(self):
        pass

    def select_best_attribute(X, y):
        bestAttribute = 1
        bestGain = 0
        for i in range(1, len(X[0])):
            gain = information_gain(X, y, i)
            if gain > bestGain:
                bestAttribute = i

        return bestAttribute


if __name__ == "__main__":
    train_data = load_data("SPECT.train")
    test_data = load_data("SPECT.test")
    print(Node.select_best_attribute(train_data[0], train_data[1]))
    print(information_gain(train_data[0], train_data[1], 22))
    pass
