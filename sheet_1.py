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
    data = pd.read_csv(path, header=None).to_numpy()
    X = data[:, 1:]
    y = data[:, 0]
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
    for entry, res in zip(X, y):
        if entry[idx] == 0:
            negCases.append(res)
        elif entry[idx] == 1:
            posCases.append(res)
        else:
            return -1
    return (
        entropy(y)
        - len(posCases) / len(y) * entropy(posCases)
        - len(negCases) / len(y) * entropy(negCases)
    )


class ID3Tree:
    def __init__(self) -> None:
        self.root = Node()

    def train(self, X, y) -> None:
        # recursively run node train
        self.root.train(X, y)

    def predict(self, X) -> None:
        # recursively run node predict
        result = []
        for entry in X:
            result.append(self.root.predict(entry))
        return np.array(result)

    def print(self):
        # recursively run node print
        self.root.print()


class Node:
    def __init__(self, label=None, attribute=None, used_attributes=[]) -> None:
        self.leaf = False
        self.attribute = attribute
        self.child_attribute = None
        self.label = label
        self.used_attributes = used_attributes
        self.root = False
        if label == None:
            self.root = True
        self.finalLabel = None

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
        if len(self.used_attributes) == len(X[0]):
            self.leaf = True
            ones = 0
            zeros = 0
            for label in y:
                if label == 1:
                    ones += 1
                else:
                    zeros += 1

            if ones > zeros:
                self.finalLabel = 1
            else:
                self.finalLabel = 0
            return

        best_attribute = Node.select_best_attribute(X, y)

        x_copy = X.copy()
        while best_attribute in self.used_attributes:
            x_copy[:, best_attribute] = -1
            best_attribute = Node.select_best_attribute(x_copy, y)

        if information_gain(X, y, best_attribute) == 0:
            self.leaf = True
            ones = 0
            zeros = 0
            for label in y:
                if label == 1:
                    ones += 1
                else:
                    zeros += 1

            if ones > zeros:
                self.finalLabel = 1
            else:
                self.finalLabel = 0
            return

        self.child_attribute = best_attribute
        self.used_attributes.append(best_attribute)

        leftData = []
        rightData = []
        left_y = []
        right_y = []
        for i in range(len(y)):
            if X[i][best_attribute] == 1:
                rightData.append(X[i])
                right_y.append(y[i])
            else:
                leftData.append(X[i])
                left_y.append(y[i])
        leftData = np.array(leftData)
        rightData = np.array(rightData)

        right_y = np.array(right_y)
        left_y = np.array(left_y)

        self.rightChild = Node(1, best_attribute, self.used_attributes)
        self.leftChild = Node(0, best_attribute, self.used_attributes)
        self.rightChild.train(rightData, right_y)
        self.leftChild.train(leftData, left_y)

        if self.leftChild.leaf and self.rightChild.leaf:
            if self.leftChild.finalLabel == self.rightChild.finalLabel:
                self.finalLabel = self.leftChild.finalLabel
                self.leaf = True
                del self.leftChild
                del self.rightChild

    def predict(self, X) -> None:
        if self.leaf:
            return self.finalLabel
        if X[self.child_attribute] == 0:
            return self.leftChild.predict(X)
        else:
            return self.rightChild.predict(X)

    def print(self):
        if self.root:
            print("Root")
        else:
            if self.leaf == True:
                print(
                    f"Attribute : {self.attribute} | Label : {self.label} | Leaf : {self.leaf} | Final : {self.finalLabel}"
                )
                return
            else:
                print(
                    f"Attribute : {self.attribute} | Label : {self.label} | Child : {self.child_attribute}"
                )
        self.leftChild.print()
        self.rightChild.print()

    def select_best_attribute(X, y):
        bestAttribute = -1
        bestGain = -1
        for i in range(0, len(X[0])):
            gain = information_gain(X, y, i)
            if gain > bestGain:
                bestGain = gain
                bestAttribute = i

        return bestAttribute


if __name__ == "__main__":
    train_data = load_data("SPECT.train")
    test_data = load_data("SPECT.test")

    tree = ID3Tree()
    tree.train(train_data[0], train_data[1])
    tree.print()
    test = tree.predict(test_data[0])

    correct = 0
    for res, val in zip(test, test_data[1]):
        if res == val:
            correct += 1

    print(f"Accuracy = {int(round(correct/len(test) * 100, 0))}%")
