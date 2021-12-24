import math
from time import time as now
import json
import os


IMAGE_SIZE = 28 * 28


def split(examples, used, trait):
    """
    examples is a list of lists. every list contains the attributes, the last item is the class. all items are 0/1.
    splits examples into two lists based on trait (attribute).
    updates used that trait was used.
    """
    newEx = [
        [],
        [],
    ]  # newEx is a list of two lists, list of Ex that Ex[trait]=0 and list of Ex that Ex[trait]=1
    if trait < 0 or trait > len(examples[0]) - 2 or used[trait] == 0:
        return newEx  # illegal trait
    for e in examples:
        newEx[e[trait]] += [e]
    used[trait] = 0  # used is a list that marks trait as used
    return newEx


def isSameClass(examples):
    """
    returns 0 if all the examples are classified as 0.
    returns 1 if all the examples are classified as 1.
    returns 7  if there are no examples.
    returns -2 if there are more zeros than ones.
    returns -1 if there are more or equal ones than zeros.
    """
    if examples == []:
        return 7
    zo = [0, 0]  # zo is a counter of zeros and ones in class
    for e in examples:
        zo[e[-1]] += 1
    if zo[0] == 0:
        return 1
    if zo[1] == 0:
        return 0
    if zo[0] > zo[1]:
        return -2
    else:
        return -1


def infoInTrait(examples, i):
    """
    calculates the information in trait i using Shannon's formula
    """
    count = [
        [0, 0],
        [0, 0],
    ]  # [no. of ex. with attr.=0 and clas.=0,no. of ex. with attr.=0 and clas.=1],
    # [no. of ex. with attr.=1 and clas.=0,no. of ex. with attr.=1 and clas.=1]
    for e in examples:
        count[e[i]][e[-1]] += 1
    x = 0
    # Shannon's formula
    if count[0][0] != 0 and count[0][1] != 0:
        x = count[0][0] * math.log((count[0][0] + count[0][1]) / count[0][0]) + count[
            0
        ][1] * math.log((count[0][0] + count[0][1]) / count[0][1])
    if count[1][0] != 0 and count[1][1] != 0:
        x += count[1][0] * math.log((count[1][0] + count[1][1]) / count[1][0]) + count[
            1
        ][1] * math.log((count[1][0] + count[1][1]) / count[1][1])
    return x


def minInfoTrait(examples, used):
    """
    used[i]=0 if trait i was already used. 1 otherwise.

    Returns the number of the trait with max. info. gain.
    If all traits were used returns -1.
    """
    minTrait = m = -1
    for i in range(len(used)):
        if used[i] == 1:
            info = infoInTrait(examples, i)
            if info < m or m == -1:
                m = info
                minTrait = i
    return minTrait


def build(examples, maxDepth=None):  # builds used
    """examples is the data set. maxDepth is the maximum depth of the tree or None for unlimited depth."""
    used = [1] * (
        len(examples[0]) - 1
    )  # used[i]=1 means that attribute i hadn't been used
    return recBuild(examples, used, 0, maxDepth)


def recBuild(examples, used, parentMaj, maxDepth):
    """
    Builds the decision tree.
    parentMaj = majority class of the parent of this node. the heuristic is that if there is no decision returns parentMaj
    maxDepth = the maximum depth of the tree. shrinks by 1 every recoursion. can be None for unlimited depth.
    """
    if maxDepth == 0:
        return [[], parentMaj, []]
    cl = isSameClass(examples)
    if cl == 0 or cl == 1:  # all zeros or all ones
        return [[], cl, []]
    if cl == 7:  # examples is empty
        return [[], parentMaj, []]
    trait = minInfoTrait(examples, used)
    if trait == -1:  # there are no more attr. for splitting
        return [[], cl + 2, []]  # cl+2 - makes cl 0/1 (-2+2 / -1+2)
    x = split(examples, used, trait)
    # if maxDepth is None, then nextMaxDepth is also None for unlimited depth.
    nextMaxDepth = None
    if maxDepth is not None:
        nextMaxDepth = maxDepth - 1
    left = recBuild(x[0], used[:], cl + 2, nextMaxDepth)
    right = recBuild(x[1], used[:], cl + 2, nextMaxDepth)
    return [left, trait, right]


def recClassifier(dtree, traits):
    """dtree is the tree, traits is an example to be classified"""
    if dtree[0] == []:  # there is no left child, means arrive to a leaf
        return dtree[1]
    return recClassifier(
        dtree[traits[dtree[1]] * 2], traits
    )  # o points to the left child, 2 points to the right child


def classifier(dtree, traits):
    """same as the recClassifier, but without recursion"""
    while dtree[0] != []:
        dtree = dtree[traits[dtree[1]] * 2]
    return dtree[1]


def gray2bin(gray, threshold):
    """converts gray pixel to a bin pixel according to the threshold"""
    if gray < threshold:
        return 0
    else:
        return 1


def minst2bin(images, labels, threshold=130):
    """
    loads the dataset and labels and converts the images to binary
    :images: path to the images dataset
    :labels: path to the labels dataset
    :threshold: the threshold between black and white when converting the images to bin
    :return: dataset with binary images and the digitset (labels) with values between 0 to 9
    :rtype: Tuple[List[List[int]], List[int]]
    """

    dataset = []
    digitset = []
    with open(images, "rb") as fimages, open(labels, "rb") as flabels:
        # move to the data
        flabels.seek(8)
        fimages.seek(16)

        # each image from the dataset being converted to binary images,
        # and the labels are inserted to digitset.
        while (image := fimages.read(IMAGE_SIZE)) != b"":
            row = [gray2bin(px, threshold) for px in image]
            row += [0]  # Make space for the label (will stay 0 for now)
            dataset.append(row)
            digitset.append(ord(flabels.read(1)))
    return dataset, digitset


def buildclassifier(dataset, digitset, images, labels, maxDepth=None, threshold=130):
    """
    builds a digit classifier from the given dataset
    the classifier contains 10 decision trees
    each tree will learn to classify a single digit

    :dataset: the dataset of binary images
    :digitset: the labels of the images
    :images: the file name of the images' dataset (used for caching)
    :labels: the file name of the labels' dataset (used for caching)
    :maxDepth: the max depth of all the decision trees (used also for caching)
    :threshold: the threshold that the dataset was built with (used for caching)
    :return: the built model
    :rtype: List[List[List[...], int, List[...]]]
    """

    DIGITS = 10
    trees = [None] * DIGITS
    # load or generate a tree for each digit
    for i in range(DIGITS):
        start = now()
        cache = cachePath(images, labels, maxDepth, threshold, i)
        cached = loadCache(cache)
        if cached is not None:
            trees[i] = cached
            print("found", i, "in", now() - start, "seconds")
        else:
            # add the binary labels to the end of the dataset
            for row, digit in enumerate(digitset):
                dataset[row][IMAGE_SIZE] = 1 if i == digit else 0
            print("start", i)
            trees[i] = build(dataset, maxDepth)
            saveCache(trees[i], cache)
            print("generated", i, "in", now() - start, "seconds")
    return trees


def classify(trees, image):
    """
    using the given model to classify the digit of the image
    :return: -2: couldn't classify,
             -1: classified as multiple digits,
             0-9: the classified digit
    """

    digit = -2
    for i, tree in enumerate(trees):
        if classifier(tree, image):
            # checks if already classified
            if digit != -2:
                return -1
            digit = i
    return digit


def tester(trees, dataset, digitset):
    """
    tests the given model with the given dataset and digitset
    :return: the success rate of the model. value in range [0,1]
    :rtype: float
    """

    success = 0
    for i, image in enumerate(dataset):
        digit = classify(trees, image)
        if digit == digitset[i]:
            success += 1
    return success / len(dataset)


def threshold(trainImages, trainLabels, testImages, testLabels, maxDepth=None):
    """
    searches the threshold that results with the max success rate of the model
    :trainImages: the file name of the images dataset for the training
    :trainLabels: the file name of the labels dataset for the training
    :testImages: the file name of the images dataset for the testing
    :testLabels: the file name of the labels dataset for the testing
    :maxDepth: the max depth of the builted models
    :return: the threshold that gives the max score with the score
    :rtype: Tuple[int, float]
    """

    def calcScore(threshold, maxThresh, maxScore):
        """
        builds and tests the model with the given threshold and compares with the max score
        :return: the threshold that gave the max score with the score
        :rtype: Tuple[int, float]
        """

        trainDataset, trainDigitset = minst2bin(trainImages, trainLabels, threshold)
        testDataset, testDigitset = minst2bin(testImages, testLabels, threshold)
        trees = buildclassifier(
            trainDataset, trainDigitset, trainImages, trainLabels, maxDepth, threshold
        )
        score = tester(trees, testDataset, testDigitset)
        print("finished a model with", threshold, "threshold and", score, "score")

        if score > maxScore:
            maxScore = score
            maxThresh = threshold
        return maxThresh, maxScore

    # calculates the score from 128 to 0 and to 256 to find the threshold
    # that gave the maximum score
    maxThresh, maxScore = calcScore(128, 0, 0)
    print(maxThresh, maxScore)
    for i in range(1, 128):
        maxThresh, maxScore = calcScore(128 + i, maxThresh, maxScore)
        maxThresh, maxScore = calcScore(128 - i, maxThresh, maxScore)
    return maxThresh, maxScore


def cachePath(images, labels, maxDepth, threshold, digit):
    """creates a path to cache a classifier model that was built with the given paramters"""
    return f".cache/{images}.{labels}.{maxDepth}.{threshold}.{digit}"


def loadCache(filename):
    """loaded a classifier model from cache if exists"""
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.loads(f.read())


def saveCache(tree, filename):
    """saves the classifier model as cache"""
    with open(filename, "w") as f:
        f.write(json.dumps(tree))


print(
    threshold(
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        maxDepth=30,
    )
)
