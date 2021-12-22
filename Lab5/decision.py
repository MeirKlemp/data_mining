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


def classifier(dtree, traits):  # same as the former without recursion
    while dtree[0] != []:
        dtree = dtree[traits[dtree[1]] * 2]
    return dtree[1]


def gray2Bin(gray, threshold):
    if gray < threshold:
        return 0
    else:
        return 1


def minst2bin(images, labels, threshold=130):
    dataset = []
    digitset = []
    with open(images, "rb") as fimages, open(labels, "rb") as flabels:
        flabels.seek(8)
        fimages.seek(16)
        x = fimages.read(1)
        c = 0
        while x != b"":
            row = [0] * (IMAGE_SIZE + 1)  # image data + digit(stays 0 for now)
            row[0] = gray2Bin(ord(x), threshold)
            for i in range(1, IMAGE_SIZE):
                row[i] = gray2Bin(ord(fimages.read(1)), threshold)
            dataset.append(row)
            digitset.append(ord(flabels.read(1)))
            x = fimages.read(1)
    return dataset, digitset


def buildclassifier(dataset, digitset, images, labels, maxDepth=None, threshold=130):
    DIGITS = 10
    trees = [None] * DIGITS
    for i in range(DIGITS):
        start = now()
        cached = loadCache(images, labels, maxDepth, threshold, i)
        if cached is not None:
            trees[i] = cached
            print("found", i, "in", now() - start, "seconds")
        else:
            for row, digit in enumerate(digitset):
                dataset[row][IMAGE_SIZE] = 1 if i == digit else 0
            print("start", i)
            trees[i] = build(dataset, maxDepth)
            saveCache(trees[i], images, labels, maxDepth, threshold, i)
            print("generated", i, "in", now() - start, "seconds")
    return trees


def classify(trees, image):
    digit = -2
    for i, tree in enumerate(trees):
        if classifier(tree, image):
            if digit != -2:
                return -1
            digit = i
    return digit


def tester(trees, dataset, digitset):
    success = 0
    for i, image in enumerate(dataset):
        digit = classify(trees, image)
        # print(digitset[i], digit)
        if digit == digitset[i]:
            success += 1
    return success / len(dataset)


def threshold(trainImages, trainLabels, testImages, testLabels, maxDepth=None):
    maxThresh = 128
    trainDataset, trainDigitset = minst2bin(trainImages, trainLabels, maxThresh)
    testDataset, testDigitset = minst2bin(testImages, testLabels, maxThresh)
    trees = buildclassifier(
        trainDataset, trainDigitset, trainImages, trainLabels, maxDepth, maxThresh
    )
    maxScore = tester(trees, testDataset, testDigitset)
    print(maxThresh, maxScore)

    for i in range(1, 128):
        tu = 128 + i
        trainDataset, trainDigitset = minst2bin(trainImages, trainLabels, tu)
        testDataset, testDigitset = minst2bin(testImages, testLabels, tu)
        trees = buildclassifier(
            trainDataset, trainDigitset, trainImages, trainLabels, maxDepth, tu
        )
        score = tester(trees, testDataset, testDigitset)
        if score > maxScore:
            maxThresh = tu
            maxScore = score
        print(tu, score)

        td = 128 - i
        trainDataset, trainDigitset = minst2bin(trainImages, trainLabels, td)
        testDataset, testDigitset = minst2bin(testImages, testLabels, td)
        trees = buildclassifier(
            trainDataset, trainDigitset, trainImages, trainLabels, maxDepth, td
        )
        score = tester(trees, testDataset, testDigitset)
        if score > maxScore:
            maxThresh = td
            maxScore = score
        print(td, score)
    return maxThresh, maxScore


def cachePath(images, labels, maxDepth, threshold, digit):
    return f".cache/{images}.{labels}.{maxDepth}.{threshold}.{digit}"


def loadCache(images, labels, maxDepth, threshold, digit):
    filename = cachePath(images, labels, maxDepth, threshold, digit)
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.loads(f.read())


def saveCache(tree, images, labels, maxDepth, threshold, digit):
    with open(cachePath(images, labels, maxDepth, threshold, digit), "w") as f:
        f.write(json.dumps(tree))


# print(
#     threshold(
#         "dig-train-images",
#         "dig-train-labels",
#         "dig-test-images",
#         "dig-test-labels",
#         maxDepth=30,
#     )
# )

thresh = 141
trainDataset, trainDigitset = minst2bin(
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte", thresh
)
testDataset, testDigitset = minst2bin(
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", thresh
)
trees = buildclassifier(
    trainDataset,
    trainDigitset,
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    10,
    thresh,
)
maxScore = tester(trees, testDataset, testDigitset)
print(thresh, maxScore)

# print(
#     threshold(
#         "train-images-idx3-ubyte",
#         "train-labels-idx1-ubyte",
#         "t10k-images-idx3-ubyte",
#         "t10k-labels-idx1-ubyte",
#         maxDepth=10,
#     )
# )

# e = [
#     [1, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1],
#     [1, 1, 1, 0, 0],
#     [1, 1, 0, 1, 0],
#     [0, 0, 1, 1, 1],
#     [1, 0, 1, 1, 0],
#     [1, 0, 0, 1, 1],
# ]

# t = build(e)
# print(classifier(t, [0, 1, 1, 1]))
