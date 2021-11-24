import random
import time

FILENAME = "itemsets.txt"
N = 10  # no. of attributes
MINSUP = 0.4


# Creates a file named filename containing m sorted itemsets of items 0..N-1
def createfile(m, filename):
    f = open(filename, "w")
    for line in range(m):
        itemset = []
        for i in range(random.randrange(N) + 1):
            item = random.randrange(N)  # random integer 0..N-1
            if item not in itemset:
                itemset += [item]
        itemset.sort()
        for i in range(len(itemset)):
            f.write(str(itemset[i]) + " ")
        f.write("\n")
    f.close()


# Returns true iff all of smallitemset items are in bigitemset (the itemsets are sorted lists)
def is_in(smallitemset, bigitemset):
    s = b = 0  # s = index of smallitemset, b = index of bigitemset
    while s < len(smallitemset) and b < len(bigitemset):
        if smallitemset[s] > bigitemset[b]:
            b += 1
        elif smallitemset[s] < bigitemset[b]:
            return False
        else:
            s += 1
            b += 1
    return s == len(smallitemset)


# Returns a list of itemsets (from the list itemsets) that are frequent
# in the itemsets in filename with the support of each itemset at the end of each itemset
def frequent_itemsets(filename, itemsets):
    f = open(filename, "r")
    filelength = 0  # filelength is the no. of itemsets in the file. we
    # use it to calculate the support of an itemset
    count = [0] * len(itemsets)  # creates a list of counters
    line = f.readline()
    while line != "":
        filelength += 1
        line = line.split()  # splits line to separate strings
        for i in range(len(line)):
            line[i] = int(line[i])  # converts line to integers
        for i in range(len(itemsets)):
            if is_in(itemsets[i], line):
                count[i] += 1
        line = f.readline()
    f.close()
    freqitemsets = []
    for i in range(len(itemsets)):
        if count[i] >= MINSUP * filelength:
            # add the support to the end of the itemset
            itemsets[i].append(count[i] / filelength)
            freqitemsets += [itemsets[i]]
    return freqitemsets


# Checks that all k-1 subsets of k-itemset are frequent itemsets.
def all_kminus1_subsets_are_frequent(filename, kitemset):
    k = len(kitemset)
    kminus1_subsets = [None] * k
    for i in range(k):
        kminus1_subsets[i] = kitemset[:i] + kitemset[i + 1 :]
    return len(frequent_itemsets(filename, kminus1_subsets)) == k


def create_kplus1_itemsets(kitemsets, filename):
    kplus1_itemsets = []
    for i in range(len(kitemsets) - 2):
        j = i + 1  # j is an index
        # compares all pairs, without the last item of the itemset and the support (2 items)
        # note that the lists are sorted.
        # and if they are equal than adds the last item of kitemsets[j] to kitemsets[i]
        # in order to create k+1 itemset
        while j < len(kitemsets) and kitemsets[i][:-2] == kitemsets[j][:-2]:
            kplus1_itemsets += [kitemsets[i][:-1] + [kitemsets[j][-2]]]
            j += 1
    # checks which of the k+1 itemsets are frequent
    freqitemsets = frequent_itemsets(filename, kplus1_itemsets)
    return list(
        filter(
            lambda i: all_kminus1_subsets_are_frequent(filename, i[:-1]), freqitemsets
        )
    )


def create_1itemsets(filename):
    # it = list([i] for i in range(N))
    it = []
    for i in range(N):
        it += [[i]]
    return frequent_itemsets(filename, it)


def minsup_itemsets(filename):
    minsupsets = kitemsets = create_1itemsets(filename)
    while kitemsets != []:
        kitemsets = create_kplus1_itemsets(kitemsets, filename)
        minsupsets += kitemsets
    return minsupsets


def create_file_with_k_sized_itemset(k, m, filename, max_seconds=60):
    succeed = False
    start = time.time()
    tries = 0
    while not succeed:
        seconds_passed = time.time() - start
        if max_seconds != 0 and seconds_passed > max_seconds:
            break
        createfile(m, filename)
        itemsets = minsup_itemsets(filename)
        succeed = any(filter(lambda i: len(i) >= k, itemsets))
        tries += 1
    return succeed, tries


succeed, tries = create_file_with_k_sized_itemset(4, 10, FILENAME)
print(tries)
if succeed:
    print(minsup_itemsets(FILENAME))
else:
    print("Failed creating file")
