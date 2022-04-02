import math
import pickle
import pandas as pd


class DecisionTree:
    __slots__ = ['value', 'left', 'right']

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def trueFalse(sentence):
    s = []
    if ' ik ' in sentence or ' meest ' in sentence or ' het ' in sentence or ' niet ' in sentence or ' ons ' in \
            sentence or ' deze ' in sentence or ' hun ' in sentence or ' zo ' in sentence or ' weten ' in sentence \
            or ' naar ' in sentence or ' onze ' in sentence or ' ze ' in sentence or ' hem ' in sentence or ' met ' in \
            sentence or ' ook ' in sentence or ' jouw ' in sentence or ' dan ' in sentence or ' ik ' in sentence or \
            ' er ' in sentence or ' ze ' in sentence or ' voor ' in sentence or ' ben ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' for ' not in sentence or ' his ' not in sentence or ' him ' not in sentence or ' with ' not in sentence or \
            'mine ' not in sentence or ' will ' not in sentence or ' not ' not in sentence or ' no ' not in sentence \
            or ' we ' not in sentence or ' them ' not in sentence or ' as ' not in sentence or ' her ' not in \
            sentence or ' here ' not in sentence or ' our ' not in sentence or ' us ' not in sentence or ' they ' \
            not in sentence or ' so ' not in sentence or ' to ' not in sentence or ' it ' not in sentence \
            or ' their ' not in sentence or ' about ' not in sentence or ' she ' not in sentence or ' there ' not \
            in sentence or ' i ' not in sentence or ' me ' not in sentence or ' he ' not \
            in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'aa' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'ij' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'oo' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'ee' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'q' in sentence:
        s.append(False)
    else:
        s.append(True)
    if ' de ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if 'en' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' een ' in sentence:
        s.append(True)
    else:
        s.append(False)
    if ' van ' in sentence:
        s.append(True)
    else:
        s.append(False)
    return s


def parentIndices(parent_examples):
    pe = list(parent_examples.index.values)
    print(pe)
    return max(set(pe), key=pe.count)


def classif(examples):
    same = False
    classification = 'en'
    countEng = list(examples.index.values).count('en')
    countDutch = list(examples.index.values).count('nl')
    if countEng == 0:
        same = True
    if countDutch == 0:
        same = True
    if countDutch > countEng:
        classification = 'nl'
    return same, classification


def entropy(x, y):
    if (x + y) > 0:
        px = x / (x + y)
        py = y / (x + y)
    else:
        return 0
    if px == 0 or px == 1 or py == 0 or py == 1:
        return 0
    return -(px * math.log(px, 2)) - (py * math.log(py, 2))


def bestAttrib(attributes, examples):
    listInd = list(examples.index.values)
    engCount = listInd.count('en')
    dutchCount = listInd.count('nl')
    N = examples.shape[0]
    incr = 0
    A = ''
    B = entropy(dutchCount, engCount)
    for a in attributes:
        trueCountA = 0
        falseCountA = 0
        trueCountDutch = 0
        falseCountDutch = 0
        trueCountEnglish = 0
        falseCountEnglish = 0
        for row in range(N - 1):
            val = examples[a][row]
            index = listInd[row]
            if val:
                trueCountA += 1
                if index == 'nl':
                    trueCountDutch += 1
                else:
                    trueCountEnglish += 1
            if not val:
                falseCountA += 1
                if index == 'nl':
                    falseCountDutch += 1
                else:
                    falseCountEnglish += 1
        rem = ((trueCountA / N) * entropy(trueCountDutch, trueCountEnglish)) + (
                (falseCountA / N) * entropy(falseCountDutch, falseCountEnglish))
        gainA = B - rem
        if gainA > incr:
            incr = gainA
            A = a
    return A


def dtree(examples, attributes, parentEx):
    if examples.empty:
        return DecisionTree(parentIndices(parentEx))
    same, classification = classif(examples)
    # print(same, "\n", classification)
    if same:
        return DecisionTree(classification)
    if not attributes:
        return DecisionTree(parentIndices(examples))
    A = bestAttrib(attributes, examples)
    leftFal = examples.loc[examples[A] == False]
    righttrue = examples.loc[examples[A] == True]
    attributes.remove(A)
    leftTree = dtree(leftFal, attributes, examples)
    rightTree = dtree(righttrue, attributes, examples)
    return DecisionTree(A, leftTree, rightTree)


def adaBbuild(examples, attributes):
    K = 10
    N = examples.shape[0]
    w = []
    for i in range(N):
        w.append(1 / N)
    ll = list(examples.index.values)
    op = []
    for l in ll:
        if l == 'nl':
            op.append(True)
        else:
            op.append(False)
    hypothesis = []
    for k in range(K):
        stump = []
        errorMin = math.inf
        y = []
        a = ''
        for l in range(len(attributes)):
            error = 0
            cor = []
            for j in range(N):
                if examples.values[j][l] is op[j]:
                    cor.append(j)
                else:
                    error += w[j]
                if error < errorMin:
                    errorMin = error
                    y = cor
                    a = attributes[l]
        for r in y:
            w[r] *= errorMin / (1 - errorMin + 0.0000001)
        w = normWts(w)
        z = math.log((1 - errorMin) / (error + 0.0000001), 2)
        stump.append(a)
        stump.append(z)
        hypothesis.append(stump)
        attributes.remove(a)
        attributes.append(a)
    return hypothesis


def normWts(weights):
    total_weight = sum(weights)
    w = []
    for weight in weights:
        w.append(weight / total_weight)
    return w


def train(examples, hypothesisOut, adaORdt):
    splChar = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word', 'no_en_word', 'aa', 'ij', 'oo', 'ee', 'q', ' de ', 'en', ' een ', ' van ']
    dutchengl = []
    trueFal = []
    train_dat = open(examples, encoding="utf8")
    for line in train_dat:
        # print(line)
        dutchengl.append(line[:2])
        sentence = line[3:].lower()
        for p in splChar:
            if p in sentence:
                sentence = sentence.replace(p, '')
        trueFal.append(trueFalse(sentence))
    data = pd.DataFrame(trueFal, columns=attributes, index=dutchengl)
    # print(data)
    if adaORdt == 'dt':
        tree = dtree(data, attributes, data)
    elif adaORdt == 'ada':
        tree = adaBbuild(data, attributes)
    pickle.dump(tree, open(hypothesisOut, 'wb'))


def decTree(data, h, r):
    if h.left is None or h.right is None:
        return h.value
    elif not data[h.value][r]:
        return decTree(data, h.left, r)
    else:
        return decTree(data, h.right, r)


def predDTree(h, file):
    op = []
    testFile = open(file, 'r')
    splChar = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word','no_en_word','aa', 'ij', 'oo', 'ee', 'q', ' de ', 'en', ' een ', ' van ']
    dutchEn = []
    truefVals = []
    for l in testFile:
        dutchEn.append(l[:2])
        sentence = l[3:].lower()
        for punc in splChar:
            if punc in sentence:
                sentence = sentence.replace(punc, '')
        truefVals.append(trueFalse(sentence))
    data = pd.DataFrame(truefVals, columns=attributes)
    for r in range(data.shape[0]):
        t = decTree(data, h, r)
        op.append(t)
    print("Decision tree prediction: " + str(op))


def predAdaB(h, file):
    op = []
    testFile = open(file, 'r')
    splChar = "!()-[]{};:\,\”\“<>.?&"
    attributes = ['nl_word','no_en_word','aa', 'ij', 'oo', 'ee', 'q', ' de ', 'en', ' een ', ' van ']
    dutchEng = []
    trueFal = []
    for l in testFile:
        dutchEng.append(l[:2])
        sentence = l[3:].lower()
        for punc in splChar:
            if punc in sentence:
                sentence = sentence.replace(punc, '')
        trueFal.append(trueFalse(sentence))
    data = pd.DataFrame(trueFal, columns=attributes)
    for r in range(data.shape[0]):
        tot = 0
        for hypo in h:
            if data[hypo[0]][r]:
                tot += hypo[1] * 1
            else:
                tot += hypo[1] * -1
        if tot > 0:
            op.append('en')
        else:
            op.append('nl')
    print("Adaboost prediction: " + str(op))


def main():
    trainPred = input("enter 'train' or 'predict'\n")
    # trainPred = "predict"
    if trainPred == 'train':
        examples = input("enter the training file path\n")
        # examples = "train.dat"
        hypothesisOut = input("enter hypothesis file path\n")
        # hypothesisOut = "xcv"
        adaOrdt = input("enter 'dt' for 'decision tree' or 'ada' for 'adaboost'\n")
        # adaOrdt = "dt"

        splchar = "!()-[]{};:\,\”\“<>.?&"
        attributes = ['nl_word', 'no_en_word', 'aa', 'ij', 'oo', 'ee', 'q', ' de ', 'en', ' een ', ' van ']
        dutchenglish = []
        yesNo = []
        trainFile = open(examples, encoding="utf8")
        for l in trainFile:
            # print(l)
            dutchenglish.append(l[:2])
            sentence = l[3:].lower()
            for p in splchar:
                if p in sentence:
                    sentence = sentence.replace(p, '')
            yesNo.append(trueFalse(sentence))
        data = pd.DataFrame(yesNo, columns=attributes, index=dutchenglish)
        # print(data)
        if adaOrdt == 'dt':
            currentDtree = dtree(data, attributes, data)
        elif adaOrdt == 'ada':
            currentDtree = adaBbuild(data, attributes)
        pickle.dump(currentDtree, open(hypothesisOut, 'wb'))
        print("completed")
    elif trainPred == 'predict':
        hypothesis = input("enter the hypothesis file path\n")
        # hypothesis = "xcv"
        file = input("enter testing file path\n")
        # file = "test.dat"

        h = pickle.load(open(hypothesis, 'rb'))
        if isinstance(h, DecisionTree):
            predDTree(h, file)
        else:
            predAdaB(h, file)
        print("completed!")


if __name__ == '__main__':
    main()
