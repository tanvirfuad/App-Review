import itertools
from nltk.classify import ClassifierI
from nltk.probability import FreqDist

class MaxVoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self._labels = sorted(set(itertools.chain(*[c.labels() for c in classifiers])))
    def labels(self):
        return self._labels
    def classify(self, feats):
        counts = FreqDist()
        for classifier in self._classifiers:
            counts[classifier.classify(feats)] += 1
        return counts.max()
