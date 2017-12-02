import collections,itertools,math
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
from nltk.corpus import LazyCorpusLoader, CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from MaxVoteClassifier import MaxVoteClassifier

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stopset = set(stopwords.words('english'))

app_reviews = LazyCorpusLoader('app_reviews', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*', encoding='ascii')

def evaluate_classifier(featureX):

    negIds = app_reviews.fileids('neg')
    posIds = app_reviews.fileids('pos')

    posFeatures = [(featureX(app_reviews.words(fileids=[f])), 'neg') for f in negIds]
    negFeatures = [(featureX(app_reviews.words(fileids=[f])), 'pos') for f in posIds] 

    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
 
    trainFeatures = negFeatures[:negCutoff] + posFeatures[:posCutoff]
    testFeatures  = negFeatures[negCutoff:] + posFeatures[posCutoff:]

    #trains a Naive Bayes Classifier
    NBclassifier = NaiveBayesClassifier.train(trainFeatures)
    #trains a Maximum Entropy or Logistic Regression Classifier
    MEclassifier = MaxentClassifier.train(trainFeatures,algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
    #trains a DecisionTree Classifier
    DTclassifier = DecisionTreeClassifier.train(trainFeatures,binary=True, entropy_cutoff=0.5, depth_cutoff=70, support_cutoff=10)

    #Combining Classifiers with Voting
    classifier = MaxVoteClassifier(NBclassifier, MEclassifier, DTclassifier)


    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets      = collections.defaultdict(set)

    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            observed = classifier.classify(features)
            testSets[observed].add(i)

    #prints metrics to show how well the feature selection
    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print 'Accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos Precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos Recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg Precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg Recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])

print '****Max Vote Calssifier****\n'

#creates a feature selection mechanism that uses all words
def word_feats(words):
    return dict([(lemmatizer.lemmatize(word), True) for word in words if word not in stopset and not(word.isnumeric()) and word.isalpha()])

#tries using all words as the feature selection mechanism
print 'Evaluating Unigram Words as features.'
evaluate_classifier(word_feats)

def unigram_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopset and not(word.isnumeric()) and word.isalpha()]
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

print '\nEvaluating Unigram + Bigram Word features.'
evaluate_classifier(unigram_bigram_word_feats)

#scores words based on chi-squared test to show information gain
def create_word_scores():
    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
 
    for word in app_reviews.words(categories=['pos']):
        if word not in stopset and not(word.isnumeric()) and word.isalpha():
            word_fd[lemmatizer.lemmatize(word)] += 1
            label_word_fd['pos'][lemmatizer.lemmatize(word)] += 1
 
    for word in app_reviews.words(categories=['neg']):
        if word not in stopset and not(word.isnumeric()) and word.isalpha():
            word_fd[lemmatizer.lemmatize(word)] += 1
            label_word_fd['neg'][lemmatizer.lemmatize(word)] += 1
 
    # n_ii = label_word_fd[label][word]
    # n_ix = word_fd[word]
    # n_xi = label_word_fd[label].N()
    # n_xx = label_word_fd.N()

    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    #builds dictionary of word scores based on chi-squared test
    word_scores = {}
 
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

#finds word scores
word_scores = create_word_scores()

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_feats(words):
    return dict([(word, True) for word in words if word in best_words])

print '\nEvaluating high informative word features'

#numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
	print ('\nEvaluating best %d word features' % (num))
	best_words = find_best_words(word_scores, num)
	evaluate_classifier(best_word_feats)


def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    return d
 
print '\nEvaluating best 200 Bigram word features'
evaluate_classifier(best_bigram_word_feats)

