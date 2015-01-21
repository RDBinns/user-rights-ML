# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer,
                                             HashingVectorizer)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import (RidgeClassifier, Perceptron,
                                  PassiveAggressiveClassifier, SGDClassifier)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics

from nltk import sent_tokenize, word_tokenize, FreqDist, WordNetLemmatizer
from nltk.corpus import stopwords

import sys
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
    print "usage: {} [FILE.csv]".format(sys.argv[0])
    sys.exit(0)

f = sys.argv[1]
csv = pd.read_csv(f, header=None, encoding='utf-8')

print np.sum(csv[2].apply(pd.value_counts))


def featurize(input):
    input = input.replace(u"\u2019", "'")
    input = input.replace("'", "")
    input = input.replace("http://", "")
    input = input.replace("https://", "")

    input = input.replace(u"\xa7", '')
    input = input.replace('`', '')
    input = input.replace("-", '')
    input = input.replace("...", '')

    tokens = [word for sent in sent_tokenize(input)
              for word in word_tokenize(sent)]

    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    tokens = [word.lower() for word in tokens]

    tokens = [word for word in tokens if len(word) >= 2]

    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]

    return tokens

classifiers = [
    SGDClassifier(alpha=.0001, n_iter=50, penalty='l2'),
    # RidgeClassifier(tol=1e-2, solver="lsqr"),
    # Perceptron(n_iter=50),
    # PassiveAggressiveClassifier(n_iter=50, loss='squared_hinge'),
    BernoulliNB(alpha=0.01),
    MultinomialNB(alpha=0.01),
    LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3),
]

vectorizers = [
    CountVectorizer(ngram_range=(1, 2),
                    tokenizer=featurize),
]

splitter = ShuffleSplit(csv.shape[0], n_iter=10, test_size=0.1)
for v in vectorizers:
    print "> Running tests with {}".format(v)
    for classifier in classifiers:
        print "  > Using classifier {}".format(classifier)
        accuracies = []
        f1s = []

        for train, test in splitter:
            pipeline = Pipeline([('vect', v),
                                 ('tfidf', TfidfTransformer(sublinear_tf=True,
                                                            use_idf=False)),
                                 ('clf', OneVsOneClassifier(classifier))])

            pipeline.fit(np.asarray(csv[1][train]),
                         np.asarray(csv[2][train]))

            y_test = csv[2][test]
            X_test = csv[1][test]
            accuracies.append(pipeline.score(X_test, y_test))
            f1s.append(sklearn.metrics.f1_score(pipeline.predict(X_test),
                                                y_test))

        accuracies = np.array(accuracies)
        f1s = np.array(f1s)
        print '    > Accuracy: {} ({})'.format(accuracies.mean(),
                                               accuracies.std()*2)
        # print accuracies
        print '    > F1 scores {} ({})'.format(f1s.mean(), f1s.std()*2)
        # print f1s
        # print pipeline.steps[0][1].get_feature_names()
