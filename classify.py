from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer,
                                             HashingVectorizer)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import (RidgeClassifier, Perceptron,
                                  PassiveAggressiveClassifier, SGDClassifier)

import sklearn.metrics

import sys
import pandas as pd
import numpy as np

if len(sys.argv < 2):
    print "usage: {} [FILE.csv]"
    sys.exit(0)

f = sys.argv[1]
csv = pd.read_csv(f, header=None)

classifiers = [
    SGDClassifier(alpha=.0001, n_iter=50, penalty='l2'),
    RidgeClassifier(tol=1e-2, solver="lsqr"),
    Perceptron(n_iter=50),
    PassiveAggressiveClassifier(n_iter=50, loss='squared_hinge', C=0.8),
    BernoulliNB(alpha=0.01),
    MultinomialNB(alpha=0.01),
    LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3),
]

vectorizers = [
    HashingVectorizer(n_features=10, non_negative=True),
    TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english'),
    CountVectorizer(ngram_range=(1, 2),
                    token_pattern=ur"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
                    min_df=1),
]

splitter = ShuffleSplit(csv.shape[0], n_iter=20, test_size=0.2)
for v in vectorizers:
    print "> Running tests with {}".format(v)
    for classifier in classifiers:
        print "  > Using classifier {}".format(classifier)
        accuracies = []
        f1s = []

        for train, test in splitter:
            vectorizer = v
            X_train_tfidf = vectorizer.fit_transform(csv[1][train])

            clf = classifier

            clf.fit(X_train_tfidf, csv[2][train])

            y_test = csv[2][test].as_matrix()
            X_new_tfidf = vectorizer.transform(csv[1][test])
            accuracies.append(clf.score(X_new_tfidf, y_test))
            f1s.append(sklearn.metrics.f1_score(clf.predict(X_new_tfidf),
                                                y_test))

        accuracies = np.array(accuracies)
        f1s = np.array(f1s)
        print '    > Accuracy: {} ({})'.format(accuracies.mean(),
                                               accuracies.std()*2)
        print '    > F1 scires {} ({})'.format(f1s.mean(), f1s.std()*2)
