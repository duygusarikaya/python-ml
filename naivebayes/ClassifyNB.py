#!/usr/bin/python

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def classify(features_train, labels_train, features_test, labels_test):

    # create classifier
    # fit the classifier on the training features and labels
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    # use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    # calculate and return the accuracy on the test data

    # Calculate Accuracy Rate manually
    count = len(["ok" for idx, label in enumerate(labels_test) if label == pred[idx]])
    print "Accuracy Rate, calculated manually : %f" % (float(count) / len(labels_test))

    # Calculate Accuracy Rate by using accuracy_score()
    accuracy = accuracy_score(pred, labels_test)
    print "Accuracy Rate, calculated using sklearn : %f" % accuracy
    return clf, accuracy
