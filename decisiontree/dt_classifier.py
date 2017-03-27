from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score


def classify(features_train, labels_train, features_test, labels_test):

    # clf = tree.DecisionTreeClassifier()  # min_samples_split=2 by default
    clf = tree.DecisionTreeClassifier(min_samples_split=50)

    t0 = time()
    clf.fit(features_train, labels_train)
    print "Training time:", round(time() - t0, 3), "s"

    t0 = time()
    pred = clf.predict(features_test)
    print "Prediction time:", round(time() - t0, 3), "s"

    # calculate and return the accuracy on the test data

    # Calculate Accuracy Rate manually
    count = len(["ok" for idx, label in enumerate(labels_test) if label == pred[idx]])
    print "Accuracy Rate, calculated manually : %f" % (float(count) / len(labels_test))

    # Calculate Accuracy Rate by using accuracy_score()
    accuracy = accuracy_score(pred, labels_test)
    print "Accuracy Rate, calculated using sklearn : %f" % accuracy
    return clf, accuracy
