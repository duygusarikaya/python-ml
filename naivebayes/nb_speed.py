#!/usr/bin/python

""" Complete the code in nb_classifier.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
"""

import sys
from nb_classifier import classify
sys.path.append("../tools/")
from prep_terrain_data import make_terrain_data
from class_vis import pretty_picture, output_image

features_train, labels_train, features_test, labels_test = make_terrain_data()

# the training data (features_train, labels_train) have both "fast" and "slow" points mixed
# in together--separate them so we can give them different colors in the scatterplot,
# and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf, accuracy = classify(features_train, labels_train, features_test, labels_test)
print accuracy

# draw the decision boundary with the text points overlaid
pretty_picture(clf, features_test, labels_test)
output_image("dt_speed.png", "png", open("dt_speed.png", "rb").read())



