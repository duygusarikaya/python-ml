#!/usr/bin/python

""" code for decision tree """

import sys
from dt_classifier import classify
sys.path.append("../tools/")
from prep_terrain_data import make_terrain_data
from class_vis import pretty_picture, output_image


features_train, labels_train, features_test, labels_test = make_terrain_data()

clf, accuracy = classify(features_train, labels_train, features_test, labels_test)


pretty_picture(clf, features_test, labels_test, "dt_speed.png")
output_image("dt_speed.png", "png", open("dt_speed.png", "rb").read())
