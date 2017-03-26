import sys
from svm_classifier import classify
sys.path.append("../tools/")
from prep_terrain_data import make_terrain_data
from class_vis import pretty_picture, output_image


features_train, labels_train, features_test, labels_test = make_terrain_data()

clf, accuracy = classify(features_train, labels_train, features_test, labels_test)

# draw the decision boundary with the text points overlaid
pretty_picture(clf, features_test, labels_test)
output_image("svm_speed.png", "png", open("svm_speed.png", "rb").read())
