# Setup
from __future__ import division, print_function, unicode_literals

import pickle

import numpy as np
import os

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from scipy.ndimage import shift


# to make output stable across runs
def seed_rand(x=42):
    return np.random.seed(x)


# To plot pretty figures

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "/Users/shirelga/liat/dev/fabus"
CHAPTER_ID = "Classification"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, fig_id + ".png")

    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# MNIST
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def fetch_mnist_data():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
        sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist


mnist = fetch_mnist_data()

print("sorted MNIST dataset: \n", mnist["data"], mnist["target"])
print("\n")
print("(data, target) shape: \n", mnist.data.shape)
print("\n")

X, y = mnist["data"], mnist["target"]

print("X - data shape: \n", X.shape)
print("\n")

print("y - target key - labels shape: \n", y.shape)
print("\n")

print("each image has 28*28  pixels=", 28 * 28)
print("features\n\n")

some_digit = X[36000]

some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
#plt.show()

print("Label of some digit\n", y[36000])

# Split data to train and test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Binary classifier , simplify the problem
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# Train SGDClassifier and train it on whole training set

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)

print("is the model predicted right?True or False:\n", sgd_clf.predict([some_digit]))
print("\n")
# Prefomance measures - evaluating a classifier
# Measuring Accuracy using cross-Validation
cross_acc_5 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("Accuracy:\n", cross_acc_5)
print("\n")
# Measuring Accuracy using Confusion Matrix
# First we need set of predictions
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Passed target classes and the predicted classes to the CM
cm = confusion_matrix(y_train_5, y_train_pred)
print("Confusion matrix: \n", cm)
print("\n")

# Accuracy of th Positive predicted value
precision_ = precision_score(y_train_5, y_train_pred)
print("This is the precision:\n", precision_)
print("\n")

# TPR
recall_ = recall_score(y_train_5, y_train_pred)
print("This is the recall:\n", recall_)
print("\n")

# f1 score is
#  TP/(TP+((FN+FP)/2))
f_one_score = f1_score(y_train_5, y_train_pred)
print("This is the f1 score:\n", f_one_score)
print("\n")

# decision_function() returns a score for each instance
y_scores = sgd_clf.decision_function([some_digit])
print("Score for each instance:", y_scores)

# make predictions based on those decision_function() scores
# using any treshold
treshold = 0
y_some_digit_pred = (y_scores > treshold)
print("Predictions using zero treshold:", y_some_digit_pred)

# decide which threshold to use
# first we get scores of all instances in the training set using  cross validation predict
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print("Dimention of the y_score generated by cross_val_predict():\n", y_scores.shape
      )
print("\n")

if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]
print("Dimention with hack around issue 9589 in sklearn shape:\n", y_scores.shape)
print("\n")

# now with the scores we can compute precision and recall
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
#save_fig("precision_recall_vs_threshold_plot")
#plt.show()

y_train_pred_90 = (y_scores > 70000)
precision_score_90 = precision_score(y_train_5, y_train_pred_90)
print("Precision for 70000 threshold:", precision_score_90)

recall_score_90acc = recall_score(y_train_5, y_train_pred_90)
print("Recall for 70000 threshold:", recall_score_90acc)


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
#save_fig("precision_vs_recall_plot")
#plt.show()

# ROC curves
# computing FPR and TPR for varios threshold values
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# plot FPR against the TPR
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
#save_fig("roc_curve_plot")
#plt.show()

# A perfect classifier will have a ROC AUC equal to 1,whereas a purely classifier will have 0.5
roc_auc = roc_auc_score(y_train_5, y_scores)
print("This is roc curve auc for SGD classifier:", roc_auc)

# train RF
forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
#plt.show()

roc_auc_RFclassifier = roc_auc_score(y_train_5, y_scores_forest)
print("This is roc curve auc for Random Forest classifier:", roc_auc_RFclassifier)

# measuring the precision and recall, f1 also

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
accuracy_random_forest = precision_score(y_train_5, y_train_pred_forest)
print("Accuracy using Random Forest:\n", accuracy_random_forest)

recall_random_forest = recall_score(y_train_5, y_train_pred_forest)
print("Recall using Random Forest:\n", recall_random_forest)

f1 = f1_score(y_train_5, y_train_pred_forest)
print('Random forest f1 score:\n', f1)

# Multi class classification
# OVA
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))

some_digit_score = sgd_clf.decision_function([some_digit])
print(some_digit_score)
print("The higher score for OvA is indeed the one corresponding to class:\n", np.argmax(some_digit_score))

print("SGD classes:\n", sgd_clf.classes_)

print("The sgd_clf.classes_[5]", sgd_clf.classes_[5])

# OvO
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))
ovo_clf.fit(X_train, y_train)
prediction_ovo = ovo_clf.predict([some_digit])
print("OvO prediction:\n", prediction_ovo)

print("Num of training binary classifiers:", len(ovo_clf.estimators_))

# Training a Random Forest Classifier
forest_clf.fit(X_train, y_train)
random_forest_predicted = forest_clf.predict([some_digit])
print("random forest predicted:\n", random_forest_predicted)

# Getting the list of probalities that classifier assigned to each instance for each class
probabilities_for_each_class = forest_clf.predict_proba([some_digit])
print("probalities for each class:\n", probabilities_for_each_class)

# evaluate SGD
acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("SGD accuracy:\n", acc)

# Scaling the inputs to increas accuracy
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train.astype(np.float64))
scaled_score = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print("This is scaled SGD accuracy:\n", scaled_score)

# Error evaluating using confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mat = confusion_matrix(y_train, y_train_pred)
print("Confusion mat:\n", conf_mat)


# confusion_matrix_image representation
def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


plt.matshow(conf_mat, cmap=plt.cm.gray)
#save_fig("confusion_matrix_plot", tight_layout=False)
#plt.show()

row_sums = conf_mat.sum(axis=1, keepdims=True)
# deviding by number of images in the corresponding class
# to compare error rates instead of absolute number of errors
norm_conf_mx = conf_mat / row_sums

# fill diagonal with zeros to keep only the errors
np.fill_diagonal(norm_conf_mx, 0)
# rows - actual classes , columns - predicted classes
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
#plt.show()

# Multilabel classification
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

is_large_from7_is_odd = knn_clf.predict([some_digit])
print("(large from 7?,is odd?) knn predictions:\n", is_large_from7_is_odd)

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
# f1_score(y_multilabel, y_train_knn_pred, average="macro")

is_exist = os.path.exists("./knn_model.pickle") and os.path.exists("./grid_search.pickle")
if is_exist:
    with open("./knn_model.pickle", "rb") as model_file:
        knn_clf = pickle.load(model_file)
    with open("grid_search.pickle", "rb") as grid_search_file:
        grid_search = pickle.load(grid_search_file)
else:
    # 1 MNIST Classifier With Over 97% Accuracy
    param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    with open("./knn_model.pickle", "wb") as model_file:
        pickle.dump(knn_clf, model_file)
    with open("./grid_search.pickle", "wb") as grid_search_file:
        pickle.dump(grid_search, grid_search_file)

best_param = grid_search.best_params_
print("best knn parameters:", best_param)

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("this is KNN accuracy score:", accuracy)

#2
# Data augmentation
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])



X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)



shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)

knn_clf.fit(X_train_augmented, y_train_augmented)



y_pred = knn_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
print("Accuracy:\n", ac_score)



