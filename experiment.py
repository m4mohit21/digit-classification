"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import *

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()
# print the height , width
print(digits.images[0].shapes)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.





# Data preprocessing
def split_train_dev_test(X,y,test_size,dev_size):
    _ = test_size + dev_size
    X_train, _xtest, y_train, _ytest = train_test_split(
    X, y, test_size=_, shuffle=False)
    X_test, X_dev, y_test, y_dev = train_test_split(
    _xtest, _ytest, test_size=dev_size, shuffle=False)
    return X_train, X_test, X_dev , y_train, y_test, y_dev
    
    

# Predict the value of the digit on the test subset
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    ###############################################################################
    # If the results from evaluating a classifier are stored in the form of a
    # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    # as follows:


    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X = data
y  = digits.target
# No. of samples in data
print(len(X))
gamma = [0.001,0.01,0.1,1,10,100]
C = [0.1,1,2,5,10]
param_groups = {
    "gamma" : gamma,
    "C" : C,
}
param_groups= get_hyperparameter_combinations(param_groups)

 
# param_groups = [{"gamma":i, "C":j} for i in gamma for j in C] 
# Create Train_test_dev size groups
test_sizes = [0.1, 0.2, 0.3] 
dev_sizes  = [0.1, 0.2, 0.3]
test_dev_size_combintion = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 

# Create a classifier: a support vector classifier
model = svm.SVC
for test_dev_size in test_dev_size_combintion:
    X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,**test_dev_size)
    train_acc, dev_acc, test_acc, optimal_param = tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,param_groups)
    _ = 1 - (sum(test_dev_size.values()))
    print(f'train_size: {_}, dev_size: {test_dev_size["dev_size"]}, test_size: {test_dev_size["test_size"]} , train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, optimal_param: {optimal_param}')