from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP import MDLP_Discretizer
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp

# NB example based on: https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
# DT example based on: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

def main():
    
    # read dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 123)

    # print(X)

    #Initialize discretizer object and fit to training data
    # discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer = MDLP_Discretizer_fxp(features=numeric_features)
    discretizer.fit(X_train, y_train)
    
    X_train_discretized = discretizer.transform(X_train)

    #apply same discretization to test set
    X_test_discretized = discretizer.transform(X_test)

    # print(type(X))
    # print(X.shape)
    # print(X_train_discretized.shape)

    # print(feature_names)
    # print(class_names)

    # print ("Original dataset:\n%s" % str(X[0:10]))
    # print ("Discretized dataset:\n%s" % str(X_train_discretized[0:10]))

    #see how feature 0 was discretized
    print("CUTPOINTS:")
    print("Interval cut-points: %s" % str(discretizer._cuts[0]))
    print("Interval cut-points: %s" % str(discretizer._cuts[1]))
    print("Interval cut-points: %s" % str(discretizer._cuts[2]))
    print("Interval cut-points: %s" % str(discretizer._cuts[3]))

    # Dump resulted dataset to file
    # np.savetxt('files/X_discretized.var', X_discretized)

    #####################
    ### NAIVE BAYES ###
    #####################

    # Create a Gaussian Classifier
    gnb = GaussianNB()
    gnb_disc = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    gnb_disc.fit(X_train_discretized, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    y_pred_disc = gnb_disc.predict(X_test_discretized)
    
    # Model Accuracy, how often is the classifier correct?
    print()
    print("NAIVE BAYES:")
    print("Accuracy (original):",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy when discretized:",metrics.accuracy_score(y_test, y_pred_disc))

    #####################
    ### DECISION TREE ###
    #####################

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    clf_disc = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    clf_disc = clf_disc.fit(X_train_discretized, y_train)

    y_pred = clf.predict(X_test)
    y_pred_disc = clf_disc.predict(X_test_discretized)

    # Model Accuracy, how often is the classifier correct?
    print()
    print("DECISION TREE::")
    print("Accuracy (original):",metrics.accuracy_score(y_test, y_pred))
    print("Accuracy when discretized:", metrics.accuracy_score(y_test, y_pred_disc))

    print()
    exit(0)

if __name__ == '__main__':
    main()

