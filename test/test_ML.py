import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import time 
import pickle
from sklearn.tree import DecisionTreeClassifier
import pymrmr
import pandas as pd


import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp
from MDLP.MDLP import MDLP_Discretizer

import load_dataset as ld

### Configuration ###
test_partition = 0.3


def scale_data(training_data, test_data):
    
    print("Scaling...")
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_data)

    test_data_scaled = scaler.fit_transform(test_data)

    return training_data_scaled, test_data_scaled

def test_NB(data, labels, datasetname, discretizer_filename):
    
    with open(discretizer_filename, 'rb') as handle:
        discretizer = pickle.load(handle)

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_partition, random_state = 123)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Apply discretization 
    X_train_discretized = discretizer.transform(X_train_scaled)
    X_test_discretized = discretizer.transform(X_test_scaled)

    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train_discretized, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test_discretized)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def test_DT(data, labels, datasetname, discretizer_filename):
    
    with open(discretizer_filename, 'rb') as handle:
        discretizer = pickle.load(handle)

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_partition, random_state = 123)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Apply discretization 
    X_train_discretized = discretizer.transform(X_train_scaled)
    X_test_discretized = discretizer.transform(X_test_scaled)

    # Create a Gaussian Classifier
    clf = DecisionTreeClassifier()

    # Train the model using the training sets
    clf.fit(X_train_discretized, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test_discretized)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def test_pymRMR(data, labels, datasetname, discretizer_filename):

    with open(discretizer_filename, 'rb') as handle:
        discretizer = pickle.load(handle)

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_partition, random_state = 123)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Apply discretization 
    print("transform...")
    X_train_discretized = discretizer.transform(X_train_scaled)

    print("pymRMR")
    # Convert data to panda dataFrame
    panda_df = pd.DataFrame(X_train_discretized.get_val())
    # panda_df = pd.DataFrame(X_train_discretized)
    panda_df.columns = panda_df.columns.astype(str)

    pymrmr.mRMR(panda_df, 'MIQ', 10)
    
    return 

def main():

    # PymRMR
#     dataset_filename = "datasets/bc-wisc-diag.mat"
#     data, labels = ld.load_dataset(dataset_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_original.bin'
#    # test_pymRMR(data, labels, dataset_filename, discretizer_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w32_f16.bin'
#     test_pymRMR(data, labels, dataset_filename, discretizer_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w16_f8.bin'
#     test_pymRMR(data, labels, dataset_filename, discretizer_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w12_f8.bin'
#     test_pymRMR(data, labels, dataset_filename, discretizer_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w8_f4.bin'
#     test_pymRMR(data, labels, dataset_filename, discretizer_filename)

#     discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w4_f2.bin'
#     test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    dataset_filename = "datasets/madelon.mat"
    data, labels = ld.load_dataset(dataset_filename)

    # discretizer_filename = 'output_files/saved/madelon.mat_discretizer_original.bin'
    # test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    # discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w16_f8.bin'
    # test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    # discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w12_f8.bin'
    # test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    # discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w8_f4.bin'
    # test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w4_f2.bin'
    test_pymRMR(data, labels, dataset_filename, discretizer_filename)

    exit(0)
    ## DT
    dataset_filename = "datasets/bc-wisc-diag.mat"
    data, labels = ld.load_dataset(dataset_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_original.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w32_f16.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w16_f8.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w12_f8.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w8_f4.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/bc-wisc-diag.mat_discretizer_w4_f2.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)
        
    dataset_filename = "datasets/madelon.mat"
    data, labels = ld.load_dataset(dataset_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_original.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w16_f8.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w12_f8.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w8_f4.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

    discretizer_filename = 'output_files/saved/madelon.mat_discretizer_w4_f2.bin'
    test_DT(data, labels, dataset_filename, discretizer_filename)

  
    

if __name__ == '__main__':
    main()