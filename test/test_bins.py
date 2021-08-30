import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp
from MDLP.MDLP import MDLP_Discretizer

import load_dataset as ld

### Configuration ###

fixedformat_pairs = [(1,2), (2,3), (4,5)]

def main():
    
    data, labels = ld.load_dataset('datasets/bc-wisc-diag.mat')
    print("Data shape: ", data.shape)
    # print(labels)

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, random_state = 123)
    
    print("Data shape: ", X_train.shape)

    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    n_features = np.arange(data.shape[1]) 

    print("Fiting discretizer...")
    # discretizer = MDLP_Discretizer_fxp(features=n_features, n_word=64, n_frac=32)
    discretizer = MDLP_Discretizer(features=n_features)
    
    discretizer.fit(X_train_scaled, y_train)
    print ("Interval cut-points: %s" % str(discretizer._cuts))

    print("Discretizing...")
    X_train_discretized = discretizer.transform(X_train_scaled)
    # print(sum(X_train_discretized == 2))
    
    n_cutpoints_found = len([l[0] for l in discretizer._cuts.values() if l])
    
    total_bins = n_cutpoints_found + len(discretizer._cuts)
    mean_bins_per_feature = total_bins / len(discretizer._cuts)

    print("Number of cutpoints found: ", n_cutpoints_found)
    print("Total bins: ", total_bins)
    print("Mean bins per feature: ", mean_bins_per_feature)

    # fixedformat_n = np.shape(fixedformat_pairs)[0]
    for fx_pair in fixedformat_pairs:

        n_word = fx_pair[0] 
        n_frac = fx_pair[1]


    exit(0)


    X_test_scaled = scaler.fit_transform(X_test)
    X_test_discretized = discretizer.transform(X_test_scaled)
    
    gnb = GaussianNB()
    gnb_disc = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    gnb_disc.fit(X_train_discretized, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    y_pred_disc = gnb_disc.predict(X_test_discretized)

    print(y_pred == y_pred_disc)


if __name__ == '__main__':
    main()