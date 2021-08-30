from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp



def main():
    
    # read dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 123)

    print(X_train[0:10])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(X_train_scaled[0:10])

    n_word = 10
    n_frac = 4

    discretizer = MDLP_Discretizer_fxp(features=numeric_features, n_word=n_word, n_frac=n_frac)
    discretizer_scaled = MDLP_Discretizer_fxp(features=numeric_features, n_word=n_word, n_frac=n_frac)
    
    discretizer.fit(X_train, y_train)
    discretizer_scaled.fit(X_train_scaled, y_train)
    
    X_train_discretized = discretizer.transform(X_train)
    X_train_scaled_discretized = discretizer_scaled.transform(X_train_scaled)

    print(X_train_discretized[0:10])
    print(X_train_scaled_discretized[0:10])

    print(np.array_equal(X_train_discretized,X_train_scaled_discretized))
    print(X_train_discretized == X_train_scaled_discretized)

    print("Interval cut-points: %s" % str(discretizer._cuts))
    print("Interval cut-points (scaled version): %s" % str(discretizer_scaled._cuts))

if __name__ == '__main__':
    main()




