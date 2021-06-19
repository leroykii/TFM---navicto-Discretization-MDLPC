import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP import MDLP_Discretizer

def main():   

    # Read dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized
    # print(X)
    # exit(0)

    X_col1 = X[:, [0]]
    discretizer = MDLP_Discretizer(features=[0])
    discretizer.fit(X_col1, y)
    X_col1_disc = discretizer.transform(X_col1)

    X_col2 = X[:, [1]]    
    discretizer2 = MDLP_Discretizer(features=[0])
    discretizer2.fit(X_col2, y)
    X_col2_disc = discretizer2.transform(X_col2)
    
    X_col3 = X[:, [2]]    
    discretizer3 = MDLP_Discretizer(features=[0])
    discretizer3.fit(X_col3, y)
    X_col3_disc = discretizer3.transform(X_col3)
    X_col4 = X[:, [3]]    
    discretizer4 = MDLP_Discretizer(features=[0])
    discretizer4.fit(X_col4, y)
    X_col4_disc = discretizer4.transform(X_col4)

    # print(X_col1)
    # print(X_col1_disc+1)
    # print ("Interval cut-points: %s" % str(discretizer._cuts[0]))
    # print ("Bin descriptions: %s" % str(discretizer._bin_descriptions[0]))


   
    # Load results obtained by using mdlp library in R
    fname = "./files/iris_dataset.txt"
    R_iris_dataset = np.loadtxt(fname, delimiter=',', usecols=(0,1,2,3))
    fname = "./files/iris_cutpoints.txt"
    R_iris_cutpoints = np.loadtxt(fname, delimiter=',')
    fname = "./files/iris_discretized.txt"
    R_iris_discretized = np.loadtxt(fname, delimiter=',', usecols=(0,1,2,3))
    # print(R_iris_dataset)
    # print(R_iris_cutpoints)
    # print(R_iris_discretized)

    # print(X_col1_disc[:,0]+1)
    # print(R_iris_discretized[:, 0])

    if ((X_col1_disc[:,0]+1) == (R_iris_discretized[:, 0])).all() :
        print("OK col1")
    else :
        print("NOK")

    # print(X_col2_disc[:,0]+1)
    # print(R_iris_discretized[:, 1])

    if ((X_col2_disc[:,0]+1) == (R_iris_discretized[:, 1])).all() :
        print("OK col2")
    else :
        print("NOK")

    if ((X_col3_disc[:,0]+1) == (R_iris_discretized[:, 2])).all() :
        print("OK col3")
    else :
        print("NOK")

    if ((X_col4_disc[:,0]+1) == (R_iris_discretized[:, 3])).all() :
        print("OK col4")
    else :
        print("NOK")

    print ("Interval cut-points 1: %s" % str(discretizer._cuts[0]))
    print ("Interval cut-points 2: %s" % str(discretizer2._cuts[0]))
    print ("Interval cut-points 3: %s" % str(discretizer3._cuts[0]))
    print ("Interval cut-points 4: %s" % str(discretizer4._cuts[0]))

    print ("Cutpoints from R:\n" , R_iris_cutpoints)

if __name__ == '__main__':
    main()