import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp
from MDLP.MDLP import MDLP_Discretizer

# Interval cut-points 1: [5.55, 6.15]
# Interval cut-points 2: [2.95, 3.3499999999999996]
# Interval cut-points 3: [2.45, 4.75]
# Interval cut-points 4: [0.8, 1.75]

def main():

    ######### USE-CASE EXAMPLE #############

    # read dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 123)

    print(X)

    #Initialize discretizer object and fit to training data
    discretizer = MDLP_Discretizer_fxp(features=numeric_features)
    #discretizer = MDLP_Discretizer(features=numeric_features)
    # discretizer.fit(X_train, y_train)
    discretizer.fit(X, y)
    X_train_discretized = discretizer.transform(X_train)

    #apply same discretization to test set
    X_test_discretized = discretizer.transform(X_test)

    X_discretized = discretizer.transform(X)

    print(type(X))
    print(X.shape)
   # print ("Interval cut-points: %s" % str(discretizer._cuts[0]))
#    print ("Bin descriptions: %s" % str(discretizer._bin_descriptions[0]))
    
    # X_col1 = X[:, [0]]
    # discretizer2 = MDLP_Discretizer_fxp(features=[0])
    # discretizer2.fit(X_col1, y)

    # X_col1_disc = discretizer2.transform(X_col1)
    # print(X_col1)
    # print(X_col1_disc+1)
    # print ("Interval cut-points: %s" % str(discretizer2._cuts[0]))
    # print ("Bin descriptions: %s" % str(discretizer2._bin_descriptions[0]))

    # exit()

    print(feature_names)
    print(class_names)

    #Print a slice of original and discretized data
    #print ("Original dataset:\n%s" % str(X_train[0:5]))
    #print ("Discretized dataset:\n%s" % str(X_train_discretized[0:5]))

    print ("Original dataset:\n%s" % str(X[0:10]))
    print ("Discretized dataset:\n%s" % str(X_discretized[0:10]))

    print ("Original dataset:\n%s" % str(X))
    print ("Discretized dataset:\n%s" % str(X_discretized))

    #see how feature 0 was discretized
    print ("Feature: %s" % feature_names[0])
    print ("Interval cut-points: %s" % str(discretizer._cuts[0]))
    print ("Interval cut-points: %s" % str(discretizer._cuts[1]))
    print ("Interval cut-points: %s" % str(discretizer._cuts[2]))
    print ("Interval cut-points: %s" % str(discretizer._cuts[3]))
    #print ("Bin descriptions: %s" % str(discretizer._bin_descriptions[0]))

    # Dump resulted dataset to file
    np.savetxt('files/X_discretized.var', X_discretized)

if __name__ == '__main__':
    main()