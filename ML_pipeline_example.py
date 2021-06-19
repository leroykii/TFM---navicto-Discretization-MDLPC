import load_dataset as ld
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris

def example_iris():

    iris = load_iris()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

def main():

    print("HOLA")
    
    data, labels = ld.load_dataset('datasets/bc-wisc-diag.mat')
    print(data)
#     print(labels)
        

    enc = KBinsDiscretizer(n_bins=10, encode='ordinal')

    #data = [0.1, 3.4, 8.8, 10.2, 2.33]

    data_binned = enc.fit_transform(data, labels)

    print(data_binned)

    #fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
    #ax1.plot(data)
    #ax2.plot(data_binned)
    print(data.shape)
    print(data_binned.shape)
    # plt.show()

    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data_binned, labels)

    print(data_binned[0,:])
    predict = clf.predict(data_binned[19:30,:])
    
    print(clf)

    
if __name__ == '__main__':
    main()