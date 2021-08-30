
from sklearn import datasets
import numpy as np
from fxpmath import Fxp

FIXEDFORMAT = Fxp(None, signed=True, n_word=64, n_frac=32) 


# TODO añadir a nombre _fxp
class test_pass_format():
    def __init__(self, features=None, raw_data_shape=None, n_word=64, n_frac=32):
       
        self._bin_descriptions = {} # <class 'dict'>
        print(features)
        self.fx_format = Fxp(None, signed=True, n_word=n_word, n_frac=n_frac) 
        self.fx_format.info()


def main():

    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

    #Initialize discretizer object and fit to training data
    # discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer = test_pass_format(features=numeric_features)
 
    exit(0)

if __name__ == '__main__':
    main()

