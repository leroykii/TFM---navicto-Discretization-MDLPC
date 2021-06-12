import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP_fxp import MDLP_Discretizer_fxp
from MDLP import MDLP_Discretizer

TEST_EPSILON = 1e-6

# read dataset
dataset = datasets.load_iris()
X, y = dataset['data'], dataset['target']
feature_names, class_names = dataset['feature_names'], dataset['target_names']
numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

# Split between training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Initialize discretizer object and fit to training data
discretizer_fxp = MDLP_Discretizer_fxp(features=numeric_features)
discretizer_fxp.fit(X_train, y_train)

discretizer = MDLP_Discretizer(features=numeric_features)
discretizer.fit(X_train, y_train)

### Initialize test data
input_values = np.array([5.1, 5.8, 6.2, 5.5, 4.9, 5.1, 5.2, 6.1, 5.8, 6.5, 6.2, 5.7, 5.7,
       6.4, 6.3, 5. , 6.1, 5.7, 5. , 5.4, 7.4, 5.7, 4.3, 5.4, 5.1, 5. ,
       6.7, 7.1, 6. , 5. , 5. , 4.4, 5.5, 4.9, 5.6, 5.1, 6.3, 6.2, 6.3,
       5.9, 7.7, 7.7, 4.6, 6.4, 4.8, 5.1, 4.4, 5.6, 5. , 5.9, 5.6, 5.8,
       5.5, 4.9, 4.7, 7. , 5.4, 6.5, 5.7, 6.1, 5.3, 5.7, 6.7, 4.6, 5.5,
       4.5, 7.3, 5. , 4.8, 6.4, 5.2, 6.9, 6.7, 7.2, 5.8, 6.7, 6.8, 6.7,
       6.5, 6.9, 6.3, 5.1, 4.9, 6.6, 5.1, 4.6, 6. , 7.9, 4.6, 5.8, 6.1,
       6.7, 5.4, 6. , 5.5, 5. , 6.5, 7.6, 4.9, 5.5])

class_values = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 1, 0, 1, 0, 0, 0, 2, 1,
        0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 2,
        0, 0, 0, 1, 0, 1, 1, 1, 1, 2, 0, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 0,
        2, 0, 0, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 1, 1, 0, 1, 2,
        0, 1, 2, 2, 0, 1, 1, 0, 2, 2, 0, 1])

class_values = class_values.reshape(100 , 1)

# Configure class
discretizer_fxp._class_labels = class_values
discretizer._class_labels = class_values

boundary_points_fxp = discretizer_fxp.feature_boundary_points_fxp(input_values)
boundary_points = discretizer.feature_boundary_points(input_values)

error_array = boundary_points_fxp.get_val() - boundary_points
error = error_array.mean()

print(error)

if (error < TEST_EPSILON):
    print("Test passed")

