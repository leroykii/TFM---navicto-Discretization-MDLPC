from __future__ import division
from re import TEMPLATE
__author__ = 'Victor Ruiz, vmr11@pitt.edu'
import numpy as np
from Entropy import entropy_numpy, entropy_numpy_fxp, cut_point_information_gain_numpy, cut_point_information_gain_numpy_fxp
from math import log
from sklearn.base import TransformerMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fxpmath import Fxp


FIXEDFORMAT = Fxp(None, signed=True, n_word=64, n_frac=32)    


def previous_item(a, val):
    idx = np.where(a == val)[0][0] - 1
    return a[idx]

# TODO añadir a nombre _fxp
class MDLP_Discretizer_fxp(TransformerMixin):
    def __init__(self, features=None, raw_data_shape=None):
        '''
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature
        :param X: pandas dataframe with data to discretize
        :param class_label: name of the column containing class in input dataframe
        :param features: if !None, features that the user wants to discretize specifically
        :return:
        '''
        #Initialize descriptions of discretization bins
        self._bin_descriptions = {} # <class 'dict'>

        #Create array with attr indices to discretize
        if features is None:  # Assume all columns are numeric and need to be discretized <class 'numpy.ndarray'>
            if raw_data_shape is None:
                raise Exception("If feautes=None, raw_data_shape must be a non-empty tuple")
            self._col_idx = range(raw_data_shape[1])
        else:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if np.issubdtype(features.dtype, np.integer):
                self._col_idx = features
            elif np.issubdtype(features.dtype, np.bool):  # features passed as mask
                if raw_data_shape is None:
                    raise Exception('If features is a boolean array, raw_data_shape must be != None')
                if len(features) != self._data_raw.shape[1]:
                    raise Exception('Column boolean mask must be of dimensions (NColumns,)')
                self._col_idx = np.where(features)
            else:
                raise Exception('features argument must a np.array of column indices or a boolean mask')

    def fit(self, X, y):
        self._data_raw = X  # copy of original input data || <class 'numpy.ndarray'> float64
        #TODO change self._data_raw to fixedpoint
        self._class_labels = y.reshape(-1, 1)  # make sure class labels is a column vector || <class 'numpy.ndarray'> int32
        self._classes = np.unique(self._class_labels) # <class 'numpy.ndarray'> int32


        if len(self._col_idx) != self._data_raw.shape[1]:  # some columns will not be discretized
            self._ignore_col_idx = np.array([var for var in range(self._data_raw.shape[1]) if var not in self._col_idx])

        # initialize feature bins cut points

        self._cuts = {f: [] for f in self._col_idx} # <class 'dict'>      # TODO change name to _fxp

        # pre-compute all boundary points in dataset
        #self._boundaries = self.compute_boundary_points_all_features() # <class 'numpy.ndarray'> float64
        self._boundaries = self.compute_boundary_points_all_features_fxp()

        # get cuts for all features
        self.all_features_accepted_cutpoints_fxp()

        #generate bin string descriptions
        self.generate_bin_descriptions()

        #Generate one-hot encoding schema

        return self

    def transform(self, X, inplace=False):
        #TODO ver qué hacer con esto
        # if inplace:
        #     discretized = X
        # else:
        #     discretized = X.copy()
        
        X_fxp = Fxp(X).like(FIXEDFORMAT)

        discretized = self.apply_cutpoints_fxp(X_fxp)
        return discretized

    #TODO ver qué hacer con esto (no se ha probado)
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, inplace=True)     

    def MDLPC_criterion_fxp(self, X_fxp, y, feature_idx, cut_point_fxp):
        '''
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        '''

        #get dataframe only with desired attribute and class columns, and split by cut_point
        left_mask = (X_fxp <= cut_point_fxp) == 1
        right_mask = (X_fxp > cut_point_fxp) == 1

        #compute information gain obtained when splitting data at cut_point   
        cut_point_gain_fxp = cut_point_information_gain_numpy_fxp(X_fxp, y, cut_point_fxp)

        #compute delta term in MDLPC criterion
        N = len(X_fxp) # number of examples in current partition
        partition_entropy_fxp = entropy_numpy_fxp(y)
        k = len(np.unique(y))
        k_left = len(np.unique(y[left_mask]))
        k_right = len(np.unique(y[right_mask]))
        entropy_left_fxp = entropy_numpy_fxp(y[left_mask])  # entropy of partition
        entropy_right_fxp = entropy_numpy_fxp(y[right_mask])
        
        # Compute delta
        entropy_left_and_right_fpx = Fxp((k_left * entropy_left_fxp) + (k_right * entropy_right_fxp)).like(FIXEDFORMAT)
        entropy_all_contribution_fpx = Fxp( - (k * partition_entropy_fxp) + entropy_left_and_right_fpx).like(FIXEDFORMAT)
        log_delta_fxp = Fxp(log(3 ** k, 2)).like(FIXEDFORMAT)   # TODO LOG PENDIENTE
        delta = Fxp(log_delta_fxp + entropy_all_contribution_fpx).like(FIXEDFORMAT)

        #to split or not to split

        log_gain_threshold_fxp = Fxp(log(N - 1, 2)).like(FIXEDFORMAT)   # TODO LOG PENDIENTE
        gain_threshold_sum = Fxp(log_gain_threshold_fxp + delta).like(FIXEDFORMAT)
        gain_threshold_fxp = Fxp(gain_threshold_sum / N).like(FIXEDFORMAT)

        if cut_point_gain_fxp > gain_threshold_fxp:
            return True
        else:
            return False

    def feature_boundary_points_fxp(self, values_fxp):
        '''
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        '''
        
        if (0): # TODO if needed
            missing_mask = np.isnan(values) # -> Comprueba si hay NaNs en los datos de entrada para eliminarlos luego
            data_partition = np.concatenate([values[:, np.newaxis], self._class_labels], axis=1) # Partición con slice/columna de datos + clase
            data_partition = data_partition[~missing_mask] # -> Elimina NaNs de los datos de entrada, junto a su clase
            #sort data by values
            data_partition = data_partition[data_partition[:, 0].argsort()]

        # class_labels_fxp = Fxp(self._class_labels).like(FIXEDFORMAT)

        # TODO: eliminar NaNs?

        # data_partition = np.concatenate([values_fxp[:, np.newaxis].get_val(), self._class_labels], axis=1)
        # data_partition_fxp = Fxp(data_partition).like(FIXEDFORMAT)
        ############################# FXP operatoin
        unique_vals_fxp = np.unique(values_fxp).like(FIXEDFORMAT) # each of this could be a bin boundary
        
        boundaries_fxp_list = []
        for i in range(1, unique_vals_fxp.size):
            previous_val_idx_fxp = np.where(values_fxp == unique_vals_fxp[i-1])[0]
            current_val_idx_fxp = np.where(values_fxp == unique_vals_fxp[i])[0]
            merged_classes_fxp = np.union1d(self._class_labels[previous_val_idx_fxp], self._class_labels[current_val_idx_fxp])
            if merged_classes_fxp.size > 1:
              #  print(merged_classes_fxp)
                boundaries_fxp_list += [unique_vals_fxp[i]]
        
        boundaries_offset_fxp_list = [previous_item(unique_vals_fxp, var) for var in boundaries_fxp_list]

        # Convert to Fxp Array
        boundaries_offset_fxp = Fxp(np.array(boundaries_offset_fxp_list)).like(FIXEDFORMAT)
        boundaries_fxp = Fxp(np.array(boundaries_fxp_list)).like(FIXEDFORMAT)
        
        partial_sum = (boundaries_fxp + boundaries_offset_fxp).like(FIXEDFORMAT)
        boundary_points_fxp =  (partial_sum / 2).like(FIXEDFORMAT)

        return boundary_points_fxp


    def compute_boundary_points_all_features_fxp(self):
        '''
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        '''
        def padded_cutpoints_array(arr, N):
            cutpoints = self.feature_boundary_points_fxp(Fxp(arr).like(FIXEDFORMAT))
            cutpoints = cutpoints.get_val()
            padding = np.array([np.nan] * (N - len(cutpoints)))
            return np.concatenate([cutpoints, padding])

        boundaries = np.empty(self._data_raw.shape)
        boundaries[:, self._col_idx] = np.apply_along_axis(padded_cutpoints_array, 0, self._data_raw[:, self._col_idx], self._data_raw.shape[0])
        # numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)[source] -> Apply a function to 1-D slices along the given axis.
        # self._data_raw[:, self._col_idx] -> datos de entrada
        # self._col_idx -> array([0, 1, 2, 3])
        # self._data_raw.shape -> (100,4) ; self._data_raw.shape[0] -> (100)
         
        mask = np.all(np.isnan(boundaries), axis=1)
        
        return boundaries[~mask]

    def boundaries_in_partition_fxp(self, X_fxp, feature_idx):
        '''
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        '''
       
        range_min_fxp, range_max_fxp = (X_fxp.min(), X_fxp.max())

        # TODO quitar los temps, hacer mejor
        
        tmp_boundaries_col = self._boundaries[:, feature_idx]
        tmp = ~np.isnan(tmp_boundaries_col)
        tmp_boundaries_fxp = Fxp(tmp_boundaries_col[tmp]).like(FIXEDFORMAT)
        mask = np.logical_and((tmp_boundaries_fxp > range_min_fxp), (tmp_boundaries_fxp < range_max_fxp))

        if (mask.any() == True):
           # print("at least one is true")
            ret_unique_fxp = np.unique(tmp_boundaries_fxp[mask]).like(FIXEDFORMAT)
            return ret_unique_fxp
        else:
            # print("no one is true")
            return np.array([])
        
    def best_cut_point_fxp(self, X_fxp, y, feature_idx):
        '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        '''
        candidates_fxp = self.boundaries_in_partition_fxp(X_fxp, feature_idx=feature_idx)

        if candidates_fxp.size == 0:
            return None

        gain_fxp = [(cut, cut_point_information_gain_numpy_fxp(X_fxp, y, cut_point_fxp=cut)) for cut in candidates_fxp]
        gain_fxp = sorted(gain_fxp, key=lambda x: x[1], reverse=True)

        return gain_fxp[0][0] # return cut point

       
    def single_feature_accepted_cutpoints_fxp(self, X_fxp, y, feature_idx):
        '''
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        '''
        # TODO, meter entrada en fxp

        # TODO gestionar missing data
        # mask = np.isnan(X)
        # X = X[~mask]
        # y = y[~mask]
        
        # stop if constant or null feature values
        if len(np.unique(X_fxp)) < 2:
            return
        #determine whether to cut and where
        cut_candidate_fxp = self.best_cut_point_fxp(X_fxp, y, feature_idx) 
        if cut_candidate_fxp == None:
            return

        cut_candidate = cut_candidate_fxp.get_val() # TODO remove conversion
        print(cut_candidate) # TODO remove print
        # decision = self.MDLPC_criterion(X, y, feature_idx, cut_candidate_fxp)  # TODO

        decision = self.MDLPC_criterion_fxp(X_fxp, y, feature_idx, cut_candidate_fxp)  # TODO

        # apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            # partition masks
            left_mask = (X_fxp <= cut_candidate_fxp) == 1
            right_mask = (X_fxp > cut_candidate_fxp) == 1

            #now we have two new partitions that need to be examined
            left_partition_fxp = X_fxp[left_mask] 
            right_partition_fxp = X_fxp[right_mask]
            if (left_partition_fxp.size == 0) or (right_partition_fxp.size == 0):
                return # extreme point selected, don't partition
            self._cuts[feature_idx] += [cut_candidate_fxp]  # accept partition
            self.single_feature_accepted_cutpoints_fxp(left_partition_fxp, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints_fxp(right_partition_fxp, y[right_mask], feature_idx)
            # order cutpoints in ascending order
            self._cuts[feature_idx] = sorted(self._cuts[feature_idx])
            return

    def all_features_accepted_cutpoints_fxp(self):
        '''
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        '''
        for attr in self._col_idx:
            X_fxp = Fxp(self._data_raw[:, attr]).like(FIXEDFORMAT)   # TODO remove conversion
            self.single_feature_accepted_cutpoints_fxp(X_fxp=X_fxp, y=self._class_labels, feature_idx=attr)
        return

    def generate_bin_descriptions(self):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        bin_label_collection = {}
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._bin_descriptions[attr] = {i: bin_labels[i] for i in range(len(bin_labels))}


    # def apply_cutpoints(self, data):
    #     '''
    #     Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
    #     the bins
    #     :param out_data_path: path to save discretized data
    #     :param out_bins_path: path to save bins description
    #     :return:
    #     '''
    #     for attr in self._col_idx:
    #         if len(self._cuts[attr]) == 0:
    #             # data[:, attr] = 'All'
    #             data[:, attr] = 0
    #         else:
    #             cuts = [-np.inf] + self._cuts[attr] + [np.inf]
    #             discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('float') - 1
    #             discretized_col[np.isnan(data[:, attr])] = np.nan
    #             data[:, attr] = discretized_col
    #     return data


    def apply_cutpoints_fxp(self, data_fxp):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''

        data = data_fxp.get_val()
        
        # Copy shape of data for the output vector
        discretized = np.ones_like(data_fxp).like(FIXEDFORMAT) * -1

        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                # data[:, attr] = 'All'
                discretized[:, attr] = 0
            else:
                cuts = [FIXEDFORMAT.lower] + self._cuts[attr] + [FIXEDFORMAT.upper]
                discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('int') - 1   # TODO np.digitize
                # discretized_col[np.isnan(data[:, attr])] = np.nan # TODO quité esto, si voy a soportar Nans, revisar
                discretized[:, attr] = discretized_col
        return discretized