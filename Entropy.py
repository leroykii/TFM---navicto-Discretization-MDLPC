from __future__ import division

__author__ = 'Victor Ruiz, vmr11@pitt.edu'

import pandas as pd
import numpy as np
from fxpmath import Fxp
from math import log

FIXEDFORMAT = Fxp(None, signed=True, n_word=64, n_frac=32)    

def entropy_numpy(data_classes, base=2):
    '''
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    classes = np.unique(data_classes)
    N = len(data_classes)
    ent = 0  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion = len(partition) / N
        #update entropy
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain_numpy(X, y, cut_point):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    entropy_full = entropy_numpy(y)  # compute entropy of full dataset (w/o split)

    #split data at cut_point
    data_left_mask = X <= cut_point #dataset[dataset[feature_label] <= cut_point]
    data_right_mask = X > cut_point #dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())

    gain = entropy_full - (N_left / N) * entropy_numpy(y[data_left_mask]) - \
        (N_right / N) * entropy_numpy(y[data_right_mask])

    return gain


def entropy_numpy_fxp(data_classes, base=2):
    '''
    Computes the entropy of a set of labels (class instantiations)
    :param base: logarithm base for computation
    :param data_classes: Series with labels of examples in a dataset
    :return: value of entropy
    '''
    classes = np.unique(data_classes)
    N = len(data_classes)
    ent_fxp = Fxp(0).like(FIXEDFORMAT)  # initialize entropy

    # iterate over classes
    for c in classes:
        partition = data_classes[data_classes == c]  # data with class = c
        proportion_fxp = Fxp(len(partition) / N).like(FIXEDFORMAT)
        
        # update entropy
        log_result_fxp = Fxp(log(proportion_fxp, base)).like(FIXEDFORMAT)
        partial_ent_fxp = Fxp(proportion_fxp * log_result_fxp).like(FIXEDFORMAT)
        ent_fxp = Fxp(ent_fxp - partial_ent_fxp).like(FIXEDFORMAT)
    
    return ent_fxp

def cut_point_information_gain_numpy_fxp(X, y, cut_point):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    
    # TODO  convert ################

    entropy_full_fxp = entropy_numpy_fxp(y)

    X_fxp = Fxp(X).like(FIXEDFORMAT) # TODO quitar conversion
    cut_point_fxp = Fxp(cut_point).like(FIXEDFORMAT) # TODO quitar conversion

    data_left_mask = (X <= cut_point_fxp) == 1
    data_right_mask = (X > cut_point_fxp) == 1
    (N, N_left, N_right) = (len(X_fxp), data_left_mask.sum(), data_right_mask.sum())

    entropy_left_fxp = entropy_numpy_fxp(y[data_left_mask])
    entropy_right_fxp = entropy_numpy_fxp(y[data_right_mask])
    ratio_left_fxp = Fxp(N_left / N).like(FIXEDFORMAT)
    ratio_right_fxp = Fxp(N_right / N).like(FIXEDFORMAT)
    entropy_left_result_fxp = Fxp(ratio_left_fxp * entropy_left_fxp).like(FIXEDFORMAT)
    entropy_right_result_fxp = Fxp(ratio_right_fxp * entropy_right_fxp).like(FIXEDFORMAT)
    entropy_substract_result_fxp = Fxp(entropy_left_result_fxp + entropy_right_result_fxp).like(FIXEDFORMAT)

    gain_fxp = Fxp(entropy_full_fxp - entropy_substract_result_fxp).like(FIXEDFORMAT)
    gain = gain_fxp.get_val() # TODO quit this conversion
    return gain
    #####################

    # # TODO quitar conversion
    # entropy_full = entropy_numpy(y)  # compute entropy of full dataset (w/o split)

    #         # right_mask = X > cut_point# 
    # # split data at cut_point
    # data_left_mask = X <= cut_point #dataset[dataset[feature_label] <= cut_point]
    # data_right_mask = X > cut_point #dataset[dataset[feature_label] > cut_point]
    # (N, N_left, N_right) = (len(X), data_left_mask.sum(), data_right_mask.sum())

    # gain = entropy_full - (N_left / N) * entropy_numpy(y[data_left_mask]) - \
    #     (N_right / N) * entropy_numpy(y[data_right_mask])

    # return gain