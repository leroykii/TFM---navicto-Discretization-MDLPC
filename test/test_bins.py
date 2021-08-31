import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import time 
import pickle

import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from MDLP.MDLP_fxp import MDLP_Discretizer_fxp
from MDLP.MDLP import MDLP_Discretizer

import load_dataset as ld

### Configuration ###

fixedformat_pairs = [(4,2), (8, 4), (12, 8), (16, 8), (32, 16)]
test_partition = .30

def pipeline(data, labels, datasetname):
    
    # Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_partition, random_state = 123)
    
    print("Data shape: ", X_train.shape)

    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    n_features = np.arange(data.shape[1]) 

    print("\n\t##### Original MDLP #####")

    t1_pstart = time.perf_counter() 

    print("Fitting discretizer...")
    # discretizer = MDLP_Discretizer_fxp(features=n_features, n_word=64, n_frac=32)
    discretizer = MDLP_Discretizer(features=n_features)
    
    discretizer.fit(X_train_scaled, y_train)
    #print ("Interval cut-points: %s" % str(discretizer._cuts))

    with open("output_files/" + datasetname[9:] + "_discretizer_original" +".bin", 'wb') as file_to_save:
        pickle.dump(discretizer, file_to_save)

    print("Discretizing...")
    X_train_discretized = discretizer.transform(X_train_scaled)
    # print(sum(X_train_discretized == 2))

    t1_pstop = time.perf_counter()
    
    n_cutpoints_found = len([l[0] for l in discretizer._cuts.values() if l])
    total_bins = n_cutpoints_found + len(discretizer._cuts)
    mean_bins_per_feature = total_bins / len(discretizer._cuts)

    print("Number of cutpoints found: ", n_cutpoints_found)
    print("Total bins: ", total_bins)
    print("Mean bins per feature: ", mean_bins_per_feature)
    print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)

    output_filename = "output_files/" + datasetname[9:] + "_original_discretizer" + ".md";
    md_file = open(output_filename, "w")

    md_file.write("cutpoints: " + str(n_cutpoints_found) + "\ntotalbins: " + str(total_bins) + "\nmeanbins: " + str(mean_bins_per_feature) + "\nperf: " + str(t1_pstop-t1_pstart))
    md_file.close()

    # fixedformat_n = np.shape(fixedformat_pairs)[0]
    for fx_pair in fixedformat_pairs:

        t1_pstart = time.perf_counter() 

        n_word = fx_pair[0] 
        n_frac = fx_pair[1]

        print("\n\t##### n_word: ", n_word, " - n_frac: ", n_frac, " #####")
        
        discretizer = MDLP_Discretizer_fxp(features=n_features, n_word=n_word, n_frac=n_frac, debug=True)

        print("Fitting discretizer...")
        discretizer.fit(X_train_scaled, y_train)

        with open("output_files/" + datasetname[9:] + "_discretizer_w" + str(n_word) + "_f" + str(n_frac) +".bin", 'wb') as file_to_save:
            pickle.dump(discretizer, file_to_save)

        print("Discretizing...")    
        X_train_discretized = discretizer.transform(X_train_scaled)

        t1_pstop = time.perf_counter()

        n_cutpoints_found = len([l[0] for l in discretizer._cuts.values() if l])
        total_bins = n_cutpoints_found + len(discretizer._cuts)
        mean_bins_per_feature = total_bins / len(discretizer._cuts)

        print("Number of cutpoints found: ", n_cutpoints_found)
        print("Total bins: ", total_bins)
        print("Mean bins per feature: ", mean_bins_per_feature)
        print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)

        output_filename = "output_files/" + datasetname[9:] + "_discretizer_w" + str(n_word) + "_f" + str(n_frac) + ".md";
        md_file = open(output_filename, "w")

        md_file.write("cutpoints: " + str(n_cutpoints_found) + "\ntotalbins: " + str(total_bins) + "\nmeanbins: " + str(mean_bins_per_feature) + "\nperf: " + str(t1_pstop-t1_pstart))
        md_file.close()

    # exit(0)


    # X_test_scaled = scaler.fit_transform(X_test)
    # X_test_discretized = discretizer.transform(X_test_scaled)
    
    # gnb = GaussianNB()
    # gnb_disc = GaussianNB()

    # # Train the model using the training sets
    # gnb.fit(X_train, y_train)
    # gnb_disc.fit(X_train_discretized, y_train)

    # # Predict the response for test dataset
    # y_pred = gnb.predict(X_test)
    # y_pred_disc = gnb_disc.predict(X_test_discretized)

    # print(y_pred == y_pred_disc)


def main():

    datasets_filenames = ["datasets/bc-wisc-diag.mat", "datasets/madelon.mat", "datasets/colon.mat", "datasets/leukemia1.mat", "datasets/TOX_171.mat"]
    
    for df in datasets_filenames:

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("____Processing dataset: ", df)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        data, labels = ld.load_dataset(df)
        print("Data shape: ", data.shape)

        pipeline(data, labels, df)
    


if __name__ == '__main__':
    main()