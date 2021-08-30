import pandas as pd
import pymrmr

# Web con info sobre PymRMR: http://home.penglab.com/proj/mRMR/FAQ_mrmr.htm#Q1.2
# Github pymRMR: https://github.com/fbrundu/pymrmr

# Input to pymRMR:
# 1) First parameter is a pandas DataFram.
# 2) Second parameter is a string which defines the internal Feature Selection method to use: "MIQ" or "MID".
# 3) Third parameter is an integer which defines the number of features that should be selected by the algorithm.

# *** MID and MIQ represent the Mutual Information Difference and Quotient schemes, respectively, 
# to combine the relevance and redundancy that are defined using Mutual Information (MI). They are the two most used mRMR schemes.


def main():

    df = pd.read_csv('files/pandas_dataset_colon.csv')

    print(df)
    print(type(df))
    
    pymrmr.mRMR(df, 'MIQ', 100)
    
if __name__ == '__main__':
    main()