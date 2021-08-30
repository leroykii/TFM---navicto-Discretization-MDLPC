import numpy as np
import pandas as pd
import pymrmr


def main():

    # Load result from navicto discretizer
    test_X = np.loadtxt('files/X_discretized.var') # <class 'numpy.ndarray'>
    print(test_X)

    # Convert data to panda dataFrame
    panda_df = pd.DataFrame(test_X)

    # Set column names to str (needed by pymRMR)
    panda_df.columns = panda_df.columns.astype(str)

    # call mRMR algorithm    
    pymrmr.mRMR(panda_df, 'MIQ', 8)

    
if __name__ == '__main__':
    main()