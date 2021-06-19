import time
from fxpmath import Fxp
import numpy as np

FIXEDFORMAT = Fxp(None, signed=True, n_word=32, n_frac=16)    
REDUCEDPRECISION = Fxp(None, signed=True, n_word=8, n_frac=4)    
SAMPLE_SIZE = 1000



def main():

    a = Fxp(np.pi).like(FIXEDFORMAT)
    b = Fxp(a).like(REDUCEDPRECISION)
    a_prima = Fxp(b).like(FIXEDFORMAT)

    print(a)
    print(b)
    print(a_prima)


if __name__ == '__main__':
    main()