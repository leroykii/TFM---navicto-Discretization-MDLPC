import time
from fxpmath import Fxp
#import matplotlib.pyplot as plt
import numpy as np

FIXEDFORMAT = Fxp(None, signed=True, n_word=32, n_frac=16)    
SAMPLE_SIZE = 1000

def procedure():
   time.sleep(2.5)

def test_profiling_methods():

    # assigning n = 50 
    n = 20000
    
    # Start the stopwatch / counter 
    t1_start = time.process_time() # It does not include time elapsed during sleep
    t1_pstart = time.perf_counter()  # Include time elapsed during sleep and is system-wide
    
    for i in range(n):
        print(i, end =' ')
    
    time.sleep(1.0)

    print() 
    
    # Stop the stopwatch / counter
    t1_stop = time.process_time()
    t1_pstop = time.perf_counter()
    
   
    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
        
    print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)


def main():

    np.random.seed(123)

    print()
    print("### Float benchmark")

    test_vector_a = np.random.rand(SAMPLE_SIZE)
    test_vector_b = np.random.rand(SAMPLE_SIZE)

    sum_result= np.zeros(SAMPLE_SIZE)
    sub_result = np.zeros(SAMPLE_SIZE)
    mult_result = np.zeros(SAMPLE_SIZE)
    div_result = np.zeros(SAMPLE_SIZE)

    print("Starting measures... :")

    t1_start = time.process_time()
    t1_pstart = time.perf_counter() 

    for i in range(1, test_vector_a.size):
        sum_result[i] = test_vector_a[i] + test_vector_b[i]
        sub_result[i] = test_vector_a[i] - test_vector_b[i]
        mult_result[i] = test_vector_a[i] * test_vector_b[i]
        # div_result[i] = test_vector_a[i] / test_vector_b[i]

        if (i % 10000 == 0):
            print("[1/2] Processed: ", 100*i/SAMPLE_SIZE, "%")

    t1_stop = time.process_time()
    t1_pstop = time.perf_counter()
    
   
  #  print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
        
    print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)
    
    #########################
    ### Fixed point benchmark
    print()
    print("### Fixed point benchmark Q32.16)")

    test_vector_a_fxp = Fxp(test_vector_a).like(FIXEDFORMAT)
    test_vector_b_fxp = Fxp(test_vector_b).like(FIXEDFORMAT)

    sum_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
    sub_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
    mult_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
    div_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)

    print("Starting measures... :")

    t1_start = time.process_time()
    t1_pstart = time.perf_counter() 

    for i in range(1, test_vector_a_fxp.size):
        sum_result_fxp[i] = Fxp((test_vector_a_fxp[i] + test_vector_b_fxp[i])).like(FIXEDFORMAT)
        sub_result_fxp[i] = Fxp((test_vector_a_fxp[i] - test_vector_b_fxp[i])).like(FIXEDFORMAT)
        mult_result_fxp[i] = Fxp((test_vector_a_fxp[i] * test_vector_b_fxp[i])).like(FIXEDFORMAT)
       # div_result_fxp[i] = Fxp((test_vector_a_fxp[i] / test_vector_b_fxp[i])).like(FIXEDFORMAT)

        if (i % 10000 == 0):
            print("[1/2] Processed: ", 100*i/SAMPLE_SIZE, "%")

    t1_stop = time.process_time()
    t1_pstop = time.perf_counter()

  #  print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
        
    print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)

    if (0):
        #########################
        ### Fixed point benchmark, using equal
        print()
        print("### Fixed point benchmark (using .equal)")

    #   test_vector_a_fxp = Fxp(test_vector_a).like(FIXEDFORMAT)
    #   test_vector_b_fxp = Fxp(test_vector_b).like(FIXEDFORMAT)

        sum_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
        sub_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
        mult_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)
        div_result_fxp = Fxp(np.zeros(SAMPLE_SIZE)).like(FIXEDFORMAT)

        print("Starting measures... :")

        t1_start = time.process_time()
        t1_pstart = time.perf_counter() 

        for i in range(1, test_vector_a_fxp.size):
            sum_result_fxp[i].equal(test_vector_a_fxp[i] + test_vector_b_fxp[i])
            sub_result_fxp[i].equal(test_vector_a_fxp[i] - test_vector_b_fxp[i])
            mult_result_fxp[i].equal(test_vector_a_fxp[i] * test_vector_b_fxp[i])
        # div_result_fxp[i] = Fxp((test_vector_a_fxp[i] / test_vector_b_fxp[i])).like(FIXEDFORMAT)

            if (i % 10000 == 0):
                print("[1/2] Processed: ", 100*i/SAMPLE_SIZE, "%")

        t1_stop = time.process_time()
        t1_pstop = time.perf_counter()

        print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
            
        print("[PERF] Elapsed time during the whole program in seconds:", t1_pstop-t1_pstart)


if __name__ == '__main__':
    main()