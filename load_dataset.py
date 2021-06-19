import scipy.io


def load_dataset(filename):

# TODO: hacer tipo match case (python 3.10, que todavía estaba en beta. Mirar si ya me deja actualizar desde anaconda)
    mat = scipy.io.loadmat(filename) 
    data = mat['data']
    labels = mat['labels']

#TODO check NaNs

    return data, labels

def main():

    # bcwisc
    mat = scipy.io.loadmat('datasets/bc-wisc-diag.mat') # (569, 30)
    data = mat['data']
    labels = mat['labels']

    # colon
    mat = scipy.io.loadmat('datasets/colon.mat') # (62, 2000)
    data = mat['data']
    labels = mat['labels']

    # leukemia1
    mat = scipy.io.loadmat('datasets/leukemia1.mat') # (72, 5327)
    data = mat['data']
    labels = mat['labels']

    # madelon
    mat = scipy.io.loadmat('datasets/madelon.mat') # (2400, 500)
    data = mat['data']
    labels = mat['labels']

    # tox_171
    mat = scipy.io.loadmat('datasets/TOX_171.mat') # (171, 5748)
    data = mat['data']
    labels = mat['labels']
    # 
    
    print("OK! Exiting...")

if __name__ == '__main__':
    main()



