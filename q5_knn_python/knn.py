import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created

    pos = np.arange(-5, 5.0, 0.1) # [-5,-4.9,-4.8 ... 4.9] dimensao = 100

    samples = samples.T
    N = samples.shape[0]

    dist = np.sqrt((samples - pos[:, None])**2)


    dist = np.sort(dist,axis = 1)[:,k-1] # esse cara vai ser utilizado para densidade
    density = (k/N) * (1/(2*dist))
    
    estDensity = np.stack((pos,density), axis = 1)
    return estDensity
