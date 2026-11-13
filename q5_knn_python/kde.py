import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points //melhor transformar para NxD
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    samples = samples.T  # NxD
     
    N = samples.shape[0]
    V = 1 #Volume Gaussiana eh 1
    u = pos[:,None] - samples.T

    kernel = gauss1D_mine(u, h)
    density = kernel.sum(axis = 1) / N
    estDensity = np.stack((pos,density), axis = 1)

    return estDensity


    
def gauss1D_mine(m, v): #sendo m = u = pos[:,None] - samples.T e v = h 
    insE = -0.5 * (m / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    return res






''' 
[[-5]
[-4.9]
[-4.8]
...
[4.9]]

-

[[1,2,3,4,5,6,7,8,9,10]]

=

cada linha vai ser as diferencas de uma pos para todos os samples

'''