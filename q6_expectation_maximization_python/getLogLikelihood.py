import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD (cada linha eh a coordenada do mean)
    # weights        : Weight vector K for K Gaussians (probabilidade daquela gaussiana)
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    logLikelihood = 0
    K = len(weights)

    for i in X:
        p_xi = 0
        for j in range(K):
            p_xi += gauss_dist(covariances[:,:,j],np.array(means[j][:]), i)*weights[j]

        logLikelihood += np.log(p_xi)

    
    return logLikelihood




def gauss_dist(covariance, mean,x):
    D = len(mean)
    diff = x - mean
    return ((1/np.sqrt(((2*np.pi)**D)*np.linalg.det(covariance))) * np.exp((-1/2)* (diff).T @ np.linalg.inv(covariance) @ (diff)))