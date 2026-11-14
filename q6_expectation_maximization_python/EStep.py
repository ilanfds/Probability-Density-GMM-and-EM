import numpy as np
from getLogLikelihood import getLogLikelihood
from getLogLikelihood import gauss_dist


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    K = len(weights)
    N = X.shape[0]
    gamma = np.zeros((N,K))


    for idx,p in enumerate(X):
        denominator = 0
        for i in range(K):
            denominator += gauss_dist(covariances[:,:,i], np.array(means[i]), p) * weights[i]

        for j in range(K):
            numerator = gauss_dist(covariances[:,:,j], np.array(means[j]), p) * weights[j]
            gamma[idx,j] = numerator / denominator

    logLikelihood = getLogLikelihood(means,weights,covariances,X)        

    
    return [logLikelihood, gamma]
