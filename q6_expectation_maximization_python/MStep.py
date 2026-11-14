import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N = X.shape[0]
    K = gamma.shape[1]
    D = X.shape[1]
    means = np.zeros((K,D))
    weights = np.zeros(K)
    covariances = np.zeros((D,D,K))

    for j in range(K):
        N_j = gamma.sum(axis = 0)[j]
        weights[j] = N_j/N
        means[j] = (1/N_j) * (gamma[:,j,None] * X).sum(axis = 0)
        covariances[:,:,j] = 1/N_j * (gamma[:,j,None] * (X - means[j])).T @ (X - means[j])

    logLikelihood = getLogLikelihood(means,weights,covariances,X)

    return weights, means, covariances, logLikelihood
