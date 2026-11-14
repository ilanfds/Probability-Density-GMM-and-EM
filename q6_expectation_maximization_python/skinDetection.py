import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####

    H, W, C = img.shape
    X = img.reshape(H * W, C)        # (N, 3) -> cada linha eh um pixel RGB
    N = X.shape[0]

    s_EM = estGaussMixEM(sdata,K,n_iter,epsilon)

    s_weights =  s_EM[0]
    s_means = s_EM[1]
    s_covariances = s_EM[2]

    ns_EM = estGaussMixEM(ndata,K,n_iter,epsilon)

    ns_weights = ns_EM[0]
    ns_means = ns_EM[1]
    ns_covariances = ns_EM[2]

    p_s = np.zeros(N)
    p_ns = np.zeros(N)

    for idx,px in enumerate(X):
            p_s[idx] = getLogLikelihood(s_means,s_weights,s_covariances,[px])
            p_ns[idx] = getLogLikelihood(ns_means,ns_weights,ns_covariances,[px])

    result = np.where(np.exp(p_s - p_ns) > theta, True, False)
    result = result.reshape(H,W)

    pass
    return result
