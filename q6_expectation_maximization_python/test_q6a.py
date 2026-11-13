import numpy as np
from getLogLikelihood import getLogLikelihood

# Testa somente a subtask 6a (nao importa sklearn/matplotlib)
data = [np.loadtxt('data1'), np.loadtxt('data2'), np.loadtxt('data3')]

weights = [0.341398243018411, 0.367330235091507, 0.291271521890082]
means = [
    [3.006132088737974,  3.093100568285389],
    [0.196675859954268, -0.034521603109466],
    [-2.957520528756456,  2.991192198151507],
]
covariances = np.zeros((2, 2, 3))
covariances[:, :, 0] = [[0.949104844872119, -0.170637132238246],
                        [-0.170637132238246, 2.011158266600814]]
covariances[:, :, 1] = [[0.837094104536474, 0.044657749659523],
                        [0.044657749659523, 1.327399518241827]]
covariances[:, :, 2] = [[1.160661833073708, 0.058151801834449],
                        [0.058151801834449, 0.927437098385088]]

expected = [-1.098653352229586e+03, -1.706951862352565e+03, -1.292882804841197e+03]
for idx in range(3):
    ll = getLogLikelihood(means, weights, covariances, data[idx])
    print('data{0}: LogLikelihood={1}, esperado={2}, diff={3}'.format(
        idx + 1, ll, expected[idx], expected[idx] - ll))
