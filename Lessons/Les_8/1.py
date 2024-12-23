import numpy as np


def calculate_inception_score_(p_yx, epsilon=1e-16):
    KL_score = np.multiply(p_yx, np.log((p_yx + epsilon) / (p_yx.mean(axis=0) + epsilon)))
    return np.round(np.exp(np.mean(KL_score.sum(axis=1))), 3)


def calculate_inception_score(p_yx, epsilon=1e-16):
    KL_score = np.mean(np.sum(p_yx * np.log((p_yx + epsilon) / (np.mean(p_yx, axis=0) + epsilon)), axis=1))
    return np.exp(KL_score).round(3)


p_yx = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
print(calculate_inception_score(p_yx, epsilon=1e-16))  # 2.828
print(calculate_inception_score_(p_yx, epsilon=1e-16))  # 2.828
