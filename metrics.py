import numpy as np


def PSNR(prediction, target):
    assert len(prediction.shape) == 2
    assert prediction.shape == target.shape

    prediction_ext = prediction.astype(np.int64)
    target_ext = target.astype(np.int64)

    mse = np.sum((prediction_ext - target_ext) ** 2) / (prediction.shape[0] * prediction.shape[1])
    if mse == 0:
        raise ValueError

    result = 10 * np.log10(255 ** 2 / mse)

    return result
