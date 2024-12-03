import numpy as np
import torch


def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x


def data_inv_transform(x):
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = recover_data.detach().numpy()
    return recover_data
