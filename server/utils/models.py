import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np


class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, channel)
        self.fc2 = nn.Linear(channel, 28 * 28)

    def forward(self, x):
        x = self.fc1(x)

        x = x.detach().cpu()
        x_max = torch.max(x)
        x_tmp = copy.deepcopy(torch.div(x, x_max))

        x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
        x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
        x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
        x_tmp = copy.deepcopy(torch.div(x_tmp, 256))

        x = copy.deepcopy(torch.mul(x_tmp, x_max))

        x_np = x.detach().numpy()
        out_square = np.square(x_np)
        aver = np.sum(out_square) / np.size(out_square)

        snr = 10  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)

        x_np = x_np + noise
        x = torch.from_numpy(x_np)
        x = x.to(torch.float32)

        x = self.fc2(x)
        return x


class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def criterion(x_in, y_in, raw_in, mlp_mnist, lambda1, lambda2):
    out_tmp1 = nn.CrossEntropyLoss()
    out_tmp2 = nn.MSELoss()
    z_in = mlp_mnist(x_in)
    mse_in = lambda2 * out_tmp2(x_in, raw_in)
    loss_channel = lambda1 * out_tmp1(z_in, y_in) + lambda2 * mse_in
    return loss_channel
