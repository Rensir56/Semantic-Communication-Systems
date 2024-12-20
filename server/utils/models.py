import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, channel)
        self.fc2 = nn.Linear(channel, 28 * 28)

    def forward(self, x):
        # 第一层全连接
        x = self.fc1(x)

        # 归一化并模拟量化
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_tmp = x / x_max  # 归一化到 [0, 1]
        x_tmp = torch.round(x_tmp * 256) / 256  # 模拟量化
        x = x_tmp * x_max  # 还原量化后的值

        # 添加噪声
        out_square = x.pow(2).mean(dim=1, keepdim=True)
        snr = 10  # dB
        aver_noise = out_square / (10 ** (snr / 10))
        noise = torch.randn_like(x) * torch.sqrt(aver_noise)
        x = x + noise

        # 第二层全连接
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
        x = self.fc4(x)  # 输出 logits
        return x


def criterion(x_in, y_in, raw_in, mlp_mnist, lambda1, lambda2):
    # 分类损失
    cross_entropy = nn.CrossEntropyLoss()
    # 重构损失
    mse_loss = nn.MSELoss()

    # 分类输出
    z_in = mlp_mnist(x_in)
    # 计算总损失
    loss = lambda1 * cross_entropy(z_in, y_in) + \
        lambda2 * mse_loss(x_in, raw_in)
    return loss

# 对抗样本生成
def generate_adversarial_samples(mlp_encoder, mlp_mnist, test_data, epsilon, num_steps=1, alpha=0.01):
    mlp_encoder.eval()
    mlp_mnist.eval()

    adversarial_samples = []
    for im, label in test_data:
        im = im.view(im.size(0), -1).to(torch.float32).requires_grad_(True)
        label = label.to(torch.long)

        perturbation = torch.zeros_like(im).to(im.device)
        for step in range(num_steps):
            out = mlp_encoder(im + perturbation)
            pred = mlp_mnist(out)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, label)

            mlp_encoder.zero_grad()
            mlp_mnist.zero_grad()
            loss.backward(retain_graph=True)

            perturbation = perturbation + alpha * im.grad.sign()
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            perturbation = torch.clamp(im + perturbation, 0, 1) - im
            im.grad.zero_()

        adversarial_samples.append((im + perturbation).detach())

    return adversarial_samples


# 测试模型
def test_model(mlp_encoder, mlp_mnist, test_data, lambda1, lambda2, adversarial_samples=None):
    eval_loss = 0
    eval_acc = 0

    mlp_encoder.eval()
    mlp_mnist.eval()

    with torch.no_grad():
        for i, (im, label) in enumerate(test_data):
            im = Variable(im).view(im.size(0), -1)
            label = Variable(label)

            if adversarial_samples is not None:
                im = adversarial_samples[i]

            out = mlp_encoder(im)
            out_mnist = mlp_mnist(out)

            _, pred = out_mnist.max(1)
            eval_acc += (pred == label).sum().item() / label.size(0)

    print(f"Test Accuracy: {eval_acc / len(test_data):.6f}")
