import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import pandas as pd
import numpy as np
from torch.autograd import Variable
from PIL import Image
import copy
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the model with custom parameters.")

    # 添加参数
    parser.add_argument('--rate', type=float, required=True, default=1,
                        help="Compression rate (0-1)")
    parser.add_argument('--model_encoder', type=str, required=True, default='MLP_MNIST_encoder_combining_1.000000.pkl',
                        help="Path to the encoder model file (e.g., MLP_MNIST_encoder_combining_1.000000.pkl)")
    parser.add_argument('--model_classifier', type=str, required=True, default='MLP_MNIST.pkl',
                        help="Path to the classifier model file (e.g., MLP_MNIST.pkl)")
    parser.add_argument('--dataset_path', type=str,
                        default='./dataset/mnist', help="Path to the MNIST dataset")
    parser.add_argument('--output_image_path', type=str, default='image_recover_combing',
                        required=True, help="Path to save the recovered images")
    parser.add_argument('--output_data_path', type=str, default='compress_data',
                        required=True, help="Path to save the recovered images")

    return parser.parse_args()


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


def load_models(encoder_path, classifier_path, channel):
    mlp_encoder = MLP(channel)
    mlp_mnist = MLP_MNIST()

    mlp_encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    mlp_mnist.load_state_dict(torch.load(classifier_path, weights_only=True))

    return mlp_encoder, mlp_mnist


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


def criterion(x_in, y_in, raw_in, mlp_mnist, lambda1, lambda2):
    out_tmp1 = nn.CrossEntropyLoss()
    out_tmp2 = nn.MSELoss()
    z_in = mlp_mnist(x_in)
    mse_in = lambda2 * out_tmp2(x_in, raw_in)
    loss_channel = lambda1 * out_tmp1(z_in, y_in) + lambda2 * mse_in
    return loss_channel


def test_model(mlp_encoder, mlp_mnist, test_data, lambda1, lambda2):
    eval_loss = 0
    eval_acc = 0
    psnr_all = []

    mlp_encoder.eval()
    mlp_mnist.eval()

    with torch.no_grad():
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)

            out = mlp_encoder(im)
            out_mnist = mlp_mnist(out)

            loss = criterion(out, label, im, mlp_mnist, lambda1, lambda2)
            eval_loss += loss.item()

            cr1 = nn.MSELoss()
            mse = cr1(out, im)
            out_np = out.detach().numpy()
            psnr = 10 * np.log10(np.max(out_np) ** 2 /
                                 mse.detach().numpy() ** 2)
            psnr_all.append(psnr)

            _, pred = out_mnist.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc

    print('Test Loss: {:.6f}, Test Accuracy: {:.6f}, Average PSNR: {:.6f}'.format(
        eval_loss / len(test_data), eval_acc / len(test_data), np.mean(psnr_all)))

    return out


def save_recovered_images(out, output_image_path):
    for ii in range(len(out)):
        image_recover = data_inv_transform(out[ii])
        pil_img = Image.fromarray(np.uint8(image_recover))
        pil_img.save(f"{output_image_path}/mnist_test_{ii}.jpg")


def evaluate_mnist_models():
    args = parse_args()

    rate = args.rate

    raw_dim = 28 * 28  # shape of the raw image
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)
    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    mlp_encoder, mlp_mnist = load_models(
        args.model_encoder, args.model_classifier, channel)
    testset = mnist.MNIST(args.dataset_path, train=False,
                          transform=data_transform, download=True)
    test_data = DataLoader(testset, batch_size=128, shuffle=False)

    out = test_model(mlp_encoder, mlp_mnist, test_data, lambda1, lambda2)

    save_recovered_images(out, args.output_image_path)


def compress():
    args = parse_args()

    rate = args.rate

    raw_dim = 28 * 28  # shape of the raw image
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)
    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    mlp_encoder, mlp_mnist = load_models(
        args.model_encoder, args.model_classifier, channel)
    testset = mnist.MNIST(args.dataset_path, train=False,
                          transform=data_transform, download=True)
    test_data = DataLoader(testset, batch_size=128, shuffle=False)

    out = test_model(mlp_encoder, mlp_mnist, test_data, lambda1, lambda2)

    compressed_np = out.detach().cpu().numpy()
    np.save(args.output_data_path + '/compressed_data.npy', compressed_np)


def reconstruct():
    args = parse_args()

    compressed_np = np.load(args.output_data_path)
    compressed_tensor = torch.from_numpy(compressed_np).float()

    save_recovered_images(compressed_tensor, args.output_image_path)


if __name__ == '__main__':
    evaluate_mnist_models()
