from models import *
from dataset import *
from prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, Subset

import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到当前目录
os.chdir(current_dir)

def evaluate_mnist_models(rate=1.0,
                          model_encoder="../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
                          model_classifier="../../semantic_extraction/MLP_MNIST.pkl",
                          dataset_path="../../semantic_extraction/dataset/mnist",
                          output_image_path="../../reconstruct_image"):

    raw_dim = 28 * 28  # shape of the raw image
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)
    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    mlp_encoder, mlp_mnist = load_models(
        model_encoder, model_classifier, channel)
    testset = mnist.MNIST(dataset_path, train=False,
                          transform=data_transform, download=True)

    subset_indices = [2, 0, 1, 3, 4, 15, 84, 9, 11, 51, 5]  # list(range(8))
    subset_testset = Subset(testset, subset_indices)

    test_data = DataLoader(subset_testset, batch_size=128, shuffle=False)
    print(len(test_data.dataset))
    # test_data = DataLoader(testset, batch_size=128, shuffle=False)

    out = test_model(mlp_encoder, mlp_mnist, test_data, lambda1, lambda2)
    # print(out.shape)

    save_recovered_images(out, output_image_path)


if __name__ == '__main__':
    evaluate_mnist_models(1.0, "../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
                          "../../semantic_extraction/MLP_MNIST.pkl", "../../semantic_extraction/dataset/mnist", "../../reconstruct_image")
