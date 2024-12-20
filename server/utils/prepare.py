import argparse
import os
import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from .models import *
from .dataset import *
from .prepare import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the model with custom parameters.")
    parser.add_argument('--rate', type=float,  # required=True,
                        default=1, help="Compression rate (0-1)")
    parser.add_argument('--model_encoder', type=str,
                        # required=True,
                        default="../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
                        help="Path to the encoder model file")
    parser.add_argument('--model_classifier', type=str,
                        # required=True,
                        default="../../semantic_extraction/MLP_MNIST.pkl",
                        help="Path to the classifier model file")
    parser.add_argument('--dataset_path', type=str,
                        default="../../semantic_extraction/dataset/mnist", help="Path to the MNIST dataset")
    parser.add_argument('--output_image_path', type=str,
                        default="../../reconstruct_image", help="Path to save images")
    return parser.parse_args()


# 加载模型
def load_models(encoder_path, classifier_path, channel):
    mlp_encoder = MLP(channel)
    mlp_mnist = MLP_MNIST()

    mlp_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    mlp_mnist.load_state_dict(torch.load(classifier_path, map_location='cpu'))

    return mlp_encoder, mlp_mnist

# 加载测试数据
def load_test_data(dataset_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = mnist.MNIST(root=dataset_path, train=False,
                          transform=transform, download=True)
    return DataLoader(testset, batch_size=128, shuffle=False)




# def save_combined_images(adversarial_samples, output_image_path):
#     os.makedirs(output_image_path, exist_ok=True)

#     idx = 0  # 用于命名图像文件
#     for batch in adversarial_samples:  # 每个 batch
#         batch = batch.detach().cpu().numpy()  # 转为 NumPy 数组
#         for sample in batch:  # 每个样本
#             sample = sample.reshape(28, 28)  # 重塑为 28x28
#             sample = (sample * 255).astype(np.uint8)  # 转换为图像像素值
#             img = Image.fromarray(sample)
#             img.save(os.path.join(output_image_path, f"adversarial_{idx}.png"))
#             idx += 1
def save_combined_images(adversarial_samples, output_image_path):
    os.makedirs(output_image_path, exist_ok=True)

    idx = 0  # 用于命名图像文件
    for batch in adversarial_samples:  # 每个 batch
        batch = batch.detach().cpu().numpy()  # 转为 NumPy 数组

        # 确保 batch 是正确的二维形状
        if len(batch.shape) == 1:  # 处理单通道数据
            batch = batch.reshape(1, -1)

        for sample in batch:  # 每个样本
            if sample.size != 28 * 28:
                raise ValueError(f"Sample size {sample.size} does not match 28x28.")

            sample = sample.reshape(28, 28)  # 重塑为 28x28
            sample = (sample * 255).astype(np.uint8)  # 转换为图像像素值
            img = Image.fromarray(sample)
            img.save(os.path.join(output_image_path, f"adversarial_{idx}.png"))
            idx += 1



def save_compressed_images(out, output_image_path):
    os.makedirs(output_image_path, exist_ok=True)
    for ii in range(len(out)):
        image_recover = data_inv_transform(out[ii])
        pil_img = Image.fromarray(np.uint8(image_recover))
        pil_img.save(f"{output_image_path}/mnist_test_{ii}.jpg")