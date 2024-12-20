import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from .black import *
from utils.models import MLP_MNIST



def black_attack(surrogate_model_path="_surrogate_model.pth", mlp_classifier_path="mlp_classifier.pth", output_dir="./reconstruct_image/black"):
    # Parameters
    num_epochs = 10
    epsilon = 0.1
    learning_rate = 0.001

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)
    subset_indices = [2, 0, 1, 3, 4, 15, 84, 9, 11,
                      51, 5, 0, 2, 3, 4, 5]  # list(range(16))
    subset_testset = Subset(testset, subset_indices)

    test_loader = DataLoader(subset_testset, batch_size=128, shuffle=False)

    # Load target model
    target_model = MLP_MNIST()
    target_model.load_state_dict(torch.load(mlp_classifier_path))

    surrogate_model = load_surrogate_model(surrogate_model_path)

    # Generate adversarial samples
    adversarial_samples = generate_adversarial_samples(surrogate_model, test_loader, epsilon=epsilon)

    save_all_adversarial_samples(test_loader, adversarial_samples, output_dir)

    # Test target model on adversarial samples
    adversarial_accuracy, pred = test_target_model_with_adversarial_samples(target_model, test_loader, adversarial_samples)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    true = torch.tensor([1, 7, 2, 0, 4, 5, 8, 9, 6, 3,
                        1, 7, 1, 0, 4, 1]).to(device)
    # print(f'Pred: {pred}')
    # print(f'True: {true}')
    # print(f'Accuracy: {adversarial_accuracy:.2f}%')

    return {
        'accuracy': adversarial_accuracy,  # 转换为0-1范围
        'predictions': pred.tolist(),  # 转换为Python列表
        'true_labels': true.tolist()  # 转换为Python列表
    }

if __name__ == "__main__":
    black_attack()