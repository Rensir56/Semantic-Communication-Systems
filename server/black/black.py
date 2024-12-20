import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os


class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_surrogate_model(surrogate_model, target_model, train_loader, num_epochs=5, lr=0.001):
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    surrogate_model.train()
    target_model.eval()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)  # Flatten the images
            target_output = target_model(images).detach()

            optimizer.zero_grad()
            output = surrogate_model(images)
            loss = nn.CrossEntropyLoss()(output, target_output.argmax(dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

def train_and_save_surrogate_model(target_model, train_loader, save_path="surrogate.pth", num_epochs=5, lr=0.001):
    """
    训练替代模型并保存权重
    :param target_model: 目标模型
    :param train_loader: 训练数据加载器
    :param save_path: 保存权重的路径
    :param num_epochs: 训练轮数
    :param lr: 学习率
    """
    surrogate_model = SurrogateModel()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    surrogate_model.train()
    target_model.eval()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1)  # Flatten the images
            target_output = target_model(images).detach()  # Get target model's output

            optimizer.zero_grad()
            output = surrogate_model(images)
            loss = nn.CrossEntropyLoss()(output, target_output.argmax(dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    torch.save(surrogate_model.state_dict(), save_path)
    print(f"Surrogate model saved to {save_path}")

def load_surrogate_model(load_path="surrogate.pth", channel=64):
    """
    加载已保存的替代模型权重
    :param load_path: 权重文件路径
    :param channel: 模型的隐藏层维度
    :return: 加载好的 SurrogateModel
    """
    surrogate_model = SurrogateModel()
    surrogate_model.load_state_dict(torch.load(load_path))
    surrogate_model.eval()
    print(f"Surrogate model loaded from {load_path}")
    return surrogate_model


def generate_adversarial_samples(surrogate_model, test_loader, epsilon=0.1):
    adversarial_samples = []
    surrogate_model.eval()

    for images, labels in test_loader:
        images = images.view(images.size(0), -1).requires_grad_(True)
        labels = labels.to(torch.long)

        output = surrogate_model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)  # Use classification loss
        loss.backward()

        perturbation = epsilon * images.grad.sign()
        images_adv = images + perturbation
        images_adv = torch.clamp(images_adv, 0, 1)  # Ensure valid pixel range
        adversarial_samples.append(images_adv.detach())
    return adversarial_samples


def test_target_model(target_model, test_loader):
    correct = 0
    total = 0
    target_model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            output = target_model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def test_target_model_with_adversarial_samples(target_model, test_loader, adversarial_samples):
    correct = 0
    total = 0
    target_model.eval()

    for i, (images, labels) in enumerate(test_loader):
        images_adv = adversarial_samples[i]
        output = target_model(images_adv)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, predicted


def save_all_adversarial_samples(test_loader, adversarial_samples, output_dir):
    """
    保存所有原始图片和对应的对抗样本到文件夹
    :param test_loader: 测试集加载器
    :param adversarial_samples: 对抗样本列表
    :param output_dir: 保存图片的文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    total_images = 0

    for batch_idx, (original_images, _) in enumerate(test_loader):
        for i in range(len(original_images)):
            original_path = os.path.join(output_dir, f"original_{total_images + i}.png")
            save_image(original_images[i].view(1, 28, 28), original_path)

            adversarial_path = os.path.join(output_dir, f"adversarial_{total_images + i}.png")
            save_image(adversarial_samples[batch_idx][i].view(1, 28, 28), adversarial_path)

        total_images += len(original_images)

    print(f"Saved {total_images} original and adversarial samples to '{output_dir}'.")
