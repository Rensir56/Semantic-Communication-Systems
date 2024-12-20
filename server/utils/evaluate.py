from .models import *
from .dataset import *
from .prepare import *
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
                          output_image_path=None,
                          output_data_path = "../../compress_data"):
 
    compressed_np = np.load(output_data_path + '/compressed_data.npy')
    compressed_tensor = torch.tensor(compressed_np)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compressed_tensor = compressed_tensor.to(device)
    print("Loaded tensor shape:", compressed_tensor.shape)
    
    eval_loss = 0
    eval_acc = 0
    psnr_all = []
    rate = 1.0
    raw_dim = 28 * 28
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)
    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    mlp_encoder, mlp_mnist = load_models(
        model_encoder, model_classifier, channel)
    mlp_encoder.eval()
    mlp_mnist.eval()

    with torch.no_grad():
        out_mnist = mlp_mnist(compressed_tensor)
        _, pred = out_mnist.max(1)

    true = torch.tensor([1, 7, 2, 0, 4, 8, 9, 6, 3,
                        1, 7, 1, 0, 4, 1, 5]).to(device)

    correct = (pred == true).sum().item()
    total = true.size(0)
    accuracy = correct / total * 100
    # print(f'Pred: {pred}')
    # print(f'True: {true}')
    # print(f'Accuracy: {accuracy:.2f}%')
    return {
       'predictions': pred.cpu().numpy().tolist(),
       'true_labels': true.cpu().numpy().tolist(),
       'accuracy': accuracy
   }

if __name__ == '__main__':
    evaluate_mnist_models(1.0, "../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
                          "../../semantic_extraction/MLP_MNIST.pkl", "../../semantic_extraction/dataset/mnist", "../../reconstruct_image", "../../compress_data")
