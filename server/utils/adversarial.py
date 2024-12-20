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

def adversarial(rate=1.0,
             model_encoder="../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
             model_classifier="../../semantic_extraction/MLP_MNIST.pkl",
             dataset_path="../../semantic_extraction/dataset/mnist",
             output_data_path="../../compress_data",
             adversarial_samples=None):

    channel = int(28 * 28 * rate)
    mlp_encoder, mlp_mnist = load_models(
        model_encoder, model_classifier, channel)
    # test_data = load_test_data(dataset_path)
    testset = mnist.MNIST(dataset_path, train=False,
                          transform=data_transform, download=True)

    subset_indices = [2, 0, 1, 3, 4, 15, 84, 9, 11,
                      51, 5, 0, 2, 3, 4, 5]  # list(range(16))
    subset_testset = Subset(testset, subset_indices)

    test_data = DataLoader(subset_testset, batch_size=128, shuffle=False)


    print("Generating adversarial samples...")
    adversarial_samples = generate_adversarial_samples(
        mlp_encoder, mlp_mnist, test_data, epsilon=0.1, num_steps=5, alpha=0.01
    )

    mlp_encoder.eval()

    with torch.no_grad():
        for i, (im, label) in enumerate(test_data):
            im = Variable(im).view(im.size(0), -1)
            label = Variable(label)

            if adversarial_samples is not None:
                im = adversarial_samples[i]

            out = mlp_encoder(im)
            print("out shape: ", out.shape)
    
    compressed_np = out.detach().cpu().numpy()
    np.save(output_data_path + '/compressed_data.npy', compressed_np)

if __name__ == '__main__':
    adversarial(1.0, "../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
             "../../semantic_extraction/MLP_MNIST.pkl", "../../semantic_extraction/dataset/mnist", "../../compress_data")
