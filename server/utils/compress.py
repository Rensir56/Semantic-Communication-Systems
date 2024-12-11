from models import *
from dataset import *
from prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, Subset
import os


def compress(rate=1.0,
             model_encoder="../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
             model_classifier="../../semantic_extraction/MLP_MNIST.pkl",
             dataset_path="../../semantic_extraction/dataset/mnist",
             output_data_path="../../compress_data"):

    os.makedirs(output_data_path, exist_ok=True)
    raw_dim = 28 * 28  # shape of the raw image
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)

    mlp_encoder, _ = load_models(
        model_encoder, model_classifier, channel)
    # testset = mnist.MNIST(dataset_path, train=False,
    #                       transform=data_transform, download=True)
    # test_data = DataLoader(testset, batch_size=128, shuffle=False)
    testset = mnist.MNIST(dataset_path, train=False,
                          transform=data_transform, download=True)

    subset_indices = [2, 0, 1, 3, 4, 15, 84, 9, 11, 51, 5]  # list(range(8))
    subset_testset = Subset(testset, subset_indices)

    test_data = DataLoader(subset_testset, batch_size=128, shuffle=False)

    mlp_encoder.eval()

    with torch.no_grad():
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)

            out = mlp_encoder(im)

    compressed_np = out.detach().cpu().numpy()
    np.save(output_data_path + '/compressed_data.npy', compressed_np)


if __name__ == '__main__':
    compress(1.0, "../../semantic_extraction/MLP_MNIST_encoder_combining_1.000000.pkl",
             "../../semantic_extraction/MLP_MNIST.pkl", "../../semantic_extraction/dataset/mnist", "../../compress_data")
