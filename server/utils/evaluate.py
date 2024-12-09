from models import *
from dataset import *
from prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    evaluate_mnist_models()
