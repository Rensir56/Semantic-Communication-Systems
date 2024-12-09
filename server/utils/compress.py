from models import *
from dataset import *
from prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader


def compress():
    args = parse_args()

    rate = args.rate

    raw_dim = 28 * 28  # shape of the raw image
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)

    mlp_encoder, _ = load_models(
        args.model_encoder, args.model_classifier, channel)
    testset = mnist.MNIST(args.dataset_path, train=False,
                          transform=data_transform, download=True)
    test_data = DataLoader(testset, batch_size=128, shuffle=False)

    mlp_encoder.eval()

    with torch.no_grad():
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)

            out = mlp_encoder(im)

    compressed_np = out.detach().cpu().numpy()
    np.save(args.output_data_path + '/compressed_data.npy', compressed_np)


if __name__ == '__main__':
    compress()
