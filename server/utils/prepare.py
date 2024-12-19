from models import *
from dataset import *
from torch.autograd import Variable
from PIL import Image
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the model with custom parameters.")

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
                        required=True, help="Path to save the recovered data")

    return parser.parse_args()


def load_models(encoder_path, classifier_path, channel):
    mlp_encoder = MLP(channel)
    mlp_mnist = MLP_MNIST()

    mlp_encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    mlp_mnist.load_state_dict(torch.load(classifier_path, weights_only=True))

    return mlp_encoder, mlp_mnist


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

        print(f'pred={pred}, true={label}')

    print('Test Loss: {:.6f}, Test Accuracy: {:.6f}, Average PSNR: {:.6f}'.format(
        eval_loss / len(test_data), eval_acc / len(test_data), np.mean(psnr_all)))

    return out


def save_recovered_images(out, output_image_path):
    for ii in range(len(out)):
        image_recover = data_inv_transform(out[ii])
        pil_img = Image.fromarray(np.uint8(image_recover))
        pil_img.save(f"{output_image_path}/mnist_test_{ii}.jpg")
