from prepare import *


def reconstruct():
    args = parse_args()

    compressed_np = np.load(f"{args.output_data_path}/compressed_data.npy")
    compressed_tensor = torch.from_numpy(compressed_np).float()

    save_recovered_images(compressed_tensor, args.output_image_path)


if __name__ == '__main__':
    reconstruct()
