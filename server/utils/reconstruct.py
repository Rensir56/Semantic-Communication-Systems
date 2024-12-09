from prepare import *
import os


def reconstruct(output_data_path="./compress_data", output_image_path="./reconstruct_image"):
    os.makedirs(output_image_path, exist_ok=True)

    compressed_np = np.load(f"{output_data_path}/compressed_data.npy")
    compressed_tensor = torch.from_numpy(compressed_np).float()

    save_recovered_images(compressed_tensor, output_image_path)


if __name__ == '__main__':
    reconstruct("../../semantic_extraction/compress_data",
                "../../reconstruct_image")
