from .prepare import *
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到当前目录
os.chdir(current_dir)


def reconstruct(output_data_path="./result/compress_data", 
                output_image_path="./result/reconstruct_image/origin", 
                uap=False):
    
    os.makedirs(output_image_path, exist_ok=True)

    compressed_np = np.load(f"{output_data_path}/compressed_data.npy")
    compressed_tensor = torch.from_numpy(compressed_np).float()

    if uap is True:
        save_combined_images(compressed_tensor, output_image_path)
    else:
        save_compressed_images(compressed_tensor, output_image_path)


if __name__ == '__main__':
    uap=False
    if uap is True:
        output_image_path="./result/reconstruct_image/white"
    else:
        output_image_path="./result/reconstruct_image/origin"
    reconstruct("../../compress_data", output_image_path, uap)
