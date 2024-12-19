from prepare import *
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到当前目录
os.chdir(current_dir)

def reconstruct(normal_output_data_path="./compress_data", 
                adversarial_output_data_path="./compress_data", 
                normal_output_image_path="./reconstruct_image",
                adversarial_output_image_path="./reconstruct_image_uap"):
    os.makedirs(normal_output_image_path, exist_ok=True)
    os.makedirs(adversarial_output_image_path, exist_ok=True)

    # 加载正常和对抗压缩数据
    normal_compressed_np = np.load(f"{normal_output_data_path}/normal_compressed_data.npy")
    adversarial_compressed_np = np.load(f"{adversarial_output_data_path}/adversarial_compressed_data.npy")

    # 将它们转换为PyTorch张量
    normal_compressed_tensor = torch.from_numpy(normal_compressed_np).float()
    adversarial_compressed_tensor = torch.from_numpy(adversarial_compressed_np).float()

    save_recovered_images(normal_compressed_tensor, normal_output_image_path)
    save_recovered_images(adversarial_compressed_tensor, adversarial_output_data_path)

if __name__ == '__main__':
    reconstruct("../../semantic_extraction/compress_data",
                "../../semantic_extraction/compress_data",
                "../../reconstruct_image_uap",
                "../../reconstruct_image")
