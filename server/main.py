from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS  # 引入CORS扩展
import os
from utils.models import *
from utils.dataset import *
from utils.prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import cv2
import numpy as np

import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# 为整个应用启用CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 路由：发送端对图片进行加工

# @app.route('api/image/bake', methods=['POST'])
# def bake_image():
#     try:
#         # 获取请求体中的数据
#         data = request.json
#         logging.info(f"Received data for baking: {data}")

#         # 这里可以添加处理图片的代码
#         # 例如：result = model.bake(data)
#         rate = data.get('rate', 0.5)
#         model_encoder = data.get(
#             'model_encoder', 'path/to/default/encoder.pkl')
#         model_classifier = data.get(
#             'model_classifier', 'path/to/default/classifier.pkl')
#         dataset_path = data.get('dataset_path', './dataset/mnist')
#         output_data_path = data.get('output_data_path', './compress_data')
#         os.makedirs(output_data_path, exist_ok=True)
#         raw_dim = 28 * 28
#         compression_rate = min((rate + 10) * 0.1, 1)
#         channel = int(compression_rate * raw_dim)
#         mlp_encoder, _ = load_models(model_encoder, model_classifier, channel)
#         testset = mnist.MNIST(dataset_path, train=False,
#                               transform=data_transform, download=True)
#         test_data = DataLoader(testset, batch_size=128, shuffle=False)

#         mlp_encoder.eval()
#         with torch.no_grad():
#             for im, label in test_data:
#                 im = Variable(im)
#                 label = Variable(label)
#                 out = mlp_encoder(im)

#         compressed_np = out.detach().cpu().numpy()
#         output_file = os.path.join(output_data_path, 'compressed_data.npy')
#         np.save(output_file, compressed_np)

#         # 返回处理结果
#         return jsonify({"status": "success", "message": "Image baked successfully", "compressed_file_path": output_file}), 200
#     except Exception as e:
#         logging.error(f"Error baking image: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500


# 新的函数来处理正常压缩和对抗压缩
@app.route('api/image/bake_with_adversarial', methods=['POST'])
def bake_with_adversarial():
    try:
        # 获取请求体中的数据
        data = request.json
        logging.info(f"Received data for baking with adversarial attack: {data}")

        # 获取参数
        rate = data.~~('rate', 0.5)
        model_encoder = data.get('model_encoder', 'path/to/default/encoder.pkl')
        model_classifier = data.get('model_classifier', 'path/to/default/classifier.pkl')
        dataset_path = data.get('dataset_path', './dataset/mnist')
        output_data_path = data.get('output_data_path', './compress_data')

        # 创建输出路径
        os.makedirs(output_data_path, exist_ok=True)

        # 设置压缩维度
        raw_dim = 28 * 28
        compression_rate = min((rate + 10) * 0.1, 1)
        channel = int(compression_rate * raw_dim)
        
        # 加载预训练的编码器模型
        mlp_encoder, _ = load_models(model_encoder, model_classifier, channel)

        # 加载MNIST数据集
        testset = mnist.MNIST(dataset_path, train=False, transform=data_transform, download=True)
        test_data = DataLoader(testset, batch_size=128, shuffle=False)

        # 加载UAP（对抗样本扰动）
        uap = cv2.imread('../uap/apply_uap/uap_single_under_CIFAR100.png')  # 读取UAP
        uap = uap.astype(np.float32)  # 转为float32

        # 对UAP进行归一化
        uap = uap / 255.0  # 将UAP归一化到[0, 1]

        # 控制扰动强度
        epsilon = 0.05  # 可调整该参数以控制扰动的强度
        uap = uap * epsilon

        mlp_encoder.eval()  # 设置模型为评估模式
        
        # 用于存储两种压缩特征
        normal_compressed_features = []
        adversarial_compressed_features = []

        with torch.no_grad():  # 关闭梯度计算，提高效率
            for im, label in test_data:
                im = Variable(im)  # 转为Variable（Tensor）
                label = Variable(label)

                # 保存正常压缩特征
                normal_compressed_feature = mlp_encoder(im)  # 正常压缩
                normal_compressed_features.append(normal_compressed_feature.detach().cpu().numpy())

                # 将UAP添加到图像上
                # 调整UAP尺寸与当前图像大小一致
                uap_resized = cv2.resize(uap, (im.shape[2], im.shape[1]))  # 调整UAP尺寸
                uap_resized = torch.tensor(uap_resized).permute(2, 0, 1).unsqueeze(0)  # 转为Tensor，调整通道顺序

                # 将UAP添加到图像上
                adversarial_image = im + uap_resized
                adversarial_image = torch.clamp(adversarial_image, 0, 1)  # 保证图像值在[0, 1]之间
                
                # 对抗样本经过编码器进行特征提取
                adversarial_compressed_feature = mlp_encoder(adversarial_image)  # 通过编码器获取压缩特征
                adversarial_compressed_features.append(adversarial_compressed_feature.detach().cpu().numpy())

        # 将两种压缩特征保存到.npy文件
        normal_compressed_np = np.concatenate(normal_compressed_features, axis=0)
        adversarial_compressed_np = np.concatenate(adversarial_compressed_features, axis=0)

        normal_output_file = os.path.join(output_data_path, 'normal_compressed_data.npy')
        adversarial_output_file = os.path.join(output_data_path, 'adversarial_compressed_data.npy')

        np.save(normal_output_file, normal_compressed_np)
        np.save(adversarial_output_file, adversarial_compressed_np)

        # 返回两个文件路径
        return jsonify({
            "status": "success", 
            "message": "Successfully baked normal and adversarial data", 
            "normal_compressed_file_path": normal_output_file, 
            "adversarial_compressed_file_path": adversarial_output_file
        }), 200

    except Exception as e:
        logging.error(f"Error baking images with adversarial attack: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 路由：接收端向发射端请求图片
@app.route('api/image/request', methods=['POST'])
def send_image():
    try:
        # 获取请求体中的图片名称
        data = request.json
        image_name = data.get('image_name')
        if not image_name:
            return jsonify({"status": "error", "message": "Image name not provided"}), 400

        # 构造图片路径
        image_folder = os.path.join(os.getcwd(), 'image')
        image_path = os.path.join(image_folder, image_name)

        # 检查图片是否存在
        if not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Image not found"}), 404

        # 构建图片URL
        image_url = url_for('serve_image', file_name='image',
                            folder='', filename=image_name, _external=True)

        # 返回图片URL
        return jsonify({"status": "success", "image_url": image_url}), 200
    except Exception as e:
        logging.error(f"Error sending image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 构造文件路径
@app.route('/image/<path:file_name>/<path:folder>/<filename>')
def serve_image(file_name, folder, filename):
    try:
        # 构造文件路径
        file_path = os.path.join(file_name, folder, filename)
        logging.info(f"Serving image from: {file_path}")

        # 返回图像文件
        return send_from_directory(file_name, os.path.join(folder, filename))
    except Exception as e:
        logging.error(f"Error serving image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 运行服务器
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # 保证输出支持UTF-8编码
    app.run(host='127.0.0.1', port=3000, debug=True)
