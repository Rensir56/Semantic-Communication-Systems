from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS  # 引入CORS扩展
import os
from utils.models import *
from utils.dataset import *
from utils.prepare import *
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# 为整个应用启用CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 路由：发送端对图片进行加工


@app.route('api/image/bake', methods=['POST'])
def bake_image():
    try:
        # 获取请求体中的数据
        data = request.json
        logging.info(f"Received data for baking: {data}")

        # 这里可以添加处理图片的代码
        # 例如：result = model.bake(data)
        rate = data.get('rate', 0.5)
        model_encoder = data.get(
            'model_encoder', 'path/to/default/encoder.pkl')
        model_classifier = data.get(
            'model_classifier', 'path/to/default/classifier.pkl')
        dataset_path = data.get('dataset_path', './dataset/mnist')
        output_data_path = data.get('output_data_path', './compress_data')
        os.makedirs(output_data_path, exist_ok=True)
        raw_dim = 28 * 28
        compression_rate = min((rate + 10) * 0.1, 1)
        channel = int(compression_rate * raw_dim)
        mlp_encoder, _ = load_models(model_encoder, model_classifier, channel)
        testset = mnist.MNIST(dataset_path, train=False,
                              transform=data_transform, download=True)
        test_data = DataLoader(testset, batch_size=128, shuffle=False)

        mlp_encoder.eval()
        with torch.no_grad():
            for im, label in test_data:
                im = Variable(im)
                label = Variable(label)
                out = mlp_encoder(im)

        compressed_np = out.detach().cpu().numpy()
        output_file = os.path.join(output_data_path, 'compressed_data.npy')
        np.save(output_file, compressed_np)

        # 返回处理结果
        return jsonify({"status": "success", "message": "Image baked successfully", "compressed_file_path": output_file}), 200
    except Exception as e:
        logging.error(f"Error baking image: {e}")
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
