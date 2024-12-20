from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import glob
import logging
from utils.models import *
from utils.dataset import *
from utils.prepare import *
from utils.compress import compress
from utils.reconstruct import reconstruct
from utils.evaluate import evaluate_mnist_models
from utils.adversarial import adversarial
from black.black_compress import black_attack

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 配置路径
MODEL_DIR = os.path.join(current_dir, '..', 'semantic_extraction')
DATASET_DIR = os.path.join(MODEL_DIR, 'dataset', 'mnist')
BLACK_DIR = os.path.join(current_dir, '..', 'server', 'black')
OUTPUT_DIR = os.path.join(current_dir, 'output')
COMPRESS_DIR = os.path.join(OUTPUT_DIR, 'compress_data')
RECONSTRUCT_DIR = os.path.join(OUTPUT_DIR, 'reconstruct_image')

# 创建必要的目录
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RECONSTRUCT_DIR, exist_ok=True)

@app.route('/api/compress', methods=['POST'])
def handle_compress():
    try:
        logging.info("Starting compression process...")
        os.makedirs(COMPRESS_DIR, exist_ok=True)
        
        # 执行压缩
        compress(
            rate=1.0,
            model_encoder=os.path.join(MODEL_DIR, 'MLP_MNIST_encoder_combining_1.000000.pkl'),
            model_classifier=os.path.join(MODEL_DIR, 'MLP_MNIST.pkl'),
            dataset_path=DATASET_DIR,
            output_data_path=COMPRESS_DIR
        )
        
        # 检查压缩文件是否生成
        compressed_file = os.path.join(COMPRESS_DIR, 'compressed_data.npy')
        if not os.path.exists(compressed_file):
            raise Exception("压缩文件未生成")
            
        return jsonify({
            'status': 'success',
            'message': '图片压缩成功'
        })
        
    except Exception as e:
        logging.error(f"Compression error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
       
@app.route('/api/reconstruct-and-evaluate', methods=['POST'])
def handle_reconstruct_and_evaluate():
    try:
        uap_mode = request.args.get('uapMode', 'none')
        logging.info(f"Starting reconstruction with UAP mode: {uap_mode}")
        
        if uap_mode == 'black':
            try:
                # 执行黑盒攻击
                black_dir = os.path.join(OUTPUT_DIR, 'reconstruct_image', 'black')
                os.makedirs(black_dir, exist_ok=True)
                
                logging.info(f"Black box attack directory: {black_dir}")
                logging.info(f"Surrogate model path: {os.path.join(BLACK_DIR, '_surrogate_model.pth')}")
                logging.info(f"MLP classifier path: {os.path.join(BLACK_DIR, 'mlp_classifier.pth')}")
                
                # 传入正确的输出路径
                black_result = black_attack(
                    surrogate_model_path=os.path.join(BLACK_DIR, '_surrogate_model.pth'),
                    mlp_classifier_path=os.path.join(BLACK_DIR, 'mlp_classifier.pth'),
                    output_dir=black_dir
                )
                
                logging.info("Black box attack completed")
                
                # 读取黑盒攻击结果图片
                black_box_images = []
                image_files = sorted(glob.glob(os.path.join(black_dir, 'adversarial_*.png')),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                logging.info(f"Found {len(image_files)} image files")
                
                for image_path in image_files:
                    with open(image_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        black_box_images.append(f'data:image/png;base64,{img_data}')
                
                if not black_box_images:
                    raise Exception("无法读取黑盒攻击图片")
                    
                return jsonify({
                    'status': 'success',
                    'reconstructed_images': black_box_images,
                    'accuracy': f"{black_result['accuracy']:.2f}",
                    'predictions': black_result['predictions'],
                    'true_labels': black_result['true_labels']
                })
            except Exception as e:
                logging.error(f"Black box attack error: {str(e)}")
                raise Exception(f"黑盒攻击失败: {str(e)}")
            
        elif uap_mode == 'white':
            # 检查压缩文件是否存在
            compressed_file = os.path.join(COMPRESS_DIR, 'compressed_data.npy')
            if not os.path.exists(compressed_file):
                raise Exception("未找到压缩数据，请先发送图片")
            
            # 定义UAP和普通重构的目录
            uap_dir = os.path.join(RECONSTRUCT_DIR, 'white')
            os.makedirs(uap_dir, exist_ok=True)
            
            # 清理目录
            for f in glob.glob(os.path.join(uap_dir, '*')):
                os.remove(f)
                
            # 执行对抗样本生成
            adversarial(
                rate=1.0,
                model_encoder=os.path.join(MODEL_DIR, 'MLP_MNIST_encoder_combining_1.000000.pkl'),
                model_classifier=os.path.join(MODEL_DIR, 'MLP_MNIST.pkl'),
                dataset_path=DATASET_DIR,
                output_data_path=COMPRESS_DIR
            )
            
            # UAP重构
            reconstruct(
                output_data_path=COMPRESS_DIR,
                output_image_path=uap_dir,
                uap=True
            )
            
            # 评估UAP结果
            uap_results = evaluate_mnist_models(
                rate=1.0,
                model_encoder=os.path.join(MODEL_DIR, 'MLP_MNIST_encoder_combining_1.000000.pkl'),
                model_classifier=os.path.join(MODEL_DIR, 'MLP_MNIST.pkl'),
                dataset_path=DATASET_DIR,
                output_image_path=uap_dir,
                output_data_path=COMPRESS_DIR
            )
            
            # 读取UAP重构图片
            white_box_images = []
            image_files = sorted(glob.glob(os.path.join(uap_dir, '*.jpg')) + 
                               glob.glob(os.path.join(uap_dir, '*.png')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            for image_path in image_files:
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    white_box_images.append(f'data:image/{"png" if image_path.endswith(".png") else "jpeg"};base64,{img_data}')
            
            return jsonify({
                'status': 'success',
                'reconstructed_images': white_box_images,
                'accuracy': f"{uap_results['accuracy'] * 100:.2f}",
                'predictions': uap_results['predictions'],
                'true_labels': uap_results['true_labels']
            })
            
        else:
            # 检查压缩文件是否���在
            compressed_file = os.path.join(COMPRESS_DIR, 'compressed_data.npy')
            if not os.path.exists(compressed_file):
                raise Exception("未找到压缩数据，请先发送图片")
            
            # 定义普通重构的目录
            normal_dir = os.path.join(RECONSTRUCT_DIR, 'origin')
            os.makedirs(normal_dir, exist_ok=True)
            
            # 清理目录
            for f in glob.glob(os.path.join(normal_dir, '*')):
                os.remove(f)
                
            # 普通重构
            reconstruct(
                output_data_path=COMPRESS_DIR,
                output_image_path=normal_dir,
                uap=False
            )
            
            # 评估正常重构结果
            eval_results = evaluate_mnist_models(
                rate=1.0,
                model_encoder=os.path.join(MODEL_DIR, 'MLP_MNIST_encoder_combining_1.000000.pkl'),
                model_classifier=os.path.join(MODEL_DIR, 'MLP_MNIST.pkl'),
                dataset_path=DATASET_DIR,
                output_image_path=normal_dir,
                output_data_path=COMPRESS_DIR
            )
            
            # 读取正常重构图片
            normal_images = []
            image_files = sorted(glob.glob(os.path.join(normal_dir, '*.jpg')) + 
                               glob.glob(os.path.join(normal_dir, '*.png')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            for image_path in image_files:
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    normal_images.append(f'data:image/{"png" if image_path.endswith(".png") else "jpeg"};base64,{img_data}')
            
            return jsonify({
                'status': 'success',
                'reconstructed_images': normal_images,
                'accuracy': f"{eval_results['accuracy'] * 100:.2f}",
                'predictions': eval_results['predictions'],
                'true_labels': eval_results['true_labels']
            })
            
    except Exception as e:
        logging.error(f"Reconstruct and evaluate error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # 确保在正确的目录下运行
    os.chdir(current_dir)
    app.run(host='127.0.0.1', port=5000, debug=True)