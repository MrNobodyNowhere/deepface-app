from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_base64_image(base64_string):
    """Convert base64 string to image array"""
    img_data = base64.b64decode(base64_string.split(',')[1] if ',' in base64_string else base64_string)
    img = Image.open(BytesIO(img_data))
    return np.array(img)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze face attributes"""
    try:
        data = request.json
        img_path = data.get('img')
        actions = data.get('actions', ['age', 'gender', 'emotion', 'race'])
        
        result = DeepFace.analyze(img_path=img_path, actions=actions, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/verify', methods=['POST'])
def verify():
    """Verify if two faces match"""
    try:
        data = request.json
        img1 = data.get('img1')
        img2 = data.get('img2')
        
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/represent', methods=['POST'])
def represent():
    """Get face embeddings"""
    try:
        data = request.json
        img_path = data.get('img')
        model_name = data.get('model_name', 'VGG-Face')
        
        result = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'DeepFace API Server',
        'endpoints': {
            '/analyze': 'POST - Analyze face attributes (age, gender, emotion, race)',
            '/verify': 'POST - Verify if two faces match',
            '/represent': 'POST - Get face embeddings',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)