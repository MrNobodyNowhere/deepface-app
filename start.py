from flask import Flask, request, jsonify
import os
import base64
import tempfile
from pathlib import Path

app = Flask(__name__)

# DO NOT IMPORT DEEPFACE HERE - ONLY IMPORT WHEN NEEDED
_deepface = None

def get_deepface():
    """Lazy import DeepFace only when API is called"""
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface

def save_base64_image(base64_str):
    """Save base64 image to temp file"""
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file.write(img_data)
    temp_file.close()
    return temp_file.name

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze face attributes"""
    temp_files = []
    try:
        DeepFace = get_deepface()
        data = request.json
        img_path = data.get('img')
        actions = data.get('actions', ['age', 'gender', 'emotion', 'race'])
        
        # Handle base64 images
        if img_path and img_path.startswith('data:image'):
            img_path = save_base64_image(img_path)
            temp_files.append(img_path)
        
        result = DeepFace.analyze(
            img_path=img_path, 
            actions=actions, 
            enforce_detection=False,
            silent=True
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

@app.route('/verify', methods=['POST'])
def verify():
    """Verify if two faces match"""
    temp_files = []
    try:
        DeepFace = get_deepface()
        data = request.json
        img1 = data.get('img1')
        img2 = data.get('img2')
        
        # Handle base64 images
        if img1 and img1.startswith('data:image'):
            img1 = save_base64_image(img1)
            temp_files.append(img1)
        if img2 and img2.startswith('data:image'):
            img2 = save_base64_image(img2)
            temp_files.append(img2)
        
        result = DeepFace.verify(
            img1_path=img1, 
            img2_path=img2, 
            enforce_detection=False,
            silent=True
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

@app.route('/represent', methods=['POST'])
def represent():
    """Get face embeddings"""
    temp_files = []
    try:
        DeepFace = get_deepface()
        data = request.json
        img_path = data.get('img')
        model_name = data.get('model_name', 'VGG-Face')
        
        # Handle base64 images
        if img_path and img_path.startswith('data:image'):
            img_path = save_base64_image(img_path)
            temp_files.append(img_path)
        
        result = DeepFace.represent(
            img_path=img_path, 
            model_name=model_name, 
            enforce_detection=False,
            silent=True
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'DeepFace API Server',
        'status': 'online',
        'endpoints': {
            '/analyze': 'POST - Analyze face attributes (age, gender, emotion, race)',
            '/verify': 'POST - Verify if two faces match',
            '/represent': 'POST - Get face embeddings',
            '/health': 'GET - Health check'
        },
        'usage': {
            'analyze': {
                'img': 'path/to/image.jpg or base64 string or URL',
                'actions': ['age', 'gender', 'emotion', 'race']
            },
            'verify': {
                'img1': 'path/to/image1.jpg or base64 string or URL',
                'img2': 'path/to/image2.jpg or base64 string or URL'
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting DeepFace API server on port {port}")
    print("Note: Models will be downloaded on first request")
    app.run(host='0.0.0.0', port=port, threaded=True)