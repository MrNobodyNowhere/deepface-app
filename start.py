from flask import Flask, request, jsonify
import os

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

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze face attributes"""
    try:
        DeepFace = get_deepface()
        data = request.json
        img_path = data.get('img')
        actions = data.get('actions', ['age', 'gender', 'emotion', 'race'])
        
        result = DeepFace.analyze(img_path=img_path, actions=actions, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify():
    """Verify if two faces match"""
    try:
        DeepFace = get_deepface()
        data = request.json
        img1 = data.get('img1')
        img2 = data.get('img2')
        
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/represent', methods=['POST'])
def represent():
    """Get face embeddings"""
    try:
        DeepFace = get_deepface()
        data = request.json
        img_path = data.get('img')
        model_name = data.get('model_name', 'VGG-Face')
        
        result = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting DeepFace API server on port {port}")
    app.run(host='0.0.0.0', port=port)