from flask import Blueprint, request, jsonify
from models import EmotionClassifier
from models.utils import preprocess_text, validate_emotion_prediction_request, validate_batch_prediction_request

# Create a Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Create a global instance of the EmotionClassifier
emotion_classifier = EmotionClassifier()

@api_bp.route('/', methods=['GET'])
def index():
    """API home endpoint"""
    return jsonify({
        "message": "Welcome to the Emotion Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/": "This information",
            "/emotions": "Get available emotions for detection",
            "/predict": "Predict emotions for a single text",
            "/predict-batch": "Predict emotions for multiple texts",
            "/health": "Check API health"
        }
    })

@api_bp.route('/emotions', methods=['GET'])
def get_emotions():
    """Returns the list of emotions this API can detect"""
    try:
        emotions = emotion_classifier.get_emotions()
        return jsonify({"emotions": emotions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/predict', methods=['POST'])
def predict_emotions():
    """Predict emotions for a single text input"""
    try:
        data = request.get_json()
        
        # Validate request data
        is_valid, errors = validate_emotion_prediction_request(data)
        if not is_valid:
            return jsonify({"error": errors}), 400
        
        # Preprocess the text
        text = preprocess_text(data['text'])
        
        # Get prediction from model
        result = emotion_classifier.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Predict emotions for multiple texts"""
    try:
        data = request.get_json()
        
        # Validate request data
        is_valid, errors = validate_batch_prediction_request(data)
        if not is_valid:
            return jsonify({"error": errors}), 400
        
        # Preprocess each text
        texts = [preprocess_text(text) for text in data['texts']]
        
        # Get predictions from model
        results = emotion_classifier.predict_batch(texts)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Make a simple prediction to check if model is working
        emotion_classifier.predict("Test message")
        return jsonify({"status": "healthy"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500