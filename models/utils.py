# Utility functions for the emotion classification model

def preprocess_text(text):
    """
    Preprocess text before prediction
    This can be expanded with more sophisticated text cleaning if needed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning - remove extra whitespace
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def validate_emotion_prediction_request(data):
    """Validate the prediction request data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    if 'text' not in data:
        errors.append("No text field provided")
        return False, errors
    
    if not data['text'] or not isinstance(data['text'], str):
        errors.append("Text field must be a non-empty string")
        return False, errors
    
    return True, []

def validate_batch_prediction_request(data):
    """Validate the batch prediction request data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    if 'texts' not in data:
        errors.append("No texts field provided")
        return False, errors
    
    if not isinstance(data['texts'], list):
        errors.append("Texts field must be a list")
        return False, errors
    
    if not data['texts']:
        errors.append("Texts list cannot be empty")
        return False, errors
    
    for i, text in enumerate(data['texts']):
        if not isinstance(text, str) or not text:
            errors.append(f"Item {i} in texts list must be a non-empty string")
            return False, errors
    
    return True, []