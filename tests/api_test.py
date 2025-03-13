import requests
import json
import sys
import os

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# API endpoint
API_URL = "http://localhost:5000"  # Change if deployed elsewhere

def test_single_prediction():
    """Test predicting emotions for a single text"""
    text = "You do right, if you don't care then fuck 'em!"
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nSingle Text Prediction:")
        print(f"Text: '{result['text']}'")
        print("Emotions:")
        for emotion, value in result['emotions'].items():
            print(f"  - {emotion}: {value}")
        print("Dominant emotions:", ", ".join(result['dominant_emotions']) or "None")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_batch_prediction():
    """Test predicting emotions for multiple texts"""
    texts = [
        "You do right, if you don't care then fuck 'em!",
        "I'm so excited about this new project!",
        "The meeting went well today.",
        "I will destroy your reputation if you cross me.",
        "I can't wait to see the results, they should be fantastic!"
    ]
    
    response = requests.post(
        f"{API_URL}/predict-batch",
        json={"texts": texts}
    )
    
    if response.status_code == 200:
        results = response.json()
        print("\nBatch Prediction Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Text: '{result['text']}'")
            print("   Dominant emotions:", ", ".join(result['dominant_emotions']) or "None")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def get_available_emotions():
    """Get the list of emotions the API can detect"""
    response = requests.get(f"{API_URL}/emotions")
    
    if response.status_code == 200:
        emotions = response.json()['emotions']
        print("\nAvailable emotions for detection:", ", ".join(emotions))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_api_info():
    """Test getting API information"""
    response = requests.get(API_URL)
    
    if response.status_code == 200:
        info = response.json()
        print("\nAPI Information:")
        print(f"Message: {info['message']}")
        print("Available endpoints:")
        for endpoint, description in info['endpoints'].items():
            print(f"  - {endpoint}: {description}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Testing Emotion Classification API")
    
    # Test health check
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        print("API is healthy!")
    else:
        print("API health check failed. Make sure the API is running.")
        exit(1)
    
    # Run tests
    test_api_info()
    get_available_emotions()
    test_single_prediction()
    test_batch_prediction()