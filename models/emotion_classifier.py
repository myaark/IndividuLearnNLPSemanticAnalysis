import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class EmotionClassifier:
    """Emotion classification model handler"""
    
    def __init__(self, model_path='emotion_classifier_model.pkl', labels_path='emotion_labels.pkl'):
        """Initialize the emotion classifier"""
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.emotion_labels = None
        
        # Load or train the model
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load the model if it exists, otherwise train a new one"""
        try:
            # Try to load the model from disk
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.labels_path, 'rb') as f:
                self.emotion_labels = pickle.load(f)
            print("Model loaded successfully from disk.")
        except FileNotFoundError:
            print("Training new model...")
            self._train_model()
    
    def _train_model(self, csv_path = os.path.join(os.path.dirname(__file__), "goemotionsfinal.csv")):
        """Train a new emotion classification model"""
        # Load data
        df = pd.read_csv(csv_path)
        X = df['text']
        y = df[['anger', 'annoyance', 'neutral', 'joy', 'optimism']]
        self.emotion_labels = y.columns.tolist()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultiOutputClassifier(MultinomialNB()))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.labels_path, 'wb') as f:
            pickle.dump(self.emotion_labels, f)
        
        print("Model trained and saved successfully.")
    
    def predict(self, text):
        """Predict emotions for a single text"""
        if not self.model:
            raise ValueError("Model not loaded. Initialize the classifier first.")
            
        prediction = self.model.predict([text])[0]
        
        # Create emotion dictionary
        emotions_dict = {emotion: int(pred) for emotion, pred in 
                      zip(self.emotion_labels, prediction)}
        
        # Get dominant emotions
        dominant = self._get_dominant_emotions(emotions_dict)
        
        return {
            "text": text,
            "emotions": emotions_dict,
            "dominant_emotions": dominant
        }
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts"""
        if not self.model:
            raise ValueError("Model not loaded. Initialize the classifier first.")
            
        predictions = self.model.predict(texts)
        
        results = []
        for i, text in enumerate(texts):
            emotions_dict = {emotion: int(pred) for emotion, pred in 
                          zip(self.emotion_labels, predictions[i])}
            dominant = self._get_dominant_emotions(emotions_dict)
            
            results.append({
                "text": text,
                "emotions": emotions_dict,
                "dominant_emotions": dominant
            })
        
        return results
    
    def get_emotions(self):
        """Return the list of emotions this model can detect"""
        return self.emotion_labels
    
    @staticmethod
    def _get_dominant_emotions(emotions_dict):
        """Returns a list of emotions that are predicted as present (1)"""
        return [emotion for emotion, value in emotions_dict.items() if value == 1]