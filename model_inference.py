"""
AI Model Inference for Answer Evaluation
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class OptimizedModel(nn.Module):
    """Model architecture (same as training)"""
    def __init__(self, input_dim=384):
        super(OptimizedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class TextPreprocessor:
    """Text preprocessing"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                 if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)


class ModelInference:
    """Model inference wrapper"""
    
    def __init__(self, model_path, sentence_model_name, device='cpu'):
        self.device = device
        self.preprocessor = TextPreprocessor()
        
        print(f"Loading AI model from: {model_path}")
        
        # Load sentence transformer
        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.sentence_model.to(device)
        
        # Load trained model
        self.model = OptimizedModel(384)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Model loaded! (R²: {checkpoint.get('val_r2', 0):.4f})")
    
    def evaluate_answer(self, student_answer, max_score=60):
        """
        Evaluate student answer and return score with feedback
        
        Args:
            student_answer: Student's answer text
            max_score: Maximum possible score
        
        Returns:
            dict with score, feedback, etc.
        """
        # Preprocess
        processed_text = self.preprocessor.preprocess(student_answer)
        
        # Get embedding
        embedding = self.sentence_model.encode([processed_text], convert_to_tensor=False)
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(embedding).to(self.device)
            prediction = self.model(input_tensor)
            normalized_score = prediction.item()
            final_score = normalized_score * max_score
        
        # Generate feedback
        feedback = self._generate_feedback(final_score, max_score, normalized_score)
        
        return {
            'final_score': round(final_score, 2),
            'normalized_score': round(normalized_score, 4),
            'percentage': f"{normalized_score * 100:.1f}%",
            'feedback': feedback,
            'max_score': max_score
        }
    
    def _generate_feedback(self, score, max_score, normalized):
        """Generate feedback based on score"""
        percentage = normalized * 100
        
        if percentage >= 85:
            return "Excellent work! Your answer demonstrates comprehensive understanding with clear explanations and relevant details."
        elif percentage >= 70:
            return "Good answer! You covered the main points well. Consider adding more specific examples or details to strengthen your response."
        elif percentage >= 55:
            return "Satisfactory answer. You have the basic understanding but could expand on key concepts and provide more detailed explanations."
        elif percentage >= 40:
            return "Your answer shows some understanding but needs significant improvement. Review the key concepts and provide more complete explanations."
        else:
            return "Your answer needs substantial improvement. Please review the material carefully and ensure you address all key points in the question."


# Global model instance (loaded once at startup)
_model_instance = None

def get_model_instance(model_path, sentence_model_name, device='cpu'):
    """Get or create model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelInference(model_path, sentence_model_name, device)
    return _model_instance

