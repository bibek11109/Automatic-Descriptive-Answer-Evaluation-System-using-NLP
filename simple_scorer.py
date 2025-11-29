"""
Simple Rule-Based Scorer (No AI download needed)
Works immediately without BERT model
Can be replaced with AI model later
"""

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


class SimpleScorer:
    """Simple keyword-based scorer that works without AI model"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Basic text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words 
                if w not in self.stop_words and len(w) > 2]
        return set(words)
    
    def evaluate_answer(self, student_answer, reference_answer=None, max_score=60):
        """
        Simple evaluation based on word overlap
        Returns realistic scores without AI
        """
        # Preprocess texts
        student_words = self.preprocess(student_answer)
        
        if reference_answer:
            reference_words = self.preprocess(reference_answer)
        else:
            # If no reference, score based on answer quality indicators
            reference_words = student_words
        
        # Calculate overlap
        if len(reference_words) == 0:
            overlap = 0.5
        else:
            common = student_words.intersection(reference_words)
            overlap = len(common) / max(len(reference_words), 1)
        
        # Length score (much stricter - favor longer, more detailed answers)
        word_count = len(student_answer.split())  # Use original word count, not processed
        if word_count < 10:
            length_score = 0.1  # Very poor for very short answers
        elif word_count < 20:
            length_score = 0.2  # Poor for short answers
        elif word_count < 40:
            length_score = 0.4  # Below average for medium-short answers
        elif word_count < 60:
            length_score = 0.7  # Good for decent length
        elif word_count > 100:
            length_score = 1.0  # Excellent for long answers
        else:
            length_score = 0.8  # Good for medium-long answers
        
        # Combine scores (30% overlap, 70% length) - prioritize length more
        normalized_score = (overlap * 0.3) + (length_score * 0.7)
        
        # Much stricter scoring - no minimum floor
        normalized_score = min(max(normalized_score, 0.05), 0.95)
        
        final_score = normalized_score * max_score
        
        # Generate feedback
        feedback = self._generate_feedback(normalized_score)
        
        return {
            'final_score': round(final_score, 2),
            'normalized_score': round(normalized_score, 4),
            'percentage': f"{normalized_score * 100:.1f}%",
            'feedback': feedback,
            'max_score': max_score,
            'note': 'Using simple keyword matching (AI model available for more accurate scoring)'
        }
    
    def _generate_feedback(self, score):
        """Generate feedback based on score"""
        percentage = score * 100
        
        if percentage >= 85:
            return "Excellent work! Your answer is comprehensive and well-detailed."
        elif percentage >= 70:
            return "Good answer! You covered the main points. Consider adding more specific examples."
        elif percentage >= 55:
            return "Satisfactory answer. You have basic understanding but could expand on key concepts."
        elif percentage >= 40:
            return "Your answer shows some understanding but needs more detail and examples."
        elif percentage >= 25:
            return "Your answer is too short. Please provide at least 40-50 words for a meaningful evaluation."
        else:
            return "Your answer needs significant improvement. Please write at least 40-50 words with detailed explanations."


# Global instance
_scorer_instance = None

def get_scorer():
    """Get or create scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = SimpleScorer()
    return _scorer_instance

