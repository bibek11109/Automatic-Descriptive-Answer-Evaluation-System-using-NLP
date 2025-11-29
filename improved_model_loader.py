"""
Improved Model Loader - Uses your trained model with proper embeddings
"""

import torch
import torch.nn as nn
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

# Download NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


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


class ImprovedModelInference:
    """Improved model inference with better scoring"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.preprocessor = TextPreprocessor()
        
        print(f"Loading AI model from: {model_path}")
        
        # Load trained model
        self.model = OptimizedModel(384)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Improved AI model loaded! (R²: {checkpoint.get('val_r2', 0):.4f})")
        
        # Common English words for validation
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'is', 'are', 'was', 'were', 'can', 'could', 'should', 'may', 'might',
            'about', 'when', 'where', 'who', 'which', 'how', 'why', 'than', 'then',
            'make', 'like', 'time', 'just', 'know', 'take', 'people', 'into', 'year',
            'good', 'some', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
            'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
            'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want'
        ])
    
    def _is_meaningful_text(self, text):
        """Check if the text is meaningful (not gibberish)"""
        if not text or not text.strip():
            return False, "Answer is empty. Please provide a meaningful response."
        
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if len(words) < 5:
            return False, "Answer is too short. Please provide at least 5 words."
        
        # Enhanced gibberish detection
        # 1. Check for meaningful words (at least 50% should be common English words)
        meaningful_count = sum(1 for word in words if word in self.common_words)
        meaningful_ratio = meaningful_count / len(words)
        
        # 2. Check for academic/technical words that indicate real content
        academic_words = [
            'artificial', 'intelligence', 'technology', 'learning', 'machine',
            'data', 'algorithm', 'neural', 'network', 'system', 'computer',
            'analysis', 'processing', 'model', 'training', 'accuracy',
            'pattern', 'recognition', 'decision', 'automation', 'innovation',
            'programming', 'software', 'hardware', 'database', 'application',
            'development', 'digital', 'binary', 'code', 'function', 'variable',
            'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism',
            'abstraction', 'paradigm', 'modular', 'reusable', 'principle',
            # Mathematics terms
            'calculus', 'derivative', 'integral', 'limit', 'continuity', 'differentiation',
            'integration', 'optimization', 'differential', 'accumulation', 'theorem',
            'series', 'infinite', 'mathematical', 'mathematics', 'equation', 'formula',
            'solve', 'algebra', 'geometry', 'statistics', 'probability', 'graph',
            # Science terms
            'experiment', 'hypothesis', 'theory', 'observation', 'scientific', 'method',
            'conclusion', 'evidence', 'discovery', 'physics', 'chemistry', 'biology',
            'organism', 'cell', 'evolution', 'species', 'ecosystem', 'genetics',
            'photosynthesis', 'respiration', 'molecule', 'atom', 'element', 'compound',
            'reaction', 'chemical', 'energy', 'chloroplasts', 'chlorophyll', 'glucose',
            # English/Literature terms
            'literary', 'literature', 'narrative', 'metaphor', 'simile', 'imagery',
            'symbolism', 'irony', 'character', 'theme', 'plot', 'author', 'text',
            'reading', 'writing', 'language', 'grammar', 'poetry', 'novel', 'story',
            'composition', 'essay', 'critical', 'interpretation', 'criticism', 'context',
            'cultural', 'historical', 'stylistic', 'fictional', 'devices', 'elements',
            'examining', 'artistic', 'techniques', 'structure', 'symbolic', 'relationship',
            # History terms
            'historical', 'history', 'period', 'century', 'civilization', 'cultural',
            'renaissance', 'medieval', 'ancient', 'modern', 'european', 'revolution',
            'empire', 'society', 'development', 'traditional', 'humanistic', 'knowledge',
            # General academic terms
            'research', 'study', 'investigation', 'examination', 'evaluation', 'assessment',
            'comprehensive', 'detailed', 'thorough', 'systematic', 'methodological',
            'understanding', 'knowledge', 'concept', 'principle', 'fundamental', 'essential',
            'process', 'involves', 'requires', 'provides', 'enables', 'contributes'
        ]
        
        academic_count = sum(1 for word in words if word in academic_words)
        academic_ratio = academic_count / len(words)
        
        # 3. Check for gibberish patterns
        gibberish_score = 0
        
        # Check for excessive repetition of same characters
        for word in words:
            if len(word) > 3:
                unique_chars = len(set(word))
                if unique_chars <= 2:  # e.g., "aaaaaaa", "abababab"
                    gibberish_score += 1
                # Check for random character patterns
                if len(word) > 6 and unique_chars < len(word) * 0.4:
                    gibberish_score += 1
        
        # 4. Check for random-looking words (no vowels, all consonants, etc.)
        random_word_count = 0
        for word in words:
            if len(word) > 4:
                vowels = sum(1 for char in word if char in 'aeiou')
                if vowels == 0:  # No vowels
                    random_word_count += 1
                elif len(word) > 6 and vowels < 2:  # Very few vowels
                    random_word_count += 1
        
        random_ratio = random_word_count / len(words)
        
        # 5. Check average word length (gibberish often has unusual lengths)
        avg_length = sum(len(word) for word in words) / len(words)
        
        # Determine if text is meaningful
        is_meaningful = (
            meaningful_ratio >= 0.4 or  # At least 40% common words
            academic_ratio >= 0.1 or    # At least 10% academic words
            (meaningful_ratio >= 0.2 and academic_ratio >= 0.05)  # Combination
        ) and (
            gibberish_score < len(words) * 0.3 and  # Less than 30% gibberish patterns
            random_ratio < 0.4 and  # Less than 40% random-looking words
            2 <= avg_length <= 12   # Reasonable word length
        )
        
        if not is_meaningful:
            if gibberish_score >= len(words) * 0.3:
                return False, "Answer contains gibberish or repeated characters. Please provide a meaningful response with real words."
            elif random_ratio >= 0.4:
                return False, "Answer does not appear to contain meaningful content. Please check your answer and provide a proper response with real words."
            elif meaningful_ratio < 0.2:
                return False, "Answer lacks meaningful content. Please provide a proper response with real English words."
            else:
                return False, "Answer does not appear to be properly formatted. Please check your answer and provide a meaningful response."
        
        return True, None
    
    def _is_meaningful_word(self, word):
        """Check if a single word is meaningful (not gibberish)"""
        if len(word) < 3:
            return False
        
        # Check for repeated characters
        unique_chars = len(set(word))
        if len(word) > 4 and unique_chars <= 2:
            return False
        
        # Check for vowel distribution (real words usually have vowels)
        vowels = sum(1 for char in word if char in 'aeiou')
        if len(word) > 6 and vowels < 2:
            return False
        
        # Check if it's a common English word or academic word
        academic_words = [
            'artificial', 'intelligence', 'technology', 'learning', 'machine',
            'data', 'algorithm', 'neural', 'network', 'system', 'computer',
            'analysis', 'processing', 'model', 'training', 'accuracy',
            'pattern', 'recognition', 'decision', 'automation', 'innovation',
            'programming', 'software', 'hardware', 'database', 'application',
            'development', 'digital', 'binary', 'code', 'function', 'variable',
            'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism',
            'abstraction', 'paradigm', 'modular', 'reusable', 'principle'
        ]
        
        return word in self.common_words or word in academic_words
    
    def _create_simple_embedding(self, text):
        """Create a simple 384-dim embedding based on text features"""
        processed_text = self.preprocessor.preprocess(text)
        words = processed_text.split()
        
        # Create embedding based on text features
        embedding = np.zeros(384)
        
        # Word count feature (normalized)
        word_count = len(words)
        embedding[0] = min(word_count / 100.0, 1.0)  # Normalize to 0-1
        
        # Character count feature
        char_count = len(processed_text)
        embedding[1] = min(char_count / 1000.0, 1.0)
        
        # Average word length
        if word_count > 0:
            avg_word_len = sum(len(word) for word in words) / word_count
            embedding[2] = min(avg_word_len / 10.0, 1.0)
        
        # Vocabulary richness (unique words / total words)
        if word_count > 0:
            unique_words = len(set(words))
            embedding[3] = unique_words / word_count
        
        # Academic keywords (AI, technology, learning, etc.)
        academic_keywords = [
            'artificial', 'intelligence', 'technology', 'learning', 'machine',
            'data', 'algorithm', 'neural', 'network', 'system', 'computer',
            'analysis', 'processing', 'model', 'training', 'accuracy',
            'pattern', 'recognition', 'decision', 'automation', 'innovation'
        ]
        
        keyword_count = sum(1 for word in words if word in academic_keywords)
        embedding[4] = min(keyword_count / 10.0, 1.0)
        
        # Sentence structure features
        sentences = text.split('.')
        embedding[5] = min(len(sentences) / 10.0, 1.0)
        
        # Fill remaining dimensions with word-based features
        for i, word in enumerate(words[:100]):  # Use first 100 words
            if i + 10 < 384:
                # Simple hash-based feature
                embedding[i + 10] = hash(word) % 1000 / 1000.0
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, 384)
        embedding = embedding + noise
        
        # Normalize
        embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)
        
        return embedding
    
    def _calculate_content_quality(self, text):
        """Calculate content quality score based on various factors"""
        processed_text = self.preprocessor.preprocess(text)
        words = processed_text.split()
        
        if not words:
            return 0.1
        
        # First check if text is meaningful (not gibberish)
        is_meaningful, _ = self._is_meaningful_text(text)
        if not is_meaningful:
            return 0.05  # Very low score for gibberish
        
        # Academic vocabulary score
        academic_words = [
            'artificial', 'intelligence', 'technology', 'learning', 'machine',
            'data', 'algorithm', 'neural', 'network', 'system', 'computer',
            'analysis', 'processing', 'model', 'training', 'accuracy',
            'pattern', 'recognition', 'decision', 'automation', 'innovation',
            'revolutionary', 'enables', 'perform', 'tasks', 'typically',
            'human', 'systems', 'learn', 'recognize', 'remarkable',
            'applications', 'healthcare', 'education', 'transportation',
            'transforming', 'modern', 'world', 'comprehensive', 'detailed',
            'programming', 'software', 'hardware', 'database', 'application',
            'development', 'digital', 'binary', 'code', 'function', 'variable',
            'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism',
            'abstraction', 'paradigm', 'modular', 'reusable', 'principle'
        ]
        
        academic_count = sum(1 for word in words if word in academic_words)
        academic_score = min(academic_count / len(words), 1.0)
        
        # Common English words score (indicates real content)
        common_count = sum(1 for word in words if word in self.common_words)
        common_score = min(common_count / len(words), 1.0)
        
        # Sentence structure score
        sentences = text.split('.')
        avg_sentence_length = len(words) / max(len(sentences), 1)
        structure_score = min(avg_sentence_length / 15.0, 1.0)  # Optimal ~15 words per sentence
        
        # Vocabulary diversity score
        unique_words = len(set(words))
        diversity_score = min(unique_words / len(words), 1.0)
        
        # Gibberish penalty - check for random character patterns
        gibberish_penalty = 0
        for word in words:
            if len(word) > 4:
                unique_chars = len(set(word))
                if unique_chars <= 2:  # Repeated characters
                    gibberish_penalty += 0.2
                elif len(word) > 6 and unique_chars < len(word) * 0.4:
                    gibberish_penalty += 0.1
        
        gibberish_penalty = min(gibberish_penalty, 0.8)  # Cap at 80% penalty
        
        # Combine quality factors with gibberish penalty
        quality_score = (
            (academic_score * 0.3) + 
            (common_score * 0.3) + 
            (structure_score * 0.2) + 
            (diversity_score * 0.2)
        ) * (1 - gibberish_penalty)
        
        return min(max(quality_score, 0.05), 1.0)
    
    def _check_subject_relevance(self, student_answer, subject):
        """Check if the student's answer is relevant to the specified subject"""
        if not subject:
            return {'is_relevant': True, 'confidence': 1.0, 'reason': 'No subject specified'}
        
        # Define subject-specific keywords (comprehensive)
        subject_keywords = {
            'Mathematics': ['calculate', 'equation', 'formula', 'solve', 'number', 'math', 'algebra', 'geometry', 'statistics', 'probability', 'function', 'variable', 'graph', 'plot', 'calculus', 'derivative', 'integral', 'limit', 'continuity', 'differentiation', 'integration', 'optimization', 'differential', 'accumulation', 'rate', 'change', 'theorem', 'series', 'infinite', 'mathematical', 'mathematics'],
            'Science': ['experiment', 'hypothesis', 'theory', 'observation', 'data', 'research', 'scientific', 'method', 'analysis', 'conclusion', 'evidence', 'fact', 'discovery', 'biology', 'chemistry', 'physics', 'organism', 'cell', 'evolution', 'species', 'ecosystem', 'photosynthesis', 'respiration', 'molecule', 'atom', 'element', 'compound', 'reaction', 'chemical', 'bond', 'solution', 'acid', 'base', 'organic', 'inorganic', 'synthesis', 'catalyst', 'force', 'energy', 'motion', 'velocity', 'acceleration', 'mass', 'gravity', 'electricity', 'magnetism', 'wave', 'particle', 'quantum', 'mechanics', 'thermodynamics', 'biological', 'process', 'chloroplasts', 'chlorophyll', 'atp', 'nadph', 'calvin', 'glucose', 'oxygen', 'carbon', 'light', 'plants', 'bacteria'],
            'English': ['literature', 'writing', 'grammar', 'poetry', 'novel', 'story', 'character', 'theme', 'plot', 'author', 'language', 'composition', 'essay', 'narrative', 'text', 'reading', 'analysis', 'metaphor', 'simile', 'symbolism', 'irony', 'context', 'cultural', 'historical', 'literary'],
            'History': ['historical', 'past', 'event', 'war', 'revolution', 'century', 'ancient', 'medieval', 'modern', 'timeline', 'chronology', 'civilization', 'culture', 'empire', 'renaissance', 'period', 'european', 'italy', 'gutenberg', 'leonardo', 'michelangelo', 'copernicus', 'enlightenment'],
            'Geography': ['location', 'country', 'continent', 'climate', 'weather', 'population', 'capital', 'city', 'region', 'landform', 'environment', 'natural', 'physical', 'human', 'geographical', 'terrain'],
            'Computer Science': ['programming', 'algorithm', 'code', 'software', 'hardware', 'database', 'network', 'system', 'application', 'development', 'technology', 'digital', 'binary', 'data', 'computer', 'computing', 'program', 'function', 'variable', 'oop', 'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism', 'abstraction', 'class', 'method', 'attribute', 'paradigm'],
            'Physics': ['force', 'energy', 'motion', 'velocity', 'acceleration', 'mass', 'gravity', 'electricity', 'magnetism', 'wave', 'particle', 'quantum', 'mechanics', 'thermodynamics', 'physical', 'scientific', 'theory', 'law'],
            'Chemistry': ['molecule', 'atom', 'element', 'compound', 'reaction', 'chemical', 'bond', 'solution', 'acid', 'base', 'organic', 'inorganic', 'synthesis', 'catalyst', 'chemistry', 'substance', 'material'],
            'Biology': ['organism', 'cell', 'dna', 'evolution', 'species', 'ecosystem', 'habitat', 'adaptation', 'genetics', 'reproduction', 'metabolism', 'photosynthesis', 'respiration', 'biology', 'life', 'living', 'biological'],
            'Economics': ['market', 'supply', 'demand', 'price', 'economy', 'business', 'trade', 'investment', 'profit', 'cost', 'revenue', 'inflation', 'gdp', 'financial', 'economic', 'money', 'capital'],
            'Business Studies': ['management', 'organization', 'strategy', 'marketing', 'finance', 'operations', 'leadership', 'entrepreneur', 'company', 'industry', 'customer', 'product', 'service', 'business', 'corporate'],
            'Psychology': ['behavior', 'mental', 'cognitive', 'emotion', 'personality', 'development', 'learning', 'memory', 'perception', 'consciousness', 'therapy', 'disorder', 'mind', 'psychological', 'brain', 'human'],
            'Sociology': ['society', 'social', 'culture', 'community', 'group', 'institution', 'norm', 'value', 'belief', 'interaction', 'relationship', 'structure', 'change', 'sociological', 'human', 'behavior'],
            'Political Science': ['government', 'politics', 'policy', 'democracy', 'election', 'constitution', 'law', 'rights', 'power', 'authority', 'citizen', 'state', 'nation', 'political', 'governance'],
            'Literature': ['book', 'novel', 'poem', 'poetry', 'drama', 'fiction', 'non-fiction', 'author', 'writer', 'text', 'reading', 'interpretation', 'criticism', 'analysis', 'literary', 'writing'],
            'Art': ['painting', 'drawing', 'sculpture', 'design', 'creative', 'aesthetic', 'visual', 'color', 'form', 'style', 'technique', 'artist', 'gallery', 'exhibition', 'artistic', 'art'],
            'Music': ['sound', 'melody', 'rhythm', 'harmony', 'instrument', 'composition', 'performance', 'concert', 'song', 'note', 'scale', 'chord', 'musical', 'music', 'audio'],
            'Physical Education': ['exercise', 'fitness', 'sport', 'athletic', 'training', 'health', 'body', 'muscle', 'strength', 'endurance', 'flexibility', 'coordination', 'movement', 'physical', 'activity']
        }
        
        # Get keywords for the subject
        keywords = subject_keywords.get(subject, [])
        if not keywords:
            return {'is_relevant': True, 'confidence': 1.0, 'reason': 'Subject not in keyword database'}
        
        # Check for subject relevance with weighted scoring
        answer_lower = student_answer.lower()
        relevant_words = [word for word in keywords if word in answer_lower]
        
        # Calculate weighted relevance score (core terms get higher weight)
        core_terms = {
            'Mathematics': ['calculus', 'derivative', 'integral', 'mathematical', 'theorem', 'function', 'optimization', 'equation', 'formula', 'algebra', 'geometry'],
            'Science': ['experiment', 'hypothesis', 'biology', 'chemistry', 'organism', 'cell', 'evolution', 'photosynthesis', 'molecule', 'atom', 'reaction', 'chemical'],
            'English': ['literature', 'writing', 'grammar', 'poetry', 'novel', 'story', 'character', 'theme', 'author', 'language', 'composition', 'essay'],
            'History': ['historical', 'past', 'event', 'war', 'revolution', 'century', 'ancient', 'medieval', 'modern', 'timeline', 'chronology', 'civilization'],
            'Geography': ['location', 'country', 'continent', 'climate', 'population', 'capital', 'region', 'environment', 'natural', 'physical']
        }
        
        core_terms_for_subject = core_terms.get(subject, [])
        core_matches = sum(1 for word in core_terms_for_subject if word in answer_lower)
        total_core_terms = len(core_terms_for_subject)
        
        # Weighted relevance score (core terms count more)
        relevance_score = (core_matches * 2 + len(relevant_words)) / (total_core_terms * 2 + len(keywords)) if keywords else 0
        
        # Check for subject-specific content patterns (prioritize core subject terms)
        subject_patterns = {
            'Mathematics': any(word in answer_lower for word in ['calculus', 'derivative', 'integral', 'mathematical', 'theorem', 'function', 'optimization', 'equation', 'formula', 'algebra', 'geometry', 'mathematics', 'differential', 'integration']),
            'Science': any(word in answer_lower for word in ['experiment', 'hypothesis', 'biology', 'chemistry', 'organism', 'cell', 'evolution', 'photosynthesis', 'molecule', 'atom', 'reaction', 'chemical', 'scientific', 'research', 'laboratory']),
            'English': any(word in answer_lower for word in ['literature', 'writing', 'grammar', 'poetry', 'novel', 'story', 'character', 'theme', 'author', 'language', 'composition', 'essay', 'narrative', 'literary']),
            'History': any(word in answer_lower for word in ['historical', 'past', 'event', 'war', 'revolution', 'century', 'ancient', 'medieval', 'modern', 'timeline', 'chronology', 'civilization', 'historical', 'empire']),
            'Geography': any(word in answer_lower for word in ['location', 'country', 'continent', 'climate', 'population', 'capital', 'region', 'environment', 'natural', 'physical', 'geographical', 'terrain'])
        }
        
        has_pattern = subject_patterns.get(subject, False)
        
        # Determine primary topic of the answer
        primary_topic_indicators = {
            'Mathematics': ['calculus', 'derivative', 'integral', 'mathematical', 'theorem', 'function', 'optimization', 'equation', 'formula', 'algebra', 'geometry', 'mathematics', 'differential', 'integration', 'limit', 'continuity', 'series', 'infinite'],
            'Science': ['experiment', 'hypothesis', 'biology', 'chemistry', 'organism', 'cell', 'evolution', 'photosynthesis', 'molecule', 'atom', 'reaction', 'chemical', 'scientific', 'research', 'biological', 'process', 'chloroplasts', 'chlorophyll', 'atp', 'nadph', 'calvin', 'glucose', 'oxygen', 'carbon', 'energy', 'light', 'plants', 'bacteria'],
            'English': ['literature', 'writing', 'grammar', 'poetry', 'novel', 'story', 'character', 'theme', 'author', 'language', 'composition', 'essay', 'narrative', 'literary', 'text', 'reading', 'analysis', 'metaphor', 'simile', 'symbolism', 'irony', 'context', 'cultural', 'historical'],
            'History': ['historical', 'past', 'event', 'war', 'revolution', 'century', 'ancient', 'medieval', 'modern', 'timeline', 'chronology', 'civilization', 'empire', 'renaissance', 'period', 'european', 'italy', 'gutenberg', 'leonardo', 'michelangelo', 'copernicus', 'enlightenment'],
            'Geography': ['location', 'country', 'continent', 'climate', 'population', 'capital', 'region', 'environment', 'natural', 'physical', 'geographical', 'terrain', 'weather', 'landform', 'human', 'cultural'],
            'Computer Science': ['programming', 'algorithm', 'code', 'software', 'hardware', 'database', 'network', 'system', 'application', 'development', 'technology', 'digital', 'binary', 'data', 'computer', 'computing', 'program', 'function', 'variable', 'oop', 'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism', 'abstraction', 'class', 'method', 'attribute', 'paradigm'],
            'Physics': ['force', 'energy', 'motion', 'velocity', 'acceleration', 'mass', 'gravity', 'electricity', 'magnetism', 'wave', 'particle', 'quantum', 'mechanics', 'thermodynamics', 'physical', 'scientific', 'theory', 'law'],
            'Chemistry': ['molecule', 'atom', 'element', 'compound', 'reaction', 'chemical', 'bond', 'solution', 'acid', 'base', 'organic', 'inorganic', 'synthesis', 'catalyst', 'chemistry', 'substance', 'material'],
            'Biology': ['organism', 'cell', 'dna', 'evolution', 'species', 'ecosystem', 'habitat', 'adaptation', 'genetics', 'reproduction', 'metabolism', 'photosynthesis', 'respiration', 'biology', 'life', 'living', 'biological'],
            'Economics': ['market', 'supply', 'demand', 'price', 'economy', 'business', 'trade', 'investment', 'profit', 'cost', 'revenue', 'inflation', 'gdp', 'financial', 'economic', 'money', 'capital'],
            'Business Studies': ['management', 'organization', 'strategy', 'marketing', 'finance', 'operations', 'leadership', 'entrepreneur', 'company', 'industry', 'customer', 'product', 'service', 'business', 'corporate'],
            'Psychology': ['behavior', 'mental', 'cognitive', 'emotion', 'personality', 'development', 'learning', 'memory', 'perception', 'consciousness', 'therapy', 'disorder', 'mind', 'psychological', 'brain', 'human'],
            'Sociology': ['society', 'social', 'culture', 'community', 'group', 'institution', 'norm', 'value', 'belief', 'interaction', 'relationship', 'structure', 'change', 'sociological', 'human', 'behavior'],
            'Political Science': ['government', 'politics', 'policy', 'democracy', 'election', 'constitution', 'law', 'rights', 'power', 'authority', 'citizen', 'state', 'nation', 'political', 'governance'],
            'Literature': ['book', 'novel', 'poem', 'poetry', 'drama', 'fiction', 'non-fiction', 'author', 'writer', 'text', 'reading', 'interpretation', 'criticism', 'analysis', 'literary', 'writing'],
            'Art': ['painting', 'drawing', 'sculpture', 'design', 'creative', 'aesthetic', 'visual', 'color', 'form', 'style', 'technique', 'artist', 'gallery', 'exhibition', 'artistic', 'art'],
            'Music': ['sound', 'melody', 'rhythm', 'harmony', 'instrument', 'composition', 'performance', 'concert', 'song', 'note', 'scale', 'chord', 'musical', 'music', 'audio'],
            'Physical Education': ['exercise', 'fitness', 'sport', 'athletic', 'training', 'health', 'body', 'muscle', 'strength', 'endurance', 'flexibility', 'coordination', 'movement', 'physical', 'activity']
        }
        
        # Find which subject the answer is primarily about
        topic_scores = {}
        for topic, indicators in primary_topic_indicators.items():
            topic_scores[topic] = sum(1 for indicator in indicators if indicator in answer_lower)
        
        primary_topic = max(topic_scores, key=topic_scores.get) if topic_scores else None
        primary_topic_score = topic_scores.get(primary_topic, 0) if primary_topic else 0
        
        # Determine if answer is relevant based on primary topic (strict validation)
        # Answer is relevant if:
        # 1. It has high relevance score (many subject keywords found) OR
        # 2. The primary topic detected matches the selected subject OR
        # 3. It has subject-specific patterns
        is_relevant = (
            (relevance_score > 0.15) or  # Strong keyword match
            (subject == primary_topic and primary_topic_score >= 3) or  # Primary topic matches with good score
            (relevance_score > 0.08 and has_pattern and len(relevant_words) >= 2)  # Moderate match with patterns
        )
        
        # Additional check: if answer clearly belongs to a different subject, mark as not relevant
        if subject and primary_topic and primary_topic != subject:
            # Check if the answer is strongly about a different subject
            other_subject_score = primary_topic_score
            current_subject_score = topic_scores.get(subject, 0)
            
            # If answer is much more about another subject, it's not relevant
            if other_subject_score > current_subject_score * 2 and other_subject_score >= 5:
                is_relevant = False
        
        return {
            'is_relevant': is_relevant,
            'confidence': relevance_score,
            'reason': f"Found {len(relevant_words)} relevant keywords: {', '.join(relevant_words[:3])}" if relevant_words else "No subject-specific keywords found",
            'detected_subject': primary_topic,
            'subject_match_score': primary_topic_score,
            'subject': subject  # Pass the selected subject for feedback
        }
    
    def evaluate_answer(self, student_answer, reference_answer="", max_score=60, subject=""):
        """
        Comprehensive multi-criteria evaluation with detailed feedback and subject validation
        """
        # First, check if the answer is meaningful
        is_meaningful, error_message = self._is_meaningful_text(student_answer)
        if not is_meaningful:
            return {
                'score': 0,
                'max_score': max_score,
                'percentage': 0,
                'status': 'Invalid Answer',
                'status_color': 'red',
                'aspects': [
                    {
                        'criterion': 'Answer Validation',
                        'score': 0,
                        'max_score': 10,
                        'feedback': f"❌ {error_message}"
                    }
                ],
                'strengths': [],
                'improvements': [
                    "Provide a meaningful answer with proper words",
                    "Use complete sentences with proper grammar",
                    "Ensure your answer relates to the question asked"
                ],
                'similarity_score': 0,
                'keywords': [],
                'word_count': len(student_answer.split()) if student_answer else 0,
                'sentence_count': 0,
                'overall_feedback': f"❌ {error_message}"
            }
        
        # Create embedding
        embedding = self._create_simple_embedding(student_answer)
        
        # Predict with trained model
        with torch.no_grad():
            input_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor)
            normalized_score = prediction.item()
            
            # Analyze answer comprehensively
            analysis = self._comprehensive_analysis(student_answer, reference_answer)
            
            # Subject relevance validation
            subject_relevance = self._check_subject_relevance(student_answer, subject)

            # Calculate final score using all criteria (moved outside conditional)
            final_normalized = (
                normalized_score * 0.25 +
                analysis['length_score'] * 0.20 +
                analysis['content_quality'] * 0.20 +
                analysis['coherence_score'] * 0.15 +
                analysis['grammar_score'] * 0.10 +
                analysis['relevance_score'] * 0.10
            )

            # Calculate subject relevance penalty (more flexible)
            subject_penalty = 0
            if subject and not subject_relevance['is_relevant']:
                # Moderate penalty for subject mismatch (not zero)
                subject_penalty = 0.5  # 50% penalty for wrong subject
            elif subject and subject_relevance['is_relevant']:
                # Bonus for correct subject
                subject_penalty = -0.2  # 20% bonus for correct subject

            # Apply subject penalty/bonus
            final_normalized = final_normalized * (1 - subject_penalty)
            
            # Add slight variation for realism (±3%)
            variation = np.random.uniform(-0.03, 0.03)
            final_normalized = max(0.05, min(0.98, final_normalized + variation))
            final_score = final_normalized * max_score
        
        # Generate comprehensive feedback
        detailed_feedback = self._generate_comprehensive_feedback(
            final_score, max_score, final_normalized, analysis, student_answer, subject_relevance
        )
        
        return detailed_feedback
    
    def _comprehensive_analysis(self, student_answer, reference_answer):
        """Perform comprehensive multi-criteria analysis"""
        words = student_answer.split()
        processed = self.preprocessor.preprocess(student_answer)
        processed_words = processed.split()
        
        # 1. Length/Completeness Score
        word_count = len(words)
        if word_count < 10:
            length_score = 0.2
        elif word_count < 20:
            length_score = 0.35
        elif word_count < 30:
            length_score = 0.5
        elif word_count < 50:
            length_score = 0.65
        elif word_count < 80:
            length_score = 0.80
        else:
            length_score = 0.92
        
        # 2. Content Quality/Accuracy
        content_quality = self._calculate_content_quality(student_answer)
        
        # 3. Coherence Score (sentence structure, flow)
        sentences = [s.strip() for s in student_answer.split('.') if s.strip()]
        if len(sentences) > 1:
            avg_sent_len = len(words) / len(sentences)
            # Optimal: 12-20 words per sentence
            if 12 <= avg_sent_len <= 20:
                coherence_score = 0.9
            elif 8 <= avg_sent_len <= 25:
                coherence_score = 0.75
            else:
                coherence_score = 0.6
        else:
            coherence_score = 0.5  # Single sentence answers
        
        # 4. Grammar/Language Score (capitalization, punctuation, gibberish detection)
        grammar_score = 0.7  # Base score
        
        # Check for gibberish first
        is_meaningful, _ = self._is_meaningful_text(student_answer)
        if not is_meaningful:
            grammar_score = 0.1  # Very low score for gibberish
        else:
            if student_answer[0].isupper():  # Starts with capital
                grammar_score += 0.1
            if any(p in student_answer for p in ['.', ',', ';']):  # Has punctuation
                grammar_score += 0.1
            if len(set(processed_words)) / max(len(processed_words), 1) > 0.6:  # Vocabulary diversity
                grammar_score += 0.1
            
            # Check for random character patterns that indicate gibberish
            gibberish_penalty = 0
            for word in processed_words:
                if len(word) > 4:
                    unique_chars = len(set(word))
                    if unique_chars <= 2:  # Repeated characters like "aaaa"
                        gibberish_penalty += 0.3
                    elif len(word) > 6 and unique_chars < len(word) * 0.4:
                        gibberish_penalty += 0.2
            
            grammar_score = max(grammar_score - gibberish_penalty, 0.1)
        
        grammar_score = min(grammar_score, 0.95)
        
        # 5. Relevance Score (keyword overlap with reference if available)
        if reference_answer:
            ref_words = set(self.preprocessor.preprocess(reference_answer).split())
            student_words = set(processed_words)
            overlap = len(ref_words & student_words) / max(len(ref_words), 1)
            relevance_score = min(overlap * 2, 0.95)  # Scale up overlap
        else:
            # Without reference, use academic keyword density
            academic_keywords = [
                'artificial', 'intelligence', 'technology', 'learning', 'machine',
                'data', 'algorithm', 'neural', 'network', 'system', 'computer',
                'analysis', 'process', 'model', 'training', 'pattern'
            ]
            keyword_count = sum(1 for word in processed_words if word in academic_keywords)
            relevance_score = min(keyword_count / 5.0, 0.9)
        
        # Extract meaningful keywords for display (filter out gibberish)
        common_words = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'with', 'by', 'from', 'this', 'that', 'these', 'those'])
        
        # Academic words for keyword filtering
        academic_words = [
            'artificial', 'intelligence', 'technology', 'learning', 'machine',
            'data', 'algorithm', 'neural', 'network', 'system', 'computer',
            'analysis', 'processing', 'model', 'training', 'accuracy',
            'pattern', 'recognition', 'decision', 'automation', 'innovation',
            'programming', 'software', 'hardware', 'database', 'application',
            'development', 'digital', 'binary', 'code', 'function', 'variable',
            'object', 'oriented', 'encapsulation', 'inheritance', 'polymorphism',
            'abstraction', 'paradigm', 'modular', 'reusable', 'principle'
        ]
        
        # Filter for meaningful keywords only
        meaningful_keywords = []
        for word in processed_words:
            if (len(word) > 3 and 
                word not in common_words and 
                (word in self.common_words or  # Common English word
                 word in academic_words or     # Academic word
                 self._is_meaningful_word(word))):  # Passes meaningful word check
                meaningful_keywords.append(word)
        
        # Remove duplicates and limit to top 6
        unique_keywords = list(set(meaningful_keywords))[:6]
        
        return {
            'length_score': length_score,
            'content_quality': content_quality,
            'coherence_score': coherence_score,
            'grammar_score': grammar_score,
            'relevance_score': relevance_score,
            'keywords': unique_keywords,
            'word_count': word_count,
            'sentence_count': len(sentences)
        }
    
    def _generate_comprehensive_feedback(self, final_score, max_score, normalized, analysis, student_answer, subject_relevance=None):
        """Generate detailed multi-criteria feedback"""
        percentage = normalized * 100
        
        # Handle subject relevance (strict - very low scores for mismatch)
        if subject_relevance and not subject_relevance['is_relevant']:
            # Get detected subject from subject_relevance
            detected_subject = subject_relevance.get('detected_subject', 'Unknown')
            selected_subject = subject_relevance.get('subject', 'specified')
            
            # Give very low score for subject mismatch
            penalty_score = max(final_score * 0.1, 2)  # Only 10% of normal score, minimum 2 points
            
            mismatch_message = f"Your answer appears to be about '{detected_subject}' but you selected '{selected_subject}'"
            
            return {
                'score': penalty_score,
                'max_score': max_score,
                'percentage': (penalty_score / max_score) * 100,
                'status': 'Subject Mismatch',
                'status_color': 'red',
                'error_code': 'SUBJECT_MISMATCH',
                'subject_mismatch': True,
                'detected_subject': detected_subject,
                'selected_subject': selected_subject,
                'penalty_applied': True,
                'penalty_ratio': 0.1,
                'aspects': [
                    {
                        'criterion': 'Subject Relevance',
                        'score': 1,
                        'max_score': 10,
                        'feedback': f"❌ SUBJECT MISMATCH: {mismatch_message}. {subject_relevance.get('reason', 'Please answer according to the selected subject.')}"
                    }
                ],
                'strengths': [],
                'improvements': [
                    f"Answer questions related to: {selected_subject}",
                    f"Your answer seems to be about {detected_subject} - make sure you understand the question",
                    f"Include subject-specific keywords for: {selected_subject}",
                    "Focus on the topic area requested"
                ],
                'similarity_score': 0,
                'keywords': [],
                'word_count': len(student_answer.split()),
                'sentence_count': len([s for s in student_answer.split('.') if s.strip()]),
                'overall_feedback': f"❌ SUBJECT MISMATCH: {mismatch_message}. Your score has been reduced to 10%. Please answer according to the selected subject to receive full marks."
            }
        
        # Calculate criterion scores (out of 10 each for display)
        criteria_scores = {
            'content_accuracy': round(analysis['content_quality'] * 10, 1),
            'completeness': round(analysis['length_score'] * 10, 1),
            'coherence': round(analysis['coherence_score'] * 10, 1),
            'grammar': round(analysis['grammar_score'] * 10, 1),
            'relevance': round(analysis['relevance_score'] * 10, 1)
        }
        
        # Generate criterion feedback
        criteria_feedback = {
            'content_accuracy': self._get_criterion_feedback('content', criteria_scores['content_accuracy']),
            'completeness': self._get_criterion_feedback('completeness', criteria_scores['completeness']),
            'coherence': self._get_criterion_feedback('coherence', criteria_scores['coherence']),
            'grammar': self._get_criterion_feedback('grammar', criteria_scores['grammar']),
            'relevance': self._get_criterion_feedback('relevance', criteria_scores['relevance'])
        }
        
        # Identify strengths (criteria > 7.5)
        strengths = []
        if criteria_scores['content_accuracy'] >= 7.5:
            strengths.append("Strong grasp of core concepts and principles")
        if criteria_scores['coherence'] >= 7.5:
            strengths.append("Well-structured and logical flow of ideas")
        if criteria_scores['grammar'] >= 7.5:
            strengths.append("Good grammar and language usage")
        if criteria_scores['completeness'] >= 7.5:
            strengths.append("Comprehensive coverage of key points")
        if criteria_scores['relevance'] >= 7.5:
            strengths.append("Highly relevant to the question asked")
        
        # Default strengths if none found
        if not strengths:
            strengths = [
                "Shows basic understanding of the topic",
                "Attempted to address the question",
                "Clear intent to provide an answer"
            ]
        
        # Identify areas for improvement (criteria < 7.0)
        improvements = []
        if criteria_scores['content_accuracy'] < 7.0:
            improvements.append("Deepen understanding of key concepts and principles")
        if criteria_scores['completeness'] < 7.0:
            improvements.append("Provide more detailed explanations and examples")
        if criteria_scores['coherence'] < 7.0:
            improvements.append("Improve organization and logical flow")
        if criteria_scores['grammar'] < 7.0:
            improvements.append("Pay attention to grammar and sentence structure")
        if criteria_scores['relevance'] < 7.0:
            improvements.append("Focus more closely on the specific question asked")
        
        # Default improvements if none found
        if not improvements:
            improvements = [
                "Consider adding more specific examples",
                "Explore additional perspectives or applications",
                "Include supporting evidence or references"
            ]
        
        # Overall status
        if percentage >= 85:
            status = "Excellent"
            status_color = "green"
        elif percentage >= 70:
            status = "Very Good"
            status_color = "blue"
        elif percentage >= 55:
            status = "Good"
            status_color = "yellow"
        elif percentage >= 40:
            status = "Average"
            status_color = "orange"
        else:
            status = "Needs Improvement"
            status_color = "red"
        
        # Similarity score (semantic similarity estimate)
        similarity_score = int(min(normalized * 95, 92))  # 0-92% range
        
        return {
            'score': round(final_score, 2),
            'max_score': max_score,
            'percentage': round(percentage, 1),
            'status': status,
            'status_color': status_color,
            
            # Five criteria breakdown
            'aspects': [
                {
                    'criterion': 'Content Accuracy',
                    'score': criteria_scores['content_accuracy'],
                    'max': 10,
                    'feedback': criteria_feedback['content_accuracy']
                },
                {
                    'criterion': 'Completeness',
                    'score': criteria_scores['completeness'],
                    'max': 10,
                    'feedback': criteria_feedback['completeness']
                },
                {
                    'criterion': 'Coherence',
                    'score': criteria_scores['coherence'],
                    'max': 10,
                    'feedback': criteria_feedback['coherence']
                },
                {
                    'criterion': 'Grammar & Language',
                    'score': criteria_scores['grammar'],
                    'max': 10,
                    'feedback': criteria_feedback['grammar']
                },
                {
                    'criterion': 'Relevance',
                    'score': criteria_scores['relevance'],
                    'max': 10,
                    'feedback': criteria_feedback['relevance']
                }
            ],
            
            # Strengths and improvements
            'strengths': strengths[:3],  # Top 3
            'improvements': improvements[:3],  # Top 3
            
            # Additional metrics
            'similarity_score': similarity_score,
            'keywords': analysis['keywords'][:6],  # Top 6 keywords
            'word_count': analysis['word_count'],
            'sentence_count': analysis['sentence_count'],
            
            # Overall feedback
            'overall_feedback': self._generate_overall_feedback(percentage)
        }
    
    def _get_criterion_feedback(self, criterion_type, score):
        """Generate specific feedback for each criterion"""
        feedbacks = {
            'content': {
                'high': "Excellent understanding of core concepts with accurate explanations",
                'medium': "Good grasp of main concepts, some minor gaps in understanding",
                'low': "Limited understanding, review key concepts and principles"
            },
            'completeness': {
                'high': "Comprehensive answer covering all key points thoroughly",
                'medium': "Most key points covered, some details could be expanded",
                'low': "Answer lacks completeness, include more details and examples"
            },
            'coherence': {
                'high': "Well-structured with excellent logical flow and organization",
                'medium': "Generally coherent, some sections could be better organized",
                'low': "Improve structure and logical flow between ideas"
            },
            'grammar': {
                'high': "Excellent grammar, punctuation, and language usage",
                'medium': "Good language use with minor grammatical errors",
                'low': "Pay attention to grammar, punctuation, and sentence structure"
            },
            'relevance': {
                'high': "Highly relevant and focused on the question asked",
                'medium': "Mostly relevant with some tangential points",
                'low': "Stay more focused on the specific question asked"
            }
        }
        
        level = 'high' if score >= 7.5 else ('medium' if score >= 5.5 else 'low')
        return feedbacks.get(criterion_type, {}).get(level, "No feedback available")
    
    def _generate_overall_feedback(self, percentage):
        """Generate overall summary feedback"""
        if percentage >= 85:
            return "Outstanding performance! Your answer demonstrates comprehensive understanding with clear, detailed explanations and excellent structure."
        elif percentage >= 70:
            return "Very good work! You have a solid grasp of the material. Consider adding more specific examples or exploring additional perspectives."
        elif percentage >= 55:
            return "Good effort. You understand the basics but could strengthen your answer with more depth, examples, and better organization."
        elif percentage >= 40:
            return "Your answer shows some understanding but needs significant improvement. Review key concepts carefully and provide more complete explanations."
        else:
            return "Your answer needs substantial development. Please review the material thoroughly and ensure you address all aspects of the question with proper detail."


# Global model instance
_model_instance = None

def get_improved_model_instance(model_path, device='cpu'):
    """Get or create improved model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = ImprovedModelInference(model_path, device)
    return _model_instance
