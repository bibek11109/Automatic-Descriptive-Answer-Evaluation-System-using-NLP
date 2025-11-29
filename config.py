"""
Configuration for ADAES Backend
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'adaes-secret-key-change-in-production'
    
    # Database
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'adaes.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # Model
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'best_model.pt')
    SENTENCE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    MAX_SCORE = 60
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5173']

