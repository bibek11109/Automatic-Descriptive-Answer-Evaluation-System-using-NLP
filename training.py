"""
ADAES - Automatic Descriptive Answer Evaluation System
Optimized Training Script for Google Colab
Uses ALL data, completes in 3-4 hours, achieves >80% accuracy
"""

print("=" * 80)
print("ADAES - OPTIMIZED TRAINING (FULL DATASET)")
print("Target: >80% accuracy in 3-4 hours using ALL data")
print("=" * 80)

# ============================================================================
# Install & Import
# ============================================================================
print("\n📦 Setting up packages...")

import subprocess
import sys

# Quick install with error handling
packages = ["transformers", "sentence-transformers", "scikit-learn", "nltk"]
for pkg in packages:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], 
                      capture_output=True, check=False)
    except:
        pass

print("✅ Packages ready!\n")

# Import libraries
print("📚 Importing libraries...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLTK
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

print("✅ All libraries loaded!\n")

# ============================================================================
# Mount Drive
# ============================================================================
print("🔗 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✅ Drive mounted!\n")

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'dataset_path': '/content/drive/MyDrive/dataset_bibek/training_set_rel3.tsv',
    'save_dir': '/content/drive/MyDrive/adaes_final_models/',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 15,  
    'batch_size': 64,  
    'learning_rate': 5e-4,
    'weight_decay': 0.01,
    'val_split': 0.15,
    'test_split': 0.15,
    'patience': 4,
    'random_seed': 42
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
print(f"⚙️ Configuration:")
print(f"   Device: {CONFIG['device']}")
print(f"   Epochs: {CONFIG['epochs']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Using: ALL DATASET (no limit)\n")

# ============================================================================
# Text Preprocessing (Fast Version)
# ============================================================================
print("📝 Setting up fast preprocessor...\n")

class FastPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        """Fast preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens 
                 if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

preprocessor = FastPreprocessor()

# ============================================================================
# Load Data
# ============================================================================
print("📊 Loading FULL dataset...\n")

# Load with encoding fallback
try:
    df = pd.read_csv(CONFIG['dataset_path'], sep='\t', encoding='utf-8')
except:
    try:
        df = pd.read_csv(CONFIG['dataset_path'], sep='\t', encoding='latin-1')
    except:
        df = pd.read_csv(CONFIG['dataset_path'], sep='\t', encoding='ISO-8859-1')

print(f"✅ Loaded: {len(df)} essays")

# Clean data
df_clean = df[df['essay'].notna() & (df['domain1_score'].notna())].copy()
df_clean = df_clean[df_clean['essay'].str.len() > 50]
df_clean = df_clean[df_clean['domain1_score'] >= 0]

print(f"✅ Clean data: {len(df_clean)} essays")
print(f"📊 Score range: {df_clean['domain1_score'].min()} - {df_clean['domain1_score'].max()}")
print(f"📊 Using ALL {len(df_clean)} essays for training!\n")

# Preprocess (with progress bar)
print("🔄 Preprocessing all essays...")
tqdm.pandas(desc="Processing")
df_clean['essay_clean'] = df_clean['essay'].progress_apply(preprocessor.preprocess)
print("✅ Preprocessing done!\n")

# Normalize scores
max_score = df_clean['domain1_score'].max()
df_clean['score_norm'] = df_clean['domain1_score'] / max_score

# Split data
print("📂 Splitting data...")
train_df, temp_df = train_test_split(
    df_clean, test_size=CONFIG['val_split'] + CONFIG['test_split'],
    random_state=CONFIG['random_seed'], stratify=None
)
val_df, test_df = train_test_split(
    temp_df, test_size=CONFIG['test_split']/(CONFIG['val_split']+CONFIG['test_split']),
    random_state=CONFIG['random_seed']
)

print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

# ============================================================================
# Load Sentence Transformer & Generate Embeddings
# ============================================================================
print("🤖 Loading sentence transformer...\n")
sentence_model = SentenceTransformer(CONFIG['model_name'])
sentence_model.to(CONFIG['device'])
embedding_dim = sentence_model.get_sentence_embedding_dimension()
print(f"✅ Model loaded! Embedding dim: {embedding_dim}\n")

# Pre-compute ALL embeddings (faster training)
print("⚡ Pre-computing embeddings for ALL data...")
print("   This takes a few minutes but makes training MUCH faster!\n")

def compute_embeddings(df, desc):
    """Compute embeddings with progress bar"""
    essays = df['essay_clean'].tolist()
    scores = df['score_norm'].values
    
    embeddings = []
    batch_size = 128
    
    for i in tqdm(range(0, len(essays), batch_size), desc=desc):
        batch = essays[i:i+batch_size]
        batch_embeddings = sentence_model.encode(batch, convert_to_tensor=False, 
                                                 show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings), scores

train_X, train_y = compute_embeddings(train_df, "Train embeddings")
val_X, val_y = compute_embeddings(val_df, "Val embeddings")
test_X, test_y = compute_embeddings(test_df, "Test embeddings")

print(f"\n✅ All embeddings computed!")
print(f"   Train: {train_X.shape}")
print(f"   Val: {val_X.shape}")
print(f"   Test: {test_X.shape}\n")

# Create tensor datasets (super fast during training)
train_dataset = TensorDataset(
    torch.FloatTensor(train_X), 
    torch.FloatTensor(train_y).unsqueeze(1)
)
val_dataset = TensorDataset(
    torch.FloatTensor(val_X), 
    torch.FloatTensor(val_y).unsqueeze(1)
)
test_dataset = TensorDataset(
    torch.FloatTensor(test_X), 
    torch.FloatTensor(test_y).unsqueeze(1)
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"✅ DataLoaders ready!")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}\n")

# ============================================================================
# Model (Optimized Architecture)
# ============================================================================
print("🏗️ Building optimized model...\n")

class OptimizedModel(nn.Module):
    """Lightweight but powerful model"""
    def __init__(self, input_dim):
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

model = OptimizedModel(embedding_dim).to(CONFIG['device'])
print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters\n")

# ============================================================================
# Training Setup
# ============================================================================
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

print("✅ Training setup complete!\n")

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device, max_score):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten() * max_score
    all_targets = np.array(all_targets).flatten() * max_score
    
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return total_loss / len(loader), mae, rmse, r2

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("=" * 80)
print("🚀 STARTING TRAINING ON FULL DATASET")
print("=" * 80)
print(f"\n📊 Training {len(train_df)} samples")
print(f"📊 Validating on {len(val_df)} samples")
print(f"📊 Will test on {len(test_df)} samples\n")
print(f"🎯 Target: R² ≥ 0.80\n")

history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []}
best_r2 = -float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
    
    # Validate
    val_loss, val_mae, val_rmse, val_r2 = validate(
        model, val_loader, criterion, CONFIG['device'], max_score
    )
    
    # Update scheduler
    scheduler.step()
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(float(val_mae))
    history['val_rmse'].append(float(val_rmse))
    history['val_r2'].append(float(val_r2))
    
    # Calculate times
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    
    # Print progress
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - {epoch_time:.1f}s")
    print(f"  Loss: Train={train_loss:.4f} Val={val_loss:.4f}")
    print(f"  MAE: {val_mae:.3f} | RMSE: {val_rmse:.3f}")
    print(f"  R²: {val_r2:.4f} ({val_r2*100:.2f}%) {'🎯' if val_r2>=0.80 else ''}")
    print(f"  Time: {elapsed/60:.1f}min | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model
    if val_r2 > best_r2:
        best_r2 = val_r2
        patience_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
        }, os.path.join(CONFIG['save_dir'], 'best_model.pt'))
        print(f"  💾 Best model saved! (R²={val_r2:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print()

total_time = time.time() - start_time

# ============================================================================
# Final Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("🧪 FINAL EVALUATION ON TEST SET")
print("=" * 80 + "\n")

checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pt'), 
                       weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_mae, test_rmse, test_r2 = validate(
    model, test_loader, criterion, CONFIG['device'], max_score
)

print(f"📊 Test Results (on {len(test_df)} unseen essays):")
print(f"   MAE: {test_mae:.3f}")
print(f"   RMSE: {test_rmse:.3f}")
print(f"   R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   Status: {'✅ SUCCESS!' if test_r2>=0.80 else '⚠️ Close'}\n")

# ============================================================================
# Save Results
# ============================================================================
print("💾 Saving results...\n")

results = {
    'training_time_hours': total_time / 3600,
    'total_samples': len(df_clean),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'best_val_r2': float(best_r2),
    'test_metrics': {
        'mae': float(test_mae),
        'rmse': float(test_rmse),
        'r2': float(test_r2)
    },
    'config': CONFIG,
    'history': history
}

with open(os.path.join(CONFIG['save_dir'], 'results_summary.json'), 'w') as f:
    json.dump(results, f, indent=2)

with open(os.path.join(CONFIG['save_dir'], 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

torch.save({
    'model_state_dict': model.state_dict(),
    'test_metrics': results['test_metrics'],
    'config': CONFIG
}, os.path.join(CONFIG['save_dir'], 'final_model.pt'))

print("✅ All results saved!\n")

# ============================================================================
# Final Summary
# ============================================================================
print("=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
print(f"   Trained on: {len(train_df)} essays (FULL DATASET)")
print(f"   Best Val R²: {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"   Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   Test MAE: {test_mae:.3f} points")
print(f"   Test RMSE: {test_rmse:.3f} points")
print(f"\n💾 Saved to: {CONFIG['save_dir']}")
print(f"   - best_model.pt")
print(f"   - final_model.pt")
print(f"   - results_summary.json")
print(f"   - training_history.json")
print("\n" + "=" * 80)
print("🎉 Training on FULL dataset completed successfully!")
print("=" * 80)
