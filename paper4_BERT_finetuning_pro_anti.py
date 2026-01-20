"""
Fine-tune BERT/RoBERTa for pro-anti classification of Reddit comments using GPT-labeled data
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    DistilBertModel, DistilBertTokenizer,
    get_linear_schedule_with_warmup
)

# Handle AdamW import for different transformers versions
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import json
import os
import glob
import signal
import sys
import pandas as pd
from collections import defaultdict, Counter

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def get_safe_device():
    """Get device safely, handling compatibility issues between PyTorch and transformers versions"""
    try:
        # Try to get device from torch context
        if hasattr(torch, 'get_default_device'):
            return torch.get_default_device()
        else:
            # Fallback for older PyTorch versions
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        # Ultimate fallback
        return torch.device('cpu')

# Import threshold optimization functions - using accuracy+F1 based optimization
from paper4_accuracy_f1_threshold_optimizer import (
    find_optimal_thresholds_accuracy_f1_global,
    apply_optimal_thresholds,
    evaluate_thresholds_accuracy_f1,
    coordinate_descent_thresholds_macro_f1,
    differential_evolution_thresholds_macro_f1,
    calculate_macro_f1_precision_score,
    predict_with_scores
)

# Model configurations
MODEL_CONFIGS = {
    'bert-base-uncased': {
        'model_class': BertModel,
        'tokenizer_class': BertTokenizer,
        'hidden_size': 768
    },
    'roberta-base': {
        'model_class': RobertaModel,
        'tokenizer_class': RobertaTokenizer,
        'hidden_size': 768
    },
    'distilbert-base-uncased': {
        'model_class': DistilBertModel,
        'tokenizer_class': DistilBertTokenizer,
        'hidden_size': 768
    }
}

# Pro-anti label names (same across all sectors) - excluding pro_and_anti
PRO_ANTI_LABELS = ['pro', 'anti', 'neither']

# Global variables for graceful shutdown
current_models = {}
training_interrupted = False

def signal_handler(sig, frame):
    """Handle interruption signals to save models gracefully"""
    global training_interrupted
    print('\n\n⚠️  Training interrupted! Saving current models...')
    training_interrupted = True
    
    # Save current models
    for sector, model_info in current_models.items():
        if model_info and 'model' in model_info:
            save_dir = f'latest_models/{sector}_pro_anti_interrupted'
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, 'model.pt')
            torch.save({
                'model_state_dict': model_info['model'].state_dict(),
                'label_names': model_info['label_names'],
                'model_name': model_info.get('model_name', 'roberta-base'),
                'epoch': model_info.get('current_epoch', 0)
            }, model_path)
            
            # Save tokenizer
            model_info['tokenizer'].save_pretrained(save_dir)
            print(f"✓ Saved {sector} pro-anti model to {save_dir}")
    
    print("Models saved successfully!")
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_pro_anti_training_data(data_dir='paper4data'):
    """Load pro-anti training data from CSV files"""
    training_data = {}
    files_loaded = 0
    
    # Look for train data files
    train_files = glob.glob(os.path.join(data_dir, '*train*_data_by_GPT_df_pro_anti*.csv')) 
    for filepath in train_files:
        try:
            df = pd.read_csv(filepath)
            print(f"Loading {filepath}: {len(df)} samples")
            
            # Group by topic (sector)
            for topic in df['topic'].unique():
                topic_df = df[df['topic'] == topic]
                
                # Convert topic to sector name
                sector = topic_to_sector(topic)
                
                if sector not in training_data:
                    training_data[sector] = {}
                
                # Convert DataFrame rows to comment -> labels format
                for _, row in topic_df.iterrows():
                    comment = row['comment']
                    # Get the label with highest score (excluding pro_and_anti)
                    scores = {
                        'pro': row['pro'],
                        'anti': row['anti'], 
                        'neither': row['neither']
                    }
                    
                    # Find the label with highest score
                    best_label = max(scores.items(), key=lambda x: x[1])[0]
                    
                    # Only include if the best score is above threshold (0.5)
                    if scores[best_label] > 0.5:
                        training_data[sector][comment] = [best_label]
                
                files_loaded += 1
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    print(f"Loaded {files_loaded} training files")
    return training_data

def load_pro_anti_validation_data(data_dir='paper4data'):
    """Load pro-anti validation data from CSV files"""
    validation_data = {}
    files_loaded = 0
    
    # Look for validation data files
    validation_files = glob.glob(os.path.join(data_dir, '*validation*_data_by_GPT_df_pro_anti*.csv'))
    
    for filepath in validation_files:
        try:
            df = pd.read_csv(filepath)
            print(f"Loading validation {filepath}: {len(df)} samples")
            
            # Group by topic (sector)
            for topic in df['topic'].unique():
                topic_df = df[df['topic'] == topic]
                
                # Convert topic to sector name
                sector = topic_to_sector(topic)
                
                if sector not in validation_data:
                    validation_data[sector] = {}
                
                # Convert DataFrame rows to comment -> labels format
                for _, row in topic_df.iterrows():
                    comment = row['comment']
                    # Get the label with highest score (excluding pro_and_anti)
                    scores = {
                        'pro': row['pro'],
                        'anti': row['anti'], 
                        'neither': row['neither']
                    }
                    
                    # Find the label with highest score
                    best_label = max(scores.items(), key=lambda x: x[1])[0]
                    
                    # Only include if the best score is above threshold (0.5)
                    if scores[best_label] > 0.5:
                        validation_data[sector][comment] = [best_label]
                
                files_loaded += 1
                
        except Exception as e:
            print(f"Error loading validation {filepath}: {e}")
    
    print(f"Loaded {files_loaded} validation files")
    return validation_data

def load_pro_anti_test_data(data_dir='paper4data'):
    """Load pro-anti test data from CSV files"""
    test_data = {}
    files_loaded = 0
    
    # Look for test data files
    test_files = glob.glob(os.path.join(data_dir, '*test*_data_by_GPT_df_pro_anti*.csv'))
    
    for filepath in test_files:
        try:
            df = pd.read_csv(filepath)
            print(f"Loading test {filepath}: {len(df)} samples")
            
            # Group by topic (sector)
            for topic in df['topic'].unique():
                topic_df = df[df['topic'] == topic]
                
                # Convert topic to sector name
                sector = topic_to_sector(topic)
                
                if sector not in test_data:
                    test_data[sector] = {}
                
                # Convert DataFrame rows to comment -> labels format
                for _, row in topic_df.iterrows():
                    comment = row['comment']
                    # Get the label with highest score (excluding pro_and_anti)
                    scores = {
                        'pro': row['pro'],
                        'anti': row['anti'], 
                        'neither': row['neither']
                    }
                    
                    # Find the label with highest score
                    best_label = max(scores.items(), key=lambda x: x[1])[0]
                    
                    # Only include if the best score is above threshold (0.5)
                    if scores[best_label] > 0.5:
                        test_data[sector][comment] = [best_label]
                
                files_loaded += 1
                
        except Exception as e:
            print(f"Error loading test {filepath}: {e}")
    
    print(f"Loaded {files_loaded} test files")
    return test_data

def topic_to_sector(topic):
    """Convert topic name to sector name"""
    topic_mapping = {
        'Electric Vehicles': 'transport',
        'Solar Power': 'housing', 
        'Vegetarianism/Veganism': 'food'
    }
    return topic_mapping.get(topic, topic.lower())

def optimize_thresholds_macro_f1_on_training(model, tokenizer, label_names, training_data, sector, device, 
                                           f1_weight=0.8, precision_weight=0.2, method='coordinate'):
    """
    Optimize thresholds using macro F1 + precision on training data
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        label_names: List of label names
        training_data: Training data dictionary
        sector: Current sector
        device: Device to run on
        f1_weight: Weight for macro F1 (default 0.8)
        precision_weight: Weight for macro precision (default 0.2)
        method: Optimization method ('coordinate' or 'differential')
    
    Returns:
        Dictionary of optimal thresholds
    """
    if sector not in training_data or not training_data[sector]:
        print(f"No training data available for {sector}")
        return {}
    
    print(f"Optimizing thresholds for {sector} using training data (macro F1 + precision)...")
    
    # Get training data for this sector
    sector_training_data = training_data[sector]
    texts = list(sector_training_data.keys())
    true_labels_list = list(sector_training_data.values())
    
    # Filter training data to only include labels that the model knows about
    filtered_texts = []
    filtered_true_labels = []
    
    for text, true_labels in zip(texts, true_labels_list):
        # Filter labels to only include those the model knows about
        filtered_labels = [label for label in true_labels if label in label_names]
        if filtered_labels:  # Only include if at least one known label
            filtered_texts.append(text)
            filtered_true_labels.append(filtered_labels)
    
    if not filtered_texts:
        print(f"No valid training data for {sector}")
        return {}
    
    # Convert true labels to binary matrix format
    all_true_labels = []
    for true_labels in filtered_true_labels:
        binary_labels = [1 if label in true_labels else 0 for label in label_names]
        all_true_labels.append(binary_labels)
    
    # Get raw prediction scores for all texts
    all_scores = []
    for text in filtered_texts:
        results = predict_with_scores(text, model, tokenizer, label_names, device)
        scores = [next((res['score'] for res in results if res['label'] == label), 0.0) 
                 for label in label_names]
        all_scores.append(scores)
    
    # Run threshold optimization using the new macro F1 + precision objective
    if method == 'coordinate':
        thresholds, combined_score = coordinate_descent_thresholds_macro_f1(
            all_scores,
            all_true_labels,
            step=0.05,
            tol=1e-3,
            max_iter=15,
            f1_weight=f1_weight,
            precision_weight=precision_weight,
            verbose=False
        )
    elif method == 'differential':
        thresholds, combined_score = differential_evolution_thresholds_macro_f1(
            all_scores,
            all_true_labels,
            f1_weight=f1_weight,
            precision_weight=precision_weight,
            verbose=False
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Store thresholds with label names
    optimal_thresholds = {
        label: threshold for label, threshold in zip(label_names, thresholds)
    }
    
    print(f"Optimal thresholds for {sector} (score={combined_score:.4f}): {optimal_thresholds}")
    return optimal_thresholds

def evaluate_pro_anti_model(model, tokenizer, label_names, validation_data, sector, optimal_thresholds, device):
    """Evaluate pro-anti model on validation data with optimal thresholds"""
    if sector not in validation_data or not validation_data[sector]:
        return {}
    
    sector_validation_data = validation_data[sector]
    texts = list(sector_validation_data.keys())
    true_labels_list = list(sector_validation_data.values())
    
    # Filter validation data to only include labels that the model was trained on
    filtered_texts = []
    filtered_true_labels = []
    
    for text, true_labels in zip(texts, true_labels_list):
        # Filter labels to only include those the model knows about
        filtered_labels = [label for label in true_labels if label in label_names]
        if filtered_labels:  # Only include if at least one known label
            filtered_texts.append(text)
            filtered_true_labels.append(filtered_labels)
    
    if not filtered_texts:
        return {}
    
    # Convert to binary format for threshold optimization
    y_true = []
    y_pred_raw = []
    y_pred_binary = []
    
    model.eval()
    for text, true_labels in zip(filtered_texts, filtered_true_labels):
        # Get predictions
        results = predict_pro_anti_scores(text, model, tokenizer, label_names, device)
        scores = [res['score'] for res in results]
        
        # True labels as binary vector
        true_binary = [1 if label in true_labels else 0 for label in label_names]
        
        # Apply optimal thresholds
        if optimal_thresholds and sector in optimal_thresholds:
            pred_binary = apply_optimal_thresholds(scores, optimal_thresholds, sector, label_names)
        else:
            # Default threshold of 0.5
            pred_binary = [1 if score >= 0.5 else 0 for score in scores]
        
        y_true.append(true_binary)
        y_pred_raw.append(scores)
        y_pred_binary.append(pred_binary)
    
    # Calculate accuracy and F1-score metrics using the new optimizer
    accuracy_f1_results = evaluate_thresholds_accuracy_f1(
        y_pred_raw, y_true, 
        [optimal_thresholds.get(sector, {}).get(label, 0.5) for label in label_names] if optimal_thresholds and sector in optimal_thresholds else [0.5] * len(label_names),
        label_names, verbose=False
    )
    
    # Extract key metrics
    combined_accuracy_f1 = accuracy_f1_results['combined_score_equal']
    macro_f1_score = accuracy_f1_results['f1_macro']
    
    # For single-label classification, calculate accuracy based on predicted vs true labels
    all_predictions = []
    all_true_labels = []
    
    for true_binary, pred_binary in zip(y_true, y_pred_binary):
        # Get the predicted label (highest score after thresholding)
        if sum(pred_binary) > 0:
            predicted_label_idx = np.argmax(pred_binary)
            predicted_label = label_names[predicted_label_idx]
        else:
            predicted_label = 'neither'
        
        # Get true label
        if sum(true_binary) > 0:
            true_label_idx = np.argmax(true_binary)
            true_label = label_names[true_label_idx]
        else:
            true_label = 'neither'
        
        all_predictions.append(predicted_label)
        all_true_labels.append(true_label)
    
    # Calculate accuracy for single-label classification
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='micro', zero_division=0)
    
    # Calculate per-class metrics
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_true_labels, all_predictions, labels=label_names, average=None, zero_division=0
    )
    
    # Calculate macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Calculate weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, labels=label_names, average='weighted', zero_division=0
    )
    
    # Print prediction distribution
    print(f"Prediction distribution: {Counter(all_predictions)}")
    print(f"True label distribution: {Counter(all_true_labels)}")
    
    return {
        'combined_accuracy_f1': combined_accuracy_f1,
        'macro_f1_score': macro_f1_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': dict(zip(label_names, precision_per_class)),
        'per_class_recall': dict(zip(label_names, recall_per_class)),
        'per_class_f1': dict(zip(label_names, f1_per_class)),
        'per_class_support': dict(zip(label_names, support_per_class)),
        'num_samples': len(filtered_texts),
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels
    }

class ProAntiDataset(Dataset):
    """Dataset for pro-anti classification"""
    
    def __init__(self, texts, labels, label_names, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.label_names = label_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        
        # Create one-hot encoded label vector
        label_vector = np.zeros(len(self.label_names))
        for label_idx in label_indices:
            if label_idx < len(self.label_names):
                label_vector[label_idx] = 1.0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float32)
        }

class ProAntiClassifier(nn.Module):
    """Model for pro-anti classification supporting different backbones"""
    
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(ProAntiClassifier, self).__init__()
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        
        # Load model with workaround for torch.get_default_device issue
        try:
            # Temporarily add get_default_device method to torch if it doesn't exist
            if not hasattr(torch, 'get_default_device'):
                def get_default_device():
                    return torch.device('cpu')
                torch.get_default_device = get_default_device
                print("Added temporary get_default_device method to torch")
            
            # Load the model
            self.backbone = config['model_class'].from_pretrained(model_name)
            
        except Exception as e:
            print(f"Warning: Model loading failed ({e}), trying alternative approach")
            # Try with different parameters
            try:
                self.backbone = config['model_class'].from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True
                )
            except Exception as e2:
                print(f"Alternative loading also failed ({e2}), trying basic loading")
                self.backbone = config['model_class'].from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
        self.softmax = nn.Softmax(dim=1)  # Use softmax for single-label classification
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Handle different output formats
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler (e.g., DistilBERT)
            pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return self.softmax(logits)

def get_tokenizer(model_name):
    """Get the appropriate tokenizer for the model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    tokenizer_class = MODEL_CONFIGS[model_name]['tokenizer_class']
    return tokenizer_class.from_pretrained(model_name)

def calculate_class_weights(training_data, label_names=None, method='inverse_frequency'):
    """
    Calculate class weights to handle class imbalance
    
    Args:
        training_data: Dictionary of training data
        label_names: List of label names
        method: Weighting method ('inverse_frequency', 'balanced', 'sqrt_inverse')
    
    Returns:
        Dictionary of class weights
    """
    if label_names is None:
        label_names = PRO_ANTI_LABELS
    
    # Count samples per class
    class_counts = {label: 0 for label in label_names}
    total_samples = 0
    
    for sector_data in training_data.values():
        for comment_labels in sector_data.values():
            for label in comment_labels:
                if label in class_counts:
                    class_counts[label] += 1
                    total_samples += 1
    
    # Calculate weights based on method
    class_weights = {}
    if method == 'inverse_frequency':
        # Standard inverse frequency weighting
        for label in label_names:
            if class_counts[label] > 0:
                class_weights[label] = total_samples / (len(label_names) * class_counts[label])
            else:
                class_weights[label] = 1.0
    
    elif method == 'balanced':
        # Balanced weights (equal weight for all classes)
        for label in label_names:
            class_weights[label] = 1.0
    
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        for label in label_names:
            if class_counts[label] > 0:
                class_weights[label] = np.sqrt(total_samples / (len(label_names) * class_counts[label]))
            else:
                class_weights[label] = 1.0
    
    elif method == 'log_inverse':
        # Logarithm of inverse frequency (even less aggressive)
        for label in label_names:
            if class_counts[label] > 0:
                class_weights[label] = np.log(total_samples / (len(label_names) * class_counts[label])) + 1
            else:
                class_weights[label] = 1.0
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights ({method}): {class_weights}")
    
    return class_weights

def prepare_pro_anti_data(training_data, label_names=None):
    """Prepare training data for pro-anti classification"""
    texts = []
    labels = []
    
    # Use the standard pro-anti labels
    if label_names is None:
        label_names = PRO_ANTI_LABELS
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    
    # Convert data to model format
    for sector_data in training_data.values():
        for comment, comment_labels in sector_data.items():
            texts.append(comment)
            # Convert labels to indices
            label_indices = [label_to_idx[label] for label in comment_labels if label in label_to_idx]
            labels.append(label_indices)
    
    return texts, labels, label_names

def predict_pro_anti_scores(text, model, tokenizer, label_names, device, max_length=128):
    """Get prediction scores for a text input"""
    model.eval()
    
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        scores = outputs[0].cpu().numpy()
    
    # Return label names with their scores
    results = []
    for idx, score in enumerate(scores):
        results.append({
            'label': label_names[idx],
            'score': float(score)
        })
    
    return results

def train_pro_anti_model_with_evaluation(train_data, validation_data, sector, model_name='roberta-base', 
                                       num_epochs=8, batch_size=16, learning_rate=2e-5, save_dir=None, 
                                       use_class_weights=True, class_weight_method='inverse_frequency'):
    """Train a pro-anti BERT model with evaluation"""
    global current_models, training_interrupted
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_texts, train_labels, label_names = prepare_pro_anti_data(train_data)
    
    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights_dict = calculate_class_weights(train_data, label_names, method=class_weight_method)
        # Convert to tensor in the correct order
        class_weights = torch.tensor([class_weights_dict[label] for label in label_names], dtype=torch.float32).to(device)
        print(f"Using class weights ({class_weight_method}): {class_weights}")
    else:
        print("Training without class weights")
    
    # Initialize model and data loaders
    tokenizer = get_tokenizer(model_name)
    train_dataset = ProAntiDataset(train_texts, train_labels, label_names, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ProAntiClassifier(model_name, len(label_names))
    model = model.to(device)
    
    # Store in global for signal handler
    current_models[sector] = {
        'model': model,
        'tokenizer': tokenizer,
        'label_names': label_names,
        'model_name': model_name,
        'current_epoch': 0
    }
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warm-up for RoBERTa
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop with evaluation
    best_micro_jaccard = 0.0
    performance_history = []
    
    # Early stopping variables
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    patience = 3  # Stop if Macro-F1 doesn't improve for 3 epochs
    
    for epoch in range(num_epochs):
        if training_interrupted:
            break
            
        current_models[sector]['current_epoch'] = epoch + 1
        
        model.train()
        total_loss = 0
        
        # Progress bar for each epoch
        progress_bar = tqdm(train_loader, desc=f"{sector.upper()} Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            if training_interrupted:
                break
                
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if class_weights is not None:
                loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels.argmax(dim=1))
            else:
                loss = nn.CrossEntropyLoss()(outputs, labels.argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        if training_interrupted:
            break
            
        avg_loss = total_loss / len(train_loader)
        print(f"{sector.upper()} Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}")
        
        # Evaluation starting from epoch 2
        if epoch >= 1:  # 0-indexed, so epoch 2 = 2nd epoch
            try:
                # Optimize thresholds on training data using macro F1 + precision optimization
                optimal_thresholds = optimize_thresholds_macro_f1_on_training(
                    model, tokenizer, label_names, train_data, sector, device, 
                    f1_weight=0.8, precision_weight=0.2, method='coordinate'
                )
                # Wrap in sector dictionary for compatibility
                if optimal_thresholds:
                    optimal_thresholds = {sector: optimal_thresholds}
                
                # Evaluate on validation data with optimal thresholds
                validation_metrics = evaluate_pro_anti_model(
                    model, tokenizer, label_names, validation_data, sector, optimal_thresholds, device
                )
                
                if validation_metrics:
                    print(
                        f"{sector.upper()} Val E{epoch + 1}: "
                        f"CombinedScore={validation_metrics['combined_accuracy_f1']:.3f} "
                        f"MacroF1={validation_metrics['macro_f1_score']:.3f} "
                        f"F1={validation_metrics['f1']:.3f} "
                        f"Acc={validation_metrics['accuracy']:.3f} "
                        f"P={validation_metrics['precision']:.3f} "
                        f"R={validation_metrics['recall']:.3f} "
                        f"WeightedF1={validation_metrics['weighted_f1']:.3f} "
                        f"W-P={validation_metrics['weighted_precision']:.3f} "
                        f"W-R={validation_metrics['weighted_recall']:.3f} "
                        f"n={validation_metrics['num_samples']}"
                    )
                  
                    # Store performance history
                    performance_history.append({
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'optimal_thresholds': optimal_thresholds[sector] if sector in optimal_thresholds else {},
                        **convert_numpy_types(validation_metrics)
                    })
                    
                    # Check for early stopping based on Macro-F1
                    current_macro_f1 = validation_metrics['macro_f1_score']
                    if current_macro_f1 > best_macro_f1:
                        best_macro_f1 = current_macro_f1
                        epochs_without_improvement = 0
                        print(f"✓ New best Macro-F1: {best_macro_f1:.4f}")
                    else:
                        epochs_without_improvement += 1
                        print(f"⚠ Macro-F1 stalled for {epochs_without_improvement} epochs (best: {best_macro_f1:.4f})")
                    
                    # Save best model based on validation combined accuracy+F1 score
                    if validation_metrics['combined_accuracy_f1'] > best_micro_jaccard:
                        best_micro_jaccard = validation_metrics['combined_accuracy_f1']
                        if save_dir:
                            best_model_dir = os.path.join(save_dir, 'best_model')
                            os.makedirs(best_model_dir, exist_ok=True)
                            
                            model_path = os.path.join(best_model_dir, 'model.pt')
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'label_names': label_names,
                                'model_name': model_name,
                                'epoch': epoch + 1,
                                'combined_accuracy_f1': best_micro_jaccard,
                                'optimal_thresholds': optimal_thresholds[sector] if sector in optimal_thresholds else {}
                            }, model_path)
                            
                            tokenizer.save_pretrained(best_model_dir)
                
            except Exception as e:
                print(f"{sector.upper()} Eval Error E{epoch + 1}: {e}")
            
            # Check for early stopping
            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered! Macro-F1 stalled for {patience} epochs.")
                print(f"Best Macro-F1 achieved: {best_macro_f1:.4f} at epoch {epoch - patience + 2}")
                break
    
    # Save final model
    if save_dir and not training_interrupted:
        final_model_dir = os.path.join(save_dir, 'final_model')
        os.makedirs(final_model_dir, exist_ok=True)
        
        model_path = os.path.join(final_model_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_names': label_names,
            'model_name': model_name,
            'epoch': num_epochs,
            'performance_history': performance_history
        }, model_path)
        
        tokenizer.save_pretrained(final_model_dir)
        
        # Save performance history
        with open(os.path.join(save_dir, 'performance_history.json'), 'w') as f:
            json.dump(performance_history, f, indent=2)
    
    return model, tokenizer, label_names, performance_history

def train_sector_models_pro_anti_with_evaluation(training_data, validation_data, 
                                               model_name='roberta-base', 
                                               num_epochs=8, batch_size=16, learning_rate=2e-5, 
                                               use_class_weights=True, class_weight_method='inverse_frequency'):
    """Train separate models for each sector for pro-anti classification with evaluation"""
    print(f"Training {len(training_data)} sectors for pro-anti classification, {num_epochs} epochs")
    
    # Train models for each sector
    sector_models = {}
    all_performance_histories = {}
    
    for sector, sector_data in training_data.items():
        print(f"\nTraining {sector.upper()}: {len(sector_data)} samples, {len(PRO_ANTI_LABELS)} labels")
        
        # Create single-sector training data
        single_sector_data = {sector: sector_data}
        
        # Set up save directory
        save_dir = f'models/pro_anti_{sector}_{model_name.replace("-", "_")}'
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Train the model with evaluation
            model, tokenizer, label_names, performance_history = train_pro_anti_model_with_evaluation(
                single_sector_data,
                validation_data,
                sector,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=save_dir,
                use_class_weights=use_class_weights,
                class_weight_method=class_weight_method
            )
            
            sector_models[sector] = {
                'model': model,
                'tokenizer': tokenizer,
                'label_names': label_names,
                'save_dir': save_dir
            }
            
            all_performance_histories[sector] = performance_history
            
        except Exception as e:
            print(f"ERROR training {sector}: {e}")
    
    # Save metadata about the training
    metadata = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'pro_anti_labels': PRO_ANTI_LABELS,
        'sector_model_paths': {sector: info['save_dir'] for sector, info in sector_models.items()},
        'all_performance_histories': convert_numpy_types(all_performance_histories)
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/pro_anti_training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return sector_models, all_performance_histories

def debug_training_pro_anti_with_evaluation():
    """Train pro-anti models with evaluation"""
    try:
        # Load training and validation data
        all_data = load_pro_anti_training_data()
        validation_data = load_pro_anti_validation_data()
        
        if not all_data:
            print("No training data found!")
            return
        
        total_train = sum(len(sector_data) for sector_data in all_data.values())
        total_validation = sum(len(sector_data) for sector_data in validation_data.values()) if validation_data else 0
        print(f"Loaded: {total_train} training samples, {total_validation} validation samples")
        
        # Train sector models with evaluation
        sector_models, performance_histories = train_sector_models_pro_anti_with_evaluation(
            all_data,
            validation_data,
            model_name='roberta-base',
                                    num_epochs=8,
            batch_size=16,
            learning_rate=6e-6,  # Lower learning rate for RoBERTa
            use_class_weights=True  # Enable class weighting for minority classes
        )
        
        # Print final summary
        print(f"\nFINAL BEST PERFORMANCE:")
        for sector, history in performance_histories.items():
            if history:
                best_performance = max(history, key=lambda x: x['combined_accuracy_f1'])
                print(f"{sector.upper()}: E{best_performance['epoch']} CombinedScore={best_performance['combined_accuracy_f1']:.3f} MacroF1={best_performance['macro_f1_score']:.3f} F1={best_performance['f1']:.3f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run training with evaluation
    debug_training_pro_anti_with_evaluation() 