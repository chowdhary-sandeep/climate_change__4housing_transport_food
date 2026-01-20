"""
Fine-tune BERT/RoBERTa for multilabel classification of Reddit comments using GPT-labeled data
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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, confusion_matrix
from tqdm import tqdm
import json
import os
import glob
import signal
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import threshold optimization functions
from paper4_multilabel_threshold_optimizer import (
    find_optimal_thresholds_jaccard_global,
    apply_optimal_thresholds,
    calculate_micro_jaccard,
    calculate_jaccard_metrics
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
            save_dir = f'latest_models/{sector}_interrupted'
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, 'model.pt')
            torch.save({
                'model_state_dict': model_info['model'].state_dict(),
                'label_names': model_info['label_names'],
                'model_name': model_info.get('model_name', 'distilbert-base-uncased'),
                'epoch': model_info.get('current_epoch', 0)
            }, model_path)
            
            # Save tokenizer
            model_info['tokenizer'].save_pretrained(save_dir)
            print(f"✓ Saved {sector} model to {save_dir}")
    
    print("Models saved successfully!")
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_all_training_data(data_dir='paper4data'):
    """Load all training data files and combine them"""
    training_data = {}
    files_loaded = 0
    
    for filename in os.listdir(data_dir):
        # Include both the base file and numbered files
        if (filename.startswith('training_data_by_GPT') and 
            filename.endswith('.json') and 
            'dict' in filename and 
            'results' not in filename):
            
            filepath = os.path.join(data_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    file_data = json.load(f)
                    
                # Merge the data
                for sector, sector_data in file_data.items():
                    if sector not in training_data:
                        training_data[sector] = {}
                    training_data[sector].update(sector_data)
                
                files_loaded += 1
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return training_data

def load_test_data():
    """Load all test data files"""
    test_data = {}
    
    test_files = glob.glob("paper4data/validation_data_by_GPT_dict_*.json")
    
    if not test_files:
        return {}
    
    for filepath in test_files:
        try:
            with open(filepath, 'r') as f:
                file_data = json.load(f)
                
            # Merge the data
            for sector, sector_data in file_data.items():
                if sector not in test_data:
                    test_data[sector] = {}
                test_data[sector].update(sector_data)
                
        except Exception as e:
            print(f"Error loading {os.path.basename(filepath)}: {e}")
    
    return test_data

def get_top_hypotheses_by_sector(training_data, top_n=7):
    """Get the top N most frequent hypotheses for each sector"""
    sector_top_hypotheses = {}
    
    for sector, comments in training_data.items():
        label_counts = {}
        
        # Count occurrences of each label
        for true_labels in comments.values():
            for label in true_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
                
        # Sort by count and get top N
        sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        top_hypotheses = [label for label, count in sorted_counts[:top_n]]
        
        sector_top_hypotheses[sector] = top_hypotheses
    
    return sector_top_hypotheses

def generate_confusion_matrix(y_true, y_pred, label_names, sector, save_dir='paper4figs'):
    """
    Generate and save confusion matrix for multilabel classification
    
    Args:
        y_true: List of true label vectors (binary)
        y_pred: List of predicted label vectors (binary)
        label_names: List of label names
        sector: Current sector name
        save_dir: Directory to save the confusion matrix
    
    Returns:
        Confusion matrix array
    """
    # Convert to numpy arrays
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Create confusion matrix for each label
    num_labels = len(label_names)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    
    # For multilabel, we need to handle each label separately
    # This creates a confusion matrix showing the relationship between predicted and true labels
    for i in range(num_labels):
        for j in range(num_labels):
            # Count samples where true label i and predicted label j
            cm[i, j] = np.sum((y_true_array[:, i] == 1) & (y_pred_array[:, j] == 1))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {sector.upper()} Sector (Multilabel)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'confusion_matrix_{sector}_multilabel.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {filepath}")
    
    return cm

def calculate_metrics_from_confusion_matrix(cm, label_names):
    """
    Calculate accuracy, precision, recall, and F1 from confusion matrix
    
    Args:
        cm: Confusion matrix
        label_names: List of label names
    
    Returns:
        Dictionary of metrics
    """
    num_labels = len(label_names)
    
    # Calculate per-label metrics
    precision_per_label = []
    recall_per_label = []
    f1_per_label = []
    
    for i in range(num_labels):
        # True positives
        tp = cm[i, i]
        # False positives (sum of column i minus true positives)
        fp = np.sum(cm[:, i]) - tp
        # False negatives (sum of row i minus true positives)
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_label.append(precision)
        recall_per_label.append(recall)
        f1_per_label.append(f1)
    
    # Calculate macro averages
    macro_precision = np.mean(precision_per_label)
    macro_recall = np.mean(recall_per_label)
    macro_f1 = np.mean(f1_per_label)
    
    # Calculate overall accuracy (sum of diagonal / total)
    total_samples = np.sum(cm)
    accuracy = np.sum(np.diag(cm)) / total_samples if total_samples > 0 else 0
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'precision_per_label': dict(zip(label_names, precision_per_label)),
        'recall_per_label': dict(zip(label_names, recall_per_label)),
        'f1_per_label': dict(zip(label_names, f1_per_label))
    }

def filter_data_by_top_hypotheses(training_data, sector_top_hypotheses):
    """Filter training data to include only top hypotheses for each sector"""
    filtered_data = {}
    
    for sector, comments in training_data.items():
        top_hypotheses = set(sector_top_hypotheses[sector])
        filtered_comments = {}
        
        for comment, true_labels in comments.items():
            # Filter labels to only include top hypotheses
            filtered_labels = [label for label in true_labels if label in top_hypotheses]
            
            # Only include comments that have at least one top hypothesis
            if filtered_labels:
                filtered_comments[comment] = filtered_labels
        
        filtered_data[sector] = filtered_comments
        
    return filtered_data

def evaluate_on_validation_data(model, tokenizer, label_names, validation_data, sector, optimal_thresholds, device):
    """Evaluate model on validation data with optimal thresholds"""
    if sector not in validation_data or not validation_data[sector]:
        return {}
    
    sector_validation_data = validation_data[sector]
    texts = list(sector_validation_data.keys())
    true_labels_list = list(sector_validation_data.values())
    
    # Filter test data to only include labels that the model was trained on
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
    
    # Convert to binary format
    y_true = []
    y_pred_raw = []
    y_pred_binary = []
    
    model.eval()
    for text, true_labels in zip(filtered_texts, filtered_true_labels):
        # Get predictions
        results = predict_with_scores(text, model, tokenizer, label_names, device)
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
    
    # Calculate metrics using confusion matrix
    # Generate and save confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred_binary, label_names, sector)
    
    # Calculate metrics from confusion matrix
    cm_metrics = calculate_metrics_from_confusion_matrix(cm, label_names)
    
    # Also calculate Jaccard metrics for comparison
    micro_jaccard = calculate_micro_jaccard(y_true, y_pred_binary)
    sample_jaccards = [calculate_jaccard_metrics(true, pred) for true, pred in zip(y_true, y_pred_binary)]
    macro_jaccard = np.mean(sample_jaccards) if sample_jaccards else 0.0
    
    return {
        'micro_jaccard': micro_jaccard,
        'macro_jaccard': macro_jaccard,
        'accuracy': cm_metrics['accuracy'],
        'precision': cm_metrics['macro_precision'],
        'recall': cm_metrics['macro_recall'],
        'f1': cm_metrics['macro_f1'],
        'macro_precision': cm_metrics['macro_precision'],
        'macro_recall': cm_metrics['macro_recall'],
        'macro_f1': cm_metrics['macro_f1'],
        'precision_per_label': cm_metrics['precision_per_label'],
        'recall_per_label': cm_metrics['recall_per_label'],
        'f1_per_label': cm_metrics['f1_per_label'],
        'confusion_matrix': cm.tolist(),
        'y_true': y_true,
        'y_pred': y_pred_binary,
        'num_samples': len(filtered_texts)
    }

class MultilabelDataset(Dataset):
    """Dataset for multilabel classification"""
    
    def __init__(self, texts, labels, label_names, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.label_names = label_names
        self.num_labels = len(label_names)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        
        # Create multilabel vector (1 for present labels, 0 for absent)
        label_vector = torch.zeros(self.num_labels)
        label_vector[label_indices] = 1
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }

class MultilabelClassifier(nn.Module):
    """Base model for multilabel classification supporting different backbones"""
    
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(MultilabelClassifier, self).__init__()
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        self.backbone = config['model_class'].from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
        self.sigmoid = nn.Sigmoid()
    
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
        return self.sigmoid(logits)

def get_tokenizer(model_name):
    """Get the appropriate tokenizer for the model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    tokenizer_class = MODEL_CONFIGS[model_name]['tokenizer_class']
    return tokenizer_class.from_pretrained(model_name)

def prepare_data(training_data, label_names=None):
    """Prepare training data for multilabel classification"""
    texts = []
    labels = []
    all_labels = set()
    
    # First pass: collect all unique labels if not provided
    if label_names is None:
        for sector_data in training_data.values():
            for comment_labels in sector_data.values():
                all_labels.update(comment_labels)
        label_names = sorted(list(all_labels))
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    
    # Second pass: convert data to model format
    for sector_data in training_data.values():
        for comment, comment_labels in sector_data.items():
            texts.append(comment)
            # Convert labels to indices
            label_indices = [label_to_idx[label] for label in comment_labels if label in label_to_idx]
            labels.append(label_indices)
    
    return texts, labels, label_names

def predict_with_scores(text, model, tokenizer, label_names, device, max_length=128):
    """Get raw prediction scores for a text input"""
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

def train_model_with_evaluation(train_data, validation_data, sector, model_name='distilbert-base-uncased', 
                               num_epochs=10, batch_size=16, learning_rate=2e-5, save_dir=None):
    """Train a multilabel BERT model with threshold optimization and test evaluation"""
    global current_models, training_interrupted
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    train_texts, train_labels, label_names = prepare_data(train_data)
    
    # Initialize model and data loaders
    tokenizer = get_tokenizer(model_name)
    train_dataset = MultilabelDataset(train_texts, train_labels, label_names, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MultilabelClassifier(model_name, len(label_names))
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
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop with evaluation
    best_micro_jaccard = 0.0
    performance_history = []
    
    for epoch in range(num_epochs):
        if training_interrupted:
            break
            
        current_models[sector]['current_epoch'] = epoch + 1
        
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            if training_interrupted:
                break
                
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.BCELoss()(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        if training_interrupted:
            break
            
        avg_loss = total_loss / len(train_loader)
        print(f"{sector.upper()} Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}")
        
        # Evaluation starting from epoch 4
        if epoch >= 3:  # 0-indexed, so epoch 3 = 4th epoch
            try:
                sector_models = {sector: current_models[sector]}
                optimal_thresholds = find_optimal_thresholds_jaccard_global(
                    sector_models, train_data, save_path=None, device=device, verbose=False
                )
                
                # Evaluate on validation data
                validation_metrics = evaluate_on_validation_data(
                    model, tokenizer, label_names, validation_data, sector, 
                    optimal_thresholds, device
                )
                
                if validation_metrics:
                    print(f"{sector.upper()} Val E{epoch + 1}: μJ={validation_metrics['micro_jaccard']:.3f} MJ={validation_metrics['macro_jaccard']:.3f} F1={validation_metrics['f1']:.3f} Acc={validation_metrics['accuracy']:.3f} P={validation_metrics['precision']:.3f} R={validation_metrics['recall']:.3f} Samples={validation_metrics['num_samples']}")
                    
                    # Store performance history
                    performance_history.append({
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'optimal_thresholds': optimal_thresholds[sector] if sector in optimal_thresholds else {},
                        **validation_metrics
                    })
                    
                    # Save best model
                    if validation_metrics['micro_jaccard'] > best_micro_jaccard:
                        best_micro_jaccard = validation_metrics['micro_jaccard']
                        if save_dir:
                            best_model_dir = os.path.join(save_dir, 'best_model')
                            os.makedirs(best_model_dir, exist_ok=True)
                            
                            model_path = os.path.join(best_model_dir, 'model.pt')
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'label_names': label_names,
                                'model_name': model_name,
                                'epoch': epoch + 1,
                                'micro_jaccard': best_micro_jaccard,
                                'optimal_thresholds': optimal_thresholds[sector] if sector in optimal_thresholds else {},
                                'confusion_matrix': validation_metrics.get('confusion_matrix', []),
                                'validation_metrics': validation_metrics
                            }, model_path)
                            
                            tokenizer.save_pretrained(best_model_dir)
                            
                            # Save confusion matrix separately for the best model
                            if 'confusion_matrix' in validation_metrics:
                                cm_save_dir = os.path.join(save_dir, 'best_model')
                                generate_confusion_matrix(
                                    validation_metrics.get('y_true', []), 
                                    validation_metrics.get('y_pred', []), 
                                    label_names, 
                                    sector, 
                                    save_dir=cm_save_dir
                                )
                
            except Exception as e:
                print(f"{sector.upper()} Eval Error E{epoch + 1}: {e}")
    
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

def train_sector_models_top_hypotheses_with_evaluation(training_data, validation_data, top_n=7, 
                                                     model_name='distilbert-base-uncased', 
                                                     num_epochs=10, batch_size=16, learning_rate=2e-5):
    """Train separate models for each sector with evaluation"""
    print(f"Training {len(training_data)} sectors with top {top_n} hypotheses, {num_epochs} epochs")
    
    # Get top hypotheses for each sector
    sector_top_hypotheses = get_top_hypotheses_by_sector(training_data, top_n)
    
    # Filter data to only include top hypotheses
    filtered_data = filter_data_by_top_hypotheses(training_data, sector_top_hypotheses)
    
    # Train models for each sector
    sector_models = {}
    all_performance_histories = {}
    
    for sector, sector_data in filtered_data.items():
        print(f"\nTraining {sector.upper()}: {len(sector_data)} samples, {len(sector_top_hypotheses[sector])} labels")
        
        # Create single-sector training data
        single_sector_data = {sector: sector_data}
        
        # Set up save directory
        save_dir = f'models/top{top_n}_{sector}_{model_name.replace("-", "_")}'
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Train the model with evaluation
            model, tokenizer, label_names, performance_history = train_model_with_evaluation(
                single_sector_data,
                validation_data,
                sector,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=save_dir
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
        'top_n': top_n,
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'sector_top_hypotheses': sector_top_hypotheses,
        'sector_model_paths': {sector: info['save_dir'] for sector, info in sector_models.items()},
        'all_performance_histories': all_performance_histories
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/top_hypotheses_training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return sector_models, all_performance_histories

def debug_training_top_hypotheses_with_evaluation():
    """Train models using only top 7 hypotheses per sector with evaluation"""
    try:
        # Load training and validation data
        all_data = load_all_training_data()
        validation_data = load_test_data()  # This actually loads validation data
        
        if not all_data:
            print("No training data found!")
            return
        
        total_train = sum(len(sector_data) for sector_data in all_data.values())
        total_validation = sum(len(sector_data) for sector_data in validation_data.values()) if validation_data else 0
        print(f"Loaded: {total_train} training samples, {total_validation} validation samples")
        
        # Train sector models with evaluation
        sector_models, performance_histories = train_sector_models_top_hypotheses_with_evaluation(
            all_data,
            validation_data,
            top_n=7,
            model_name='distilbert-base-uncased',
            num_epochs=10,
            batch_size=16,
            learning_rate=2e-5
        )
        
        # Print final summary
        print(f"\nFINAL BEST PERFORMANCE:")
        for sector, history in performance_histories.items():
            if history:
                best_performance = max(history, key=lambda x: x['micro_jaccard'])
                print(f"{sector.upper()}: E{best_performance['epoch']} μJ={best_performance['micro_jaccard']:.3f} MJ={best_performance['macro_jaccard']:.3f} F1={best_performance['f1']:.3f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run training with evaluation
    debug_training_top_hypotheses_with_evaluation() 