"""
Simplified evaluation script for multilabel BERT classification with sector-specific models
"""
import json
import torch
import numpy as np
from paper4_BERT_finetuning import predict_with_scores
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import f1_score, hamming_loss

def load_all_training_data():
    """Load and combine all training data files"""
    # Find all training data files
    data_files = glob.glob('paper4data/training_data_by_GPT_dict_*.json')
    
    # Initialize combined data structure
    combined_data = defaultdict(dict)
    
    # Load and combine all files
    for file_path in data_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sector, comments in data.items():
                combined_data[sector].update(comments)
    
    print(f"Loaded {len(data_files)} training data files")
    total_samples = sum(len(comments) for comments in combined_data.values())
    print(f"Total combined samples: {total_samples}")
    print("\nSamples per sector:")
    for sector, comments in combined_data.items():
        print(f"  {sector}: {len(comments)}")
    
    return dict(combined_data)

def load_best_models():
    """Load the best trained models for each sector"""
    model_dirs = glob.glob('models/top7_*_distilbert_base_uncased/best_model')
    sector_models = {}
    
    for model_dir in model_dirs:
        # Extract sector name from path (handle both Windows and Unix paths)
        path_parts = model_dir.replace('\\', '/').split('/')
        if len(path_parts) >= 2:
            # Extract sector from directory name like 'top7_food_distilbert_base_uncased'
            dir_name = path_parts[1]  # e.g., 'top7_food_distilbert_base_uncased'
            if dir_name.startswith('top7_') and '_distilbert' in dir_name:
                # Extract the sector part between 'top7_' and '_distilbert'
                sector = dir_name.split('_')[1]  # Extract 'food', 'housing', 'transport'
            else:
                print(f"Unexpected directory name format: {dir_name}")
                continue
        else:
            print(f"Unexpected path format: {model_dir}")
            continue
        
        try:
            # Load model checkpoint (set weights_only=False for compatibility with older PyTorch saves)
            checkpoint = torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu', weights_only=False)
            
            # Load tokenizer
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            
            # Load model
            from paper4_BERT_finetuning import MultilabelClassifier
            model_name = checkpoint.get('model_name', 'distilbert-base-uncased')
            model = MultilabelClassifier(model_name, len(checkpoint['label_names']))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            sector_models[sector] = {
                'model': model,
                'tokenizer': tokenizer,
                'label_names': checkpoint['label_names'],
                'checkpoint': checkpoint,
                'model_dir': model_dir
            }
            
            print(f"Loaded {sector} model: Epoch {checkpoint['epoch']}, μJ={checkpoint.get('micro_jaccard', 'N/A')}")
            print(f"  Labels: {checkpoint['label_names']}")
            
        except Exception as e:
            print(f"Error loading {sector} model from {model_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    return sector_models

def compute_example_based_f1(y_true, y_pred):
    """
    Compute Example-based F₁ score for multilabel classification
    
    For each example (comment), compute F₁ between predicted and true label sets,
    then average across all examples.
    """
    f1_scores = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Convert to sets of indices where label is 1
        true_set = set(i for i, label in enumerate(true_labels) if label == 1)
        pred_set = set(i for i, label in enumerate(pred_labels) if label == 1)
        
        # Compute precision and recall for this example
        if len(pred_set) == 0 and len(true_set) == 0:
            # Perfect match when both are empty
            f1_scores.append(1.0)
        elif len(pred_set) == 0:
            # No predictions but there are true labels
            f1_scores.append(0.0)
        elif len(true_set) == 0:
            # Predictions but no true labels
            f1_scores.append(0.0)
        else:
            # Normal case: compute F₁
            intersection = len(true_set.intersection(pred_set))
            precision = intersection / len(pred_set)
            recall = intersection / len(true_set)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
    
    return np.mean(f1_scores)

def compute_hamming_accuracy(y_true, y_pred):
    """
    Compute Hamming Accuracy (1 - Hamming Loss) for multilabel classification
    
    This measures the fraction of all (example, label) pairs that are correctly predicted.
    """
    # sklearn's hamming_loss computes the fraction of wrong predictions
    # Hamming accuracy = 1 - hamming_loss
    return 1 - hamming_loss(y_true, y_pred)

def compute_macro_f1(y_true, y_pred, label_names):
    """
    Compute Macro-F₁ score for multilabel classification
    
    Compute F₁ score for each label separately, then average across labels.
    """
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Compute F₁ for each label
    f1_scores = []
    for i in range(len(label_names)):
        # Extract binary predictions for this label
        true_label = y_true_array[:, i]
        pred_label = y_pred_array[:, i]
        
        # Compute F₁ for this label
        label_f1 = f1_score(true_label, pred_label, zero_division=0)
        f1_scores.append(label_f1)
    
    return np.mean(f1_scores), f1_scores

def compute_multilabel_metrics(y_true, y_pred, label_names):
    """
    Compute all three multilabel metrics
    """
    example_f1 = compute_example_based_f1(y_true, y_pred)
    hamming_acc = compute_hamming_accuracy(y_true, y_pred)
    macro_f1, per_label_f1 = compute_macro_f1(y_true, y_pred, label_names)
    
    return {
        'example_based_f1': example_f1,
        'hamming_accuracy': hamming_acc,
        'macro_f1': macro_f1,
        'per_label_f1': dict(zip(label_names, per_label_f1))
    }

def evaluate_sector_model(model_info, sector_data, sector):
    """Evaluate a sector model and generate confusion matrix using optimal thresholds"""
    model = model_info['model']
    tokenizer = model_info['tokenizer']
    label_names = model_info['label_names']
    checkpoint = model_info['checkpoint']
    
    print(f"\nEvaluating {sector.upper()} model...")
    
    # Get optimal thresholds if available
    optimal_thresholds = checkpoint.get('optimal_thresholds', {})
    print(f"Using optimal thresholds: {optimal_thresholds}")
    
    # Get test samples (use a subset for evaluation)
    texts = list(sector_data.keys())
    labels_list = list(sector_data.values())
    
    # Use first 100 samples for evaluation
    num_eval_samples = min(100, len(texts))
    eval_texts = texts[:num_eval_samples]
    eval_labels = labels_list[:num_eval_samples]
    
    # Filter to only include labels the model knows about
    filtered_texts = []
    filtered_labels = []
    
    for text, true_labels in zip(eval_texts, eval_labels):
        filtered_labels_for_text = [label for label in true_labels if label in label_names]
        if filtered_labels_for_text:
            filtered_texts.append(text)
            filtered_labels.append(filtered_labels_for_text)
    
    if not filtered_texts:
        print(f"No valid samples for {sector}")
        return None
    
    # Get predictions
    y_true = []
    y_pred_default = []
    y_pred_optimal = []
    
    for text, true_labels in zip(filtered_texts, filtered_labels):
        # Get model predictions
        results = predict_with_scores(text, model, tokenizer, label_names, 'cpu')
        scores = [res['score'] for res in results]
        
        # Convert to binary predictions using default threshold 0.5
        pred_binary_default = [1 if score >= 0.5 else 0 for score in scores]
        
        # Convert to binary predictions using optimal thresholds if available
        pred_binary_optimal = []
        for i, (label, score) in enumerate(zip(label_names, scores)):
            threshold = optimal_thresholds.get(label, 0.5)
            pred_binary_optimal.append(1 if score >= threshold else 0)
        
        # Convert true labels to binary vector
        true_binary = [1 if label in true_labels else 0 for label in label_names]
        
        y_true.append(true_binary)
        y_pred_default.append(pred_binary_default)
        y_pred_optimal.append(pred_binary_optimal)
    
    # Use optimal thresholds if available, otherwise default
    y_pred_to_use = y_pred_optimal if optimal_thresholds else y_pred_default
    threshold_type = "optimal" if optimal_thresholds else "default (0.5)"
    
    # Compute multilabel metrics
    metrics = compute_multilabel_metrics(y_true, y_pred_to_use, label_names)
    
    print(f"{sector.upper()} Results (using {threshold_type} thresholds):")
    print(f"  Example-based F₁: {metrics['example_based_f1']:.3f}")
    print(f"  Hamming Accuracy:  {metrics['hamming_accuracy']:.3f}")
    print(f"  Macro-F₁:          {metrics['macro_f1']:.3f}")
    print(f"  Samples evaluated: {len(filtered_texts)}")
    
    # Print per-label F₁ scores
    print(f"  Per-label F₁ scores:")
    for label, f1 in metrics['per_label_f1'].items():
        print(f"    {label:<35} {f1:.3f}")
    
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred_to_use,
        'y_pred_default': y_pred_default,
        'y_pred_optimal': y_pred_optimal,
        'optimal_thresholds': optimal_thresholds,
        'threshold_type': threshold_type,
        'num_samples': len(filtered_texts)
    }

def main():
    """Main execution function"""
    print("🔍 EVALUATING SECTOR-SPECIFIC MULTILABEL BERT MODELS")
    
    # Load training data
    all_data = load_all_training_data()
    
    # Load best models
    sector_models = load_best_models()
    
    if not sector_models:
        print("No trained models found!")
        return
    
    # Evaluate each sector model
    evaluation_results = {}
    
    for sector, model_info in sector_models.items():
        if sector in all_data:
            results = evaluate_sector_model(model_info, all_data[sector], sector)
            if results:
                evaluation_results[sector] = results
    
    # Print summary
    print(f"\n📊 MULTILABEL EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Sector':<12} {'Threshold':<12} {'Example-F₁':<12} {'Hamming-Acc':<12} {'Macro-F₁':<12} {'Samples':<8}")
    print("-" * 90)
    for sector, results in evaluation_results.items():
        metrics = results['metrics']
        threshold_type = results['threshold_type']
        print(f"{sector.upper():<12} {threshold_type:<12} "
              f"{metrics['example_based_f1']:<12.3f} {metrics['hamming_accuracy']:<12.3f} "
              f"{metrics['macro_f1']:<12.3f} {results['num_samples']:<8}")
    
    print("\n📋 METRICS EXPLANATION:")
    print("• Example-based F₁: Average F₁ score computed per comment (rewards partial matches)")
    print("• Hamming Accuracy: Fraction of all (comment, label) decisions that are correct")
    print("• Macro-F₁: Average F₁ score across all labels (treats each label equally)")
    
    print(f"\n✅ Evaluation complete for {len(evaluation_results)} sectors using optimal thresholds")
    
    # Generate LaTeX table
    generate_latex_table(evaluation_results)

def generate_latex_table(evaluation_results):
    """Generate LaTeX table for multilabel evaluation results"""
    print("\n" + "=" * 60)
    print("LATEX TABLE FOR MULTILABEL METRICS")
    print("=" * 60)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Sector} & \\textbf{Example-F₁} & \\textbf{Hamming Acc.} & \\textbf{Macro-F₁} \\\\")
    print("\\hline")
    
    for sector, results in evaluation_results.items():
        metrics = results['metrics']
        sector_name = sector.capitalize()
        example_f1 = metrics['example_based_f1']
        hamming_acc = metrics['hamming_accuracy']
        macro_f1 = metrics['macro_f1']
        
        print(f"{sector_name} & {example_f1:.3f} & {hamming_acc:.3f} & {macro_f1:.3f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Multilabel Classification Performance with Optimal Thresholds}")
    print("\\label{tab:multilabel_results}")
    print("\\end{table}")
    
    print("\nCopy-paste ready LaTeX rows:")
    print("-" * 40)
    for sector, results in evaluation_results.items():
        metrics = results['metrics']
        sector_name = sector.capitalize()
        example_f1 = metrics['example_based_f1']
        hamming_acc = metrics['hamming_accuracy']
        macro_f1 = metrics['macro_f1']
        
        print(f"{sector_name} & {example_f1:.3f} & {hamming_acc:.3f} & {macro_f1:.3f} \\\\")

if __name__ == "__main__":
    main() 