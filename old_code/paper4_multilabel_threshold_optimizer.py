"""
Multilabel Threshold Optimizer for BERT/RoBERTa Models
======================================================

This module provides functions to find optimal thresholds for multilabel classification
models using Jaccard index optimization. It supports both coordinate descent and 
differential evolution methods, with two optimization strategies:
1. Per-hypothesis optimization (optimizes each label independently)
2. Global optimization (optimizes all thresholds jointly)

Usage in notebook:
    from paper4_multilabel_threshold_optimizer import find_optimal_thresholds_jaccard_global
    
    # Global optimization (recommended)
    optimal_thresholds = find_optimal_thresholds_jaccard_global(
        sector_models=all_sector_models,
        all_data=all_data,
        save_path='optimal_thresholds_jaccard.json'
    )
    
    # Per-hypothesis optimization
    optimal_thresholds = find_optimal_thresholds_jaccard_per_hypothesis(
        sector_models=all_sector_models,
        all_data=all_data,
        save_path='optimal_thresholds_jaccard_per_hypothesis.json'
    )
"""

import itertools
import numpy as np
import json
from scipy.optimize import differential_evolution
from tqdm import tqdm


def calculate_jaccard_metrics(y_true, y_pred):
    """Calculate Jaccard metrics for binary predictions"""
    # Convert to sets for easier intersection/union operations
    true_set = set(i for i, val in enumerate(y_true) if val == 1)
    pred_set = set(i for i, val in enumerate(y_pred) if val == 1)
    
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0  # perfect match for empty sets
    
    intersection = len(true_set.intersection(pred_set))
    union = len(true_set.union(pred_set))
    
    return intersection / union if union > 0 else 0.0


def calculate_micro_jaccard(all_true_labels, all_pred_labels):
    """Calculate micro-averaged Jaccard index across all samples and labels"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
        for i in range(len(true_labels)):
            if true_labels[i] == 1 and pred_labels[i] == 1:
                total_tp += 1
            elif true_labels[i] == 0 and pred_labels[i] == 1:
                total_fp += 1
            elif true_labels[i] == 1 and pred_labels[i] == 0:
                total_fn += 1
    
    denominator = total_tp + total_fp + total_fn
    return total_tp / denominator if denominator > 0 else 0.0


def calculate_micro_jaccard_per_hypothesis(all_true_labels, all_pred_labels, hypothesis_idx):
    """Calculate micro-averaged Jaccard index for a single hypothesis"""
    tp = fp = fn = 0
    
    for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
        if true_labels[hypothesis_idx] == 1 and pred_labels[hypothesis_idx] == 1:
            tp += 1
        elif true_labels[hypothesis_idx] == 0 and pred_labels[hypothesis_idx] == 1:
            fp += 1
        elif true_labels[hypothesis_idx] == 1 and pred_labels[hypothesis_idx] == 0:
            fn += 1
    
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else 0.0


def apply_thresh(scores, thresholds):
    """Apply thresholds to prediction scores"""
    return [[1 if score >= threshold else 0 
             for score, threshold in zip(scores_i, thresholds)]
            for scores_i in scores]


def apply_thresh_single(scores, threshold):
    """Apply threshold to prediction scores for a single hypothesis"""
    return [1 if score >= threshold else 0 for score in scores]


def tune_thresholds_coordinate(all_scores, all_truth, step=0.05, tol=1e-3, max_iter=10):
    """Find optimal thresholds using coordinate descent (global optimization)"""
    num_labels = len(all_scores[0])
    thresholds = np.full(num_labels, 0.5)
    best = calculate_micro_jaccard(all_truth, apply_thresh(all_scores, thresholds))
    
    for it in range(max_iter):
        moved = False
        for i in range(num_labels):
            candidates = np.clip(thresholds[i] + np.linspace(-step, step, int(2*step/step)+1), 0, 1)
            best_t, best_score = thresholds[i], best
            
            for t in candidates:
                th = thresholds.copy()
                th[i] = t
                score = calculate_micro_jaccard(all_truth, apply_thresh(all_scores, th))
                if score > best_score:
                    best_t, best_score = t, score
                    
            if abs(best_t - thresholds[i]) > tol:
                thresholds[i], best = best_t, best_score
                moved = True
                
        if not moved:
            break
            
    return thresholds, best


def tune_threshold_coordinate_single(hypothesis_scores, hypothesis_truth, step=0.05, tol=1e-3, max_iter=10):
    """Find optimal threshold using coordinate descent for a single hypothesis"""
    threshold = 0.5
    best = calculate_micro_jaccard_per_hypothesis(
        hypothesis_truth, 
        [[1 if s >= threshold else 0 for s in hypothesis_scores]], 
        0
    )
    
    for it in range(max_iter):
        moved = False
        candidates = np.clip(threshold + np.linspace(-step, step, int(2*step/step)+1), 0, 1)
        best_t, best_score = threshold, best
        
        for t in candidates:
            preds = [[1 if s >= t else 0 for s in hypothesis_scores]]
            score = calculate_micro_jaccard_per_hypothesis(hypothesis_truth, preds, 0)
            if score > best_score:
                best_t, best_score = t, score
                
        if abs(best_t - threshold) > tol:
            threshold, best = best_t, best_score
            moved = True
                
        if not moved:
            break
            
    return threshold, best


def micro_jaccard_from_thresholds(thresholds, all_scores, all_truth):
    """Calculate micro-Jaccard from thresholds and scores"""
    # apply thresholds
    all_pred = [
        [int(score >= t) for score, t in zip(scores, thresholds)]
        for scores in all_scores
    ]
    # compute micro-jaccard
    tp = fp = fn = 0
    for y_true, y_pred in zip(all_truth, all_pred):
        for yt, yp in zip(y_true, y_pred):
            if yt==1 and yp==1: tp+=1
            elif yt==0 and yp==1: fp+=1
            elif yt==1 and yp==0: fn+=1
    return tp / (tp+fp+fn+1e-12)


def optimize_with_de(all_scores, all_truth, num_labels):
    """Optimize thresholds using differential evolution (global optimization)"""
    # objective to _minimize_  negative micro-jaccard
    def obj(thresholds):
        return -micro_jaccard_from_thresholds(thresholds, all_scores, all_truth)
    bounds = [(0.0,1.0)] * num_labels
    result = differential_evolution(
        obj, bounds, 
        strategy='best1bin',
        maxiter=1000, popsize=15,
        tol=1e-5, polish=True
    )
    best_t = result.x
    best_j = -result.fun
    return best_t, best_j


def optimize_with_de_single(hypothesis_scores, hypothesis_truth):
    """Optimize threshold using differential evolution for a single hypothesis"""
    def obj(threshold):
        preds = [[1 if s >= threshold[0] else 0 for s in hypothesis_scores]]
        return -calculate_micro_jaccard_per_hypothesis(hypothesis_truth, preds, 0)
    
    bounds = [(0.0, 1.0)]
    result = differential_evolution(
        obj, bounds,
        strategy='best1bin',
        maxiter=1000, popsize=15,
        tol=1e-5, polish=True
    )
    best_t = result.x[0]
    best_j = -result.fun
    return best_t, best_j


def predict_with_scores(text, model, tokenizer, label_names, device, max_length=128):
    """
    Get raw prediction scores for a text input
    
    Note: This function assumes you have the predict_with_scores function from your BERT training module.
    If not, you'll need to import it or define it separately.
    """
    import torch
    
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


def find_optimal_thresholds_jaccard_global(sector_models, all_data, save_path='optimal_thresholds_jaccard.json', 
                                          device='cpu', verbose=True):
    """
    Find optimal thresholds for each label in each sector using global optimization (recommended)
    
    This method optimizes all thresholds jointly, which typically gives better results than
    per-hypothesis optimization.
    
    Args:
        sector_models (dict): Dictionary with sector names as keys and model info as values.
                             Each model info should contain 'model', 'tokenizer', and 'label_names'.
        all_data (dict): Dictionary with sector data. Each sector should map comment text to true labels.
        save_path (str): Path to save the optimal thresholds JSON file.
        device (str): Device to run predictions on ('cpu' or 'cuda').
        verbose (bool): Whether to print detailed progress information.
    
    Returns:
        dict: Dictionary containing optimal thresholds for each sector and hypothesis.
    """
    if verbose:
        print("\n" + "="*60)
        print("FINDING OPTIMAL THRESHOLDS USING GLOBAL OPTIMIZATION")
        print("="*60)
    
    optimal_thresholds = {}
    
    sector_iterator = tqdm(sector_models.items(), desc="Processing sectors") if verbose else sector_models.items()
    
    for sector, model_info in sector_iterator:
        if verbose:
            print(f"\n--- {sector.upper()} SECTOR ---")
        
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        label_names = model_info['label_names']
        num_labels = len(label_names)
        
        # Get all texts and true labels for this sector
        sector_data = all_data[sector]
        texts = list(sector_data.keys())
        true_labels_list = list(sector_data.values())
        
        # Convert true labels to binary matrix format
        all_true_labels = []
        for true_labels in true_labels_list:
            binary_labels = [1 if label in true_labels else 0 for label in label_names]
            all_true_labels.append(binary_labels)
        
        # Get raw prediction scores for all texts
        all_scores = []
        text_iterator = tqdm(texts, desc="Getting predictions", leave=False) if verbose else texts
        
        for text in text_iterator:
            results = predict_with_scores(text, model, tokenizer, label_names, device)
            scores = [next((res['score'] for res in results if res['label'] == label), 0.0) 
                     for label in label_names]
            all_scores.append(scores)
            
        if verbose:
            print(f"\nOptimizing thresholds for {sector} sector with {len(texts)} samples...")
        
        # Method 1: Coordinate descent
        if verbose:
            print("\nUsing coordinate descent:")
        cd_thresholds, cd_jaccard = tune_thresholds_coordinate(
            all_scores,
            all_true_labels,
            step=0.05,
            tol=1e-3,
            max_iter=10
        )
        
        # Method 2: Differential evolution
        if verbose:
            print("\nUsing differential evolution:")
        de_thresholds, de_jaccard = optimize_with_de(all_scores, all_true_labels, num_labels)
        
        # Compare and choose best method
        if de_jaccard > cd_jaccard:
            best_thresholds = de_thresholds
            if verbose:
                print(f"\nDifferential evolution performed better (Jaccard: {de_jaccard:.3f} vs {cd_jaccard:.3f})")
        else:
            best_thresholds = cd_thresholds
            if verbose:
                print(f"\nCoordinate descent performed better (Jaccard: {cd_jaccard:.3f} vs {de_jaccard:.3f})")
        
        # Store best thresholds with label names
        optimal_thresholds[sector] = {
            label: threshold for label, threshold in zip(label_names, best_thresholds)
        }
        
        if verbose:
            print(f"\nBest thresholds for {sector}:")
            for label, threshold in optimal_thresholds[sector].items():
                print(f"  {label}: {threshold:.2f}")
    
    # Save thresholds to file
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        if verbose:
            print(f"\nOptimal thresholds saved to '{save_path}'")
    
    return optimal_thresholds


def find_optimal_thresholds_jaccard_per_hypothesis(sector_models, all_data, save_path='optimal_thresholds_jaccard_per_hypothesis.json', 
                                                  device='cpu', verbose=True):
    """
    Find optimal thresholds independently for each hypothesis in each sector
    
    This method optimizes each threshold independently, which may be suboptimal compared
    to global optimization but can be useful for analysis.
    
    Args:
        sector_models (dict): Dictionary with sector names as keys and model info as values.
                             Each model info should contain 'model', 'tokenizer', and 'label_names'.
        all_data (dict): Dictionary with sector data. Each sector should map comment text to true labels.
        save_path (str): Path to save the optimal thresholds JSON file.
        device (str): Device to run predictions on ('cpu' or 'cuda').
        verbose (bool): Whether to print detailed progress information.
    
    Returns:
        dict: Dictionary containing optimal thresholds for each sector and hypothesis.
    """
    if verbose:
        print("\n" + "="*60)
        print("FINDING OPTIMAL THRESHOLDS PER HYPOTHESIS")
        print("="*60)
    
    optimal_thresholds = {}
    
    sector_iterator = tqdm(sector_models.items(), desc="Processing sectors") if verbose else sector_models.items()
    
    for sector, model_info in sector_iterator:
        if verbose:
            print(f"\n--- {sector.upper()} SECTOR ---")
        
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        label_names = model_info['label_names']
        
        # Get data for this sector
        sector_data = all_data[sector]
        texts = list(sector_data.keys())
        true_labels_list = list(sector_data.values())
        
        # Convert true labels to binary matrix format
        all_true_labels = []
        for true_labels in true_labels_list:
            binary_labels = [1 if label in true_labels else 0 for label in label_names]
            all_true_labels.append(binary_labels)
        
        # Get raw prediction scores
        all_scores = []
        text_iterator = tqdm(texts, desc="Getting predictions", leave=False) if verbose else texts
        
        for text in text_iterator:
            results = predict_with_scores(text, model, tokenizer, label_names, device)
            scores = [next((res['score'] for res in results if res['label'] == label), 0.0) 
                     for label in label_names]
            all_scores.append(scores)
            
        optimal_thresholds[sector] = {}
        
        # Process each hypothesis independently
        for hypothesis_idx, label in enumerate(label_names):
            if verbose:
                print(f"\nOptimizing threshold for hypothesis: {label}")
            
            # Extract scores and truth for this hypothesis
            hypothesis_scores = [scores[hypothesis_idx] for scores in all_scores]
            hypothesis_truth = [[labels[hypothesis_idx]] for labels in all_true_labels]
            
            # Method 1: Coordinate descent
            cd_threshold, cd_jaccard = tune_threshold_coordinate_single(
                hypothesis_scores,
                hypothesis_truth
            )
            
            # Method 2: Differential evolution
            de_threshold, de_jaccard = optimize_with_de_single(
                hypothesis_scores,
                hypothesis_truth
            )
            
            # Choose best method for this hypothesis
            if de_jaccard > cd_jaccard:
                best_threshold = de_threshold
                if verbose:
                    print(f"DE better (Jaccard: {de_jaccard:.3f} vs {cd_jaccard:.3f})")
            else:
                best_threshold = cd_threshold
                if verbose:
                    print(f"CD better (Jaccard: {cd_jaccard:.3f} vs {de_jaccard:.3f})")
            
            optimal_thresholds[sector][label] = best_threshold
            if verbose:
                print(f"Optimal threshold: {best_threshold:.3f}")
    
    # Save thresholds to file
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        if verbose:
            print(f"\nOptimal thresholds saved to '{save_path}'")
    
    return optimal_thresholds


# Keep the original function name for backward compatibility
def find_optimal_thresholds_jaccard(sector_models, all_data, save_path='optimal_thresholds_jaccard.json', 
                                   device='cpu', verbose=True):
    """
    Find optimal thresholds using global optimization (recommended approach)
    
    This is an alias for find_optimal_thresholds_jaccard_global for backward compatibility.
    """
    return find_optimal_thresholds_jaccard_global(sector_models, all_data, save_path, device, verbose)


def load_optimal_thresholds(file_path='optimal_thresholds_jaccard.json'):
    """
    Load optimal thresholds from a JSON file
    
    Args:
        file_path (str): Path to the JSON file containing optimal thresholds
        
    Returns:
        dict: Dictionary containing optimal thresholds for each sector and hypothesis
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def apply_optimal_thresholds(predictions, optimal_thresholds, sector, label_names):
    """
    Apply optimal thresholds to raw prediction scores
    
    Args:
        predictions (list): List of prediction scores for each label
        optimal_thresholds (dict): Dictionary containing optimal thresholds
        sector (str): Sector name
        label_names (list): List of label names
        
    Returns:
        list: Binary predictions after applying optimal thresholds
    """
    binary_predictions = []
    sector_thresholds = optimal_thresholds.get(sector, {})
    
    for i, label in enumerate(label_names):
        threshold = sector_thresholds.get(label, 0.5)  # Default to 0.5 if no optimal threshold found
        binary_predictions.append(1 if predictions[i] >= threshold else 0)
    
    return binary_predictions


if __name__ == "__main__":
    # Example usage when run as a script
    print("Multilabel Threshold Optimizer")
    print("This module is designed to be imported in a Jupyter notebook.")
    print("\nExample usage:")
    print("from paper4_multilabel_threshold_optimizer import find_optimal_thresholds_jaccard")
    print("optimal_thresholds = find_optimal_thresholds_jaccard(sector_models, all_data)")
    print("\nFor global optimization (recommended):")
    print("from paper4_multilabel_threshold_optimizer import find_optimal_thresholds_jaccard_global")
    print("\nFor per-hypothesis optimization:")
    print("from paper4_multilabel_threshold_optimizer import find_optimal_thresholds_jaccard_per_hypothesis") 