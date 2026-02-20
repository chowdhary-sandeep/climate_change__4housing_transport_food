"""
Threshold optimization for pro-anti classification based on accuracy and F1-score
This module provides functions to find optimal classification thresholds by maximizing 
a combination of accuracy and F1-score using various optimization methods.
"""

import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scipy.optimize import differential_evolution
import itertools

def predict_with_scores(text, model, tokenizer, label_names, device, max_length=128):
    """Get prediction scores for a text input"""
    model.eval()
    
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    import torch
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

def calculate_accuracy_f1_score(scores, true_labels, thresholds, f1_weight=0.5):
    """
    Calculate combined accuracy and F1-score metric for single-label classification
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        thresholds (list): List of thresholds for each label
        f1_weight (float): Weight for F1-score vs accuracy (0.5 = equal weight)
    
    Returns:
        float: Combined accuracy and F1-score metric
    """
    # Apply thresholds to get predictions
    predictions = []
    true_binary = []
    
    for sample_scores, sample_true in zip(scores, true_labels):
        # Apply thresholds to get binary predictions
        pred_binary = [1 if score >= thresh else 0 for score, thresh in zip(sample_scores, thresholds)]
        
        # For single-label classification: find the label with highest score among those above threshold
        if sum(pred_binary) > 0:
            # Among labels that pass threshold, pick the one with highest score
            valid_scores = [score if pred else -float('inf') for score, pred in zip(sample_scores, pred_binary)]
            pred_label_idx = np.argmax(valid_scores)
        else:
            # If no label passes threshold, pick the one with highest score
            pred_label_idx = np.argmax(sample_scores)
        
        # Convert true labels to single label
        if sum(sample_true) > 0:
            true_label_idx = np.argmax(sample_true)
        else:
            true_label_idx = len(sample_true) - 1  # Default to last label (typically 'neither')
        
        predictions.append(pred_label_idx)
        true_binary.append(true_label_idx)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_binary, predictions)
    
    # Calculate F1-score (macro average)
    f1_macro = f1_score(true_binary, predictions, average='macro', zero_division=0)
    
    # Combined metric
    combined_score = (1 - f1_weight) * accuracy + f1_weight * f1_macro
    
    return combined_score

def calculate_macro_f1_precision_score(scores, true_labels, thresholds, f1_weight=0.8, precision_weight=0.2):
    """
    Calculate combined macro F1 and macro precision score for single-label classification
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        thresholds (list): List of thresholds for each label
        f1_weight (float): Weight for macro F1-score (default 0.8)
        precision_weight (float): Weight for macro precision (default 0.2)
    
    Returns:
        float: Combined macro F1 and precision metric
    """
    # Apply thresholds to get predictions
    predictions = []
    true_binary = []
    
    for sample_scores, sample_true in zip(scores, true_labels):
        # Apply thresholds to get binary predictions
        pred_binary = [1 if score >= thresh else 0 for score, thresh in zip(sample_scores, thresholds)]
        
        # For single-label classification: find the label with highest score among those above threshold
        if sum(pred_binary) > 0:
            # Among labels that pass threshold, pick the one with highest score
            valid_scores = [score if pred else -float('inf') for score, pred in zip(sample_scores, pred_binary)]
            pred_label_idx = np.argmax(valid_scores)
        else:
            # If no label passes threshold, pick the one with highest score
            pred_label_idx = np.argmax(sample_scores)
        
        # Convert true labels to single label
        if sum(sample_true) > 0:
            true_label_idx = np.argmax(sample_true)
        else:
            true_label_idx = len(sample_true) - 1  # Default to last label (typically 'neither')
        
        predictions.append(pred_label_idx)
        true_binary.append(true_label_idx)
    
    # Calculate macro F1-score
    f1_macro = f1_score(true_binary, predictions, average='macro', zero_division=0)
    
    # Calculate macro precision
    precision_macro = precision_recall_fscore_support(
        true_binary, predictions, average='macro', zero_division=0
    )[0]
    
    # Combined metric
    combined_score = f1_weight * f1_macro + precision_weight * precision_macro
    
    return combined_score

def grid_search_thresholds(scores, true_labels, step=0.1, f1_weight=0.5, verbose=False):
    """
    Find optimal thresholds using grid search
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        step (float): Step size for grid search
        f1_weight (float): Weight for F1-score vs accuracy
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (best_thresholds, best_score)
    """
    num_labels = len(scores[0])
    
    # Create threshold grid
    threshold_range = np.arange(0.1, 1.0, step)
    
    best_score = -1
    best_thresholds = [0.5] * num_labels
    
    # Generate all combinations of thresholds
    threshold_combinations = list(itertools.product(threshold_range, repeat=num_labels))
    
    if verbose:
        print(f"Grid search: testing {len(threshold_combinations)} combinations...")
    
    iterator = tqdm(threshold_combinations, desc="Grid search") if verbose else threshold_combinations
    
    for thresholds in iterator:
        score = calculate_accuracy_f1_score(scores, true_labels, thresholds, f1_weight)
        
        if score > best_score:
            best_score = score
            best_thresholds = list(thresholds)
    
    return best_thresholds, best_score

def coordinate_descent_thresholds_macro_f1(scores, true_labels, step=0.05, tol=1e-3, max_iter=20, 
                                         f1_weight=0.8, precision_weight=0.2, verbose=False):
    """
    Find optimal thresholds using coordinate descent with macro F1 + precision objective
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        step (float): Step size for coordinate descent
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        f1_weight (float): Weight for macro F1-score
        precision_weight (float): Weight for macro precision
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (best_thresholds, best_score)
    """
    num_labels = len(scores[0])
    
    # Initialize thresholds
    thresholds = [0.5] * num_labels
    
    best_score = calculate_macro_f1_precision_score(scores, true_labels, thresholds, f1_weight, precision_weight)
    
    if verbose:
        print(f"Initial score: {best_score:.4f}")
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to improve each threshold
        for i in range(num_labels):
            current_threshold = thresholds[i]
            best_threshold_for_label = current_threshold
            best_score_for_label = best_score
            
            # Try different threshold values for this label
            for direction in [-1, 1]:
                test_threshold = current_threshold + direction * step
                if 0.0 <= test_threshold <= 1.0:
                    test_thresholds = thresholds.copy()
                    test_thresholds[i] = test_threshold
                    
                    score = calculate_macro_f1_precision_score(scores, true_labels, test_thresholds, f1_weight, precision_weight)
                    
                    if score > best_score_for_label:
                        best_threshold_for_label = test_threshold
                        best_score_for_label = score
                        improved = True
            
            thresholds[i] = best_threshold_for_label
            best_score = best_score_for_label
        
        if verbose:
            print(f"Iteration {iteration + 1}: score = {best_score:.4f}")
        
        if not improved or best_score < tol:
            break
    
    return thresholds, best_score

def differential_evolution_thresholds_macro_f1(scores, true_labels, f1_weight=0.8, precision_weight=0.2, verbose=False):
    """
    Find optimal thresholds using differential evolution with macro F1 + precision objective
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        f1_weight (float): Weight for macro F1-score
        precision_weight (float): Weight for macro precision
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (best_thresholds, best_score)
    """
    num_labels = len(scores[0])
    
    def objective(thresholds):
        # Differential evolution minimizes, so we return negative score
        return -calculate_macro_f1_precision_score(scores, true_labels, thresholds, f1_weight, precision_weight)
    
    # Define bounds for each threshold (0.1 to 0.9)
    bounds = [(0.1, 0.9) for _ in range(num_labels)]
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        disp=verbose
    )
    
    best_thresholds = result.x.tolist()
    best_score = -result.fun  # Convert back to positive score
    
    return best_thresholds, best_score

def coordinate_descent_thresholds(scores, true_labels, step=0.05, tol=1e-3, max_iter=20, f1_weight=0.5, verbose=False):
    """
    Find optimal thresholds using coordinate descent
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        step (float): Step size for coordinate descent
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        f1_weight (float): Weight for F1-score vs accuracy
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (best_thresholds, best_score)
    """
    num_labels = len(scores[0])
    
    # Initialize thresholds
    thresholds = [0.5] * num_labels
    best_score = calculate_accuracy_f1_score(scores, true_labels, thresholds, f1_weight)
    
    if verbose:
        print(f"Initial score: {best_score:.4f}")
    
    for iteration in range(max_iter):
        improved = False
        
        # Optimize each threshold in turn
        for label_idx in range(num_labels):
            current_threshold = thresholds[label_idx]
            best_label_threshold = current_threshold
            best_label_score = best_score
            
            # Try different threshold values for this label
            for threshold in np.arange(0.05, 0.95, step):
                test_thresholds = thresholds.copy()
                test_thresholds[label_idx] = threshold
                
                score = calculate_accuracy_f1_score(scores, true_labels, test_thresholds, f1_weight)
                
                if score > best_label_score:
                    best_label_score = score
                    best_label_threshold = threshold
                    improved = True
            
            thresholds[label_idx] = best_label_threshold
            best_score = best_label_score
        
        if verbose:
            print(f"Iteration {iteration + 1}: score = {best_score:.4f}")
        
        # Check for convergence
        if not improved:
            if verbose:
                print("Converged: no improvement found")
            break
    
    return thresholds, best_score

def differential_evolution_thresholds(scores, true_labels, num_labels, f1_weight=0.5, verbose=False):
    """
    Find optimal thresholds using differential evolution
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        num_labels (int): Number of labels
        f1_weight (float): Weight for F1-score vs accuracy
        verbose (bool): Whether to print progress
    
    Returns:
        tuple: (best_thresholds, best_score)
    """
    
    def objective_function(thresholds):
        # Minimize negative score (since DE minimizes)
        return -calculate_accuracy_f1_score(scores, true_labels, thresholds, f1_weight)
    
    # Define bounds for each threshold (between 0.05 and 0.95)
    bounds = [(0.05, 0.95) for _ in range(num_labels)]
    
    # Run differential evolution
    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        disp=verbose
    )
    
    best_thresholds = result.x.tolist()
    best_score = -result.fun  # Convert back from negative
    
    return best_thresholds, best_score

def apply_optimal_thresholds(scores, optimal_thresholds, sector, label_names):
    """
    Apply optimal thresholds to prediction scores
    
    Args:
        scores (list): Raw prediction scores
        optimal_thresholds (dict): Dictionary of optimal thresholds by sector
        sector (str): Current sector name
        label_names (list): List of label names
    
    Returns:
        list: Binary predictions after applying thresholds
    """
    if sector not in optimal_thresholds:
        # Default threshold of 0.5
        return [1 if score >= 0.5 else 0 for score in scores]
    
    sector_thresholds = optimal_thresholds[sector]
    binary_predictions = []
    
    for i, (score, label) in enumerate(zip(scores, label_names)):
        threshold = sector_thresholds.get(label, 0.5)
        binary_predictions.append(1 if score >= threshold else 0)
    
    return binary_predictions

def find_optimal_thresholds_accuracy_f1_global(sector_models, all_data, save_path='optimal_thresholds_accuracy_f1.json',
                                               device='cpu', verbose=True, f1_weight=0.5, method='auto'):
    """
    Find optimal thresholds for each label in each sector using accuracy and F1-score optimization
    
    Args:
        sector_models (dict): Dictionary with sector names as keys and model info as values
        all_data (dict): Dictionary with sector data
        save_path (str): Path to save the optimal thresholds JSON file
        device (str): Device to run predictions on ('cpu' or 'cuda')
        verbose (bool): Whether to print detailed progress information
        f1_weight (float): Weight for F1-score vs accuracy (0.5 = equal weight)
        method (str): Optimization method ('grid', 'coordinate', 'differential', 'auto')
    
    Returns:
        dict: Dictionary containing optimal thresholds for each sector
    """
    if verbose:
        print("\n" + "="*60)
        print("FINDING OPTIMAL THRESHOLDS USING ACCURACY + F1-SCORE OPTIMIZATION")
        print("="*60)
        print(f"F1-score weight: {f1_weight:.2f}, Accuracy weight: {1-f1_weight:.2f}")
    
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
        
        best_thresholds = [0.5] * num_labels
        best_score = 0
        best_method = None
        
        # Try different optimization methods
        methods_to_try = []
        if method == 'auto':
            methods_to_try = ['coordinate', 'differential']
            if num_labels <= 3:  # Only use grid search for small number of labels
                methods_to_try.insert(0, 'grid')
        else:
            methods_to_try = [method]
        
        for opt_method in methods_to_try:
            if verbose:
                print(f"\nUsing {opt_method} optimization:")
            
            try:
                if opt_method == 'grid':
                    thresholds, score = grid_search_thresholds(
                        all_scores, all_true_labels, step=0.1, f1_weight=f1_weight, verbose=verbose
                    )
                elif opt_method == 'coordinate':
                    thresholds, score = coordinate_descent_thresholds(
                        all_scores, all_true_labels, step=0.05, f1_weight=f1_weight, verbose=verbose
                    )
                elif opt_method == 'differential':
                    thresholds, score = differential_evolution_thresholds(
                        all_scores, all_true_labels, num_labels, f1_weight=f1_weight, verbose=verbose
                    )
                
                if verbose:
                    print(f"{opt_method.capitalize()} result: Combined score = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
                    best_method = opt_method
                    
            except Exception as e:
                if verbose:
                    print(f"Error with {opt_method} method: {e}")
                continue
        
        # Store best thresholds with label names
        optimal_thresholds[sector] = {
            label: threshold for label, threshold in zip(label_names, best_thresholds)
        }
        
        if verbose:
            print(f"\nBest method: {best_method} (Combined score: {best_score:.4f})")
            print(f"Best thresholds for {sector}:")
            for label, threshold in optimal_thresholds[sector].items():
                print(f"  {label}: {threshold:.3f}")
    
    # Save thresholds to file
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        if verbose:
            print(f"\nOptimal thresholds saved to '{save_path}'")
    
    return optimal_thresholds

def evaluate_thresholds_accuracy_f1(scores, true_labels, thresholds, label_names, verbose=False):
    """
    Evaluate thresholds using accuracy and F1-score metrics
    
    Args:
        scores (list): List of prediction scores for each sample
        true_labels (list): List of true binary labels for each sample
        thresholds (list): List of thresholds for each label
        label_names (list): List of label names
        verbose (bool): Whether to print detailed results
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Apply thresholds to get predictions
    predictions = []
    true_single = []
    
    for sample_scores, sample_true in zip(scores, true_labels):
        # Apply thresholds to get binary predictions
        pred_binary = [1 if score >= thresh else 0 for score, thresh in zip(sample_scores, thresholds)]
        
        # For single-label classification: find the label with highest score among those above threshold
        if sum(pred_binary) > 0:
            # Among labels that pass threshold, pick the one with highest score
            valid_scores = [score if pred else -float('inf') for score, pred in zip(sample_scores, pred_binary)]
            pred_label_idx = np.argmax(valid_scores)
        else:
            # If no label passes threshold, pick the one with highest score
            pred_label_idx = np.argmax(sample_scores)
        
        if sum(sample_true) > 0:
            true_label_idx = np.argmax(sample_true)
        else:
            true_label_idx = len(sample_true) - 1
        
        predictions.append(pred_label_idx)
        true_single.append(true_label_idx)
    
    # Calculate metrics
    accuracy = accuracy_score(true_single, predictions)
    
    # Calculate per-class and averaged F1-scores
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        true_single, predictions, average=None, zero_division=0, labels=list(range(len(label_names)))
    )
    
    f1_macro = f1_score(true_single, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(true_single, predictions, average='weighted', zero_division=0)
    f1_micro = f1_score(true_single, predictions, average='micro', zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'per_class_f1': dict(zip(label_names, f1_per_class)),
        'per_class_precision': dict(zip(label_names, precision)),
        'per_class_recall': dict(zip(label_names, recall)),
        'per_class_support': dict(zip(label_names, support)),
        'combined_score_equal': 0.5 * accuracy + 0.5 * f1_macro,
        'combined_score_f1_heavy': 0.3 * accuracy + 0.7 * f1_macro
    }
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"F1-weighted: {f1_weighted:.4f}")
        print(f"F1-micro: {f1_micro:.4f}")
        print(f"Combined (equal): {results['combined_score_equal']:.4f}")
        print(f"Combined (F1-heavy): {results['combined_score_f1_heavy']:.4f}")
        
        print("\nPer-class metrics:")
        for i, label in enumerate(label_names):
            print(f"  {label}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1_per_class[i]:.3f}, Support={int(support[i])}")
    
    return results