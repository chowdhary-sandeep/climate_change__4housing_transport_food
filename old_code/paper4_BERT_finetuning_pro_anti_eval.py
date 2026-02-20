"""
Evaluate trained pro-anti BERT models on test data
"""

# Import all functions from the original training file
from paper4_BERT_finetuning_pro_anti import (
    load_pro_anti_test_data,
    load_pro_anti_validation_data,
    evaluate_pro_anti_model,
    ProAntiClassifier,
    get_tokenizer,
    convert_numpy_types,
    predict_pro_anti_scores
)

# Import threshold optimization functions - using accuracy+F1 based optimization
from paper4_accuracy_f1_threshold_optimizer import (
    apply_optimal_thresholds,
    coordinate_descent_thresholds,
    differential_evolution_thresholds,
    coordinate_descent_thresholds_macro_f1,
    differential_evolution_thresholds_macro_f1,
    predict_with_scores
)

import torch
import json
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def optimize_thresholds_on_validation(model, tokenizer, label_names, validation_data, sector, device):
    """Optimize thresholds using validation data"""
    if sector not in validation_data or not validation_data[sector]:
        print(f"No validation data available for {sector}")
        return {}
    
    print(f"Optimizing thresholds for {sector} using validation data...")
    
    # Get validation data for this sector
    sector_validation_data = validation_data[sector]
    texts = list(sector_validation_data.keys())
    true_labels_list = list(sector_validation_data.values())
    
    # Filter validation data to only include labels that the model knows about
    filtered_texts = []
    filtered_true_labels = []
    
    for text, true_labels in zip(texts, true_labels_list):
        # Filter labels to only include those the model knows about
        filtered_labels = [label for label in true_labels if label in label_names]
        if filtered_labels:  # Only include if at least one known label
            filtered_texts.append(text)
            filtered_true_labels.append(filtered_labels)
    
    if not filtered_texts:
        print(f"No valid validation data for {sector}")
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
    
    # Run threshold optimization using coordinate descent for macro F1 + precision
    thresholds, combined_score = coordinate_descent_thresholds_macro_f1(
        all_scores,
        all_true_labels,
        step=0.05,
        tol=1e-3,
        max_iter=15,
        f1_weight=0.8,
        precision_weight=0.2,
        verbose=False
    )
    
    # Store thresholds with label names
    optimal_thresholds = {
        label: threshold for label, threshold in zip(label_names, thresholds)
    }
    
    print(f"Optimal thresholds for {sector}: {optimal_thresholds}")
    return optimal_thresholds

def generate_confusion_matrix(all_true_labels, all_predictions, label_names, sector, save_dir='paper4figs'):
    """Generate and save confusion matrix for a sector"""
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=label_names)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {sector.upper()} Sector', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f'confusion_matrix_{sector}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {filepath}")
    
    return cm

def collect_all_examples(model, tokenizer, label_names, test_data, sector, optimal_thresholds, device):
    """Collect all examples (both correctly and incorrectly labeled) from test data"""
    if sector not in test_data or not test_data[sector]:
        return []
    
    sector_test_data = test_data[sector]
    texts = list(sector_test_data.keys())
    true_labels_list = list(sector_test_data.values())
    
    all_examples = []
    
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
        return []
    
    model.eval()
    for text, true_labels in zip(filtered_texts, filtered_true_labels):
        # Get predictions
        results = predict_pro_anti_scores(text, model, tokenizer, label_names, device)
        scores = [res['score'] for res in results]
        
        # True labels as binary vector
        true_binary = [1 if label in true_labels else 0 for label in label_names]
        
        # Apply optimal thresholds
        if optimal_thresholds:
            # optimal_thresholds is already the sector-specific thresholds
            pred_binary = apply_optimal_thresholds(scores, {sector: optimal_thresholds}, sector, label_names)
        else:
            # Default threshold of 0.5
            pred_binary = [1 if score >= 0.5 else 0 for score in scores]
        
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
        
        # Check if prediction is correct
        is_correct = predicted_label == true_label
        
        # Store all examples (both correct and incorrect)
        all_examples.append({
            'sector': sector,
            'comment': text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'score_pro': float(scores[label_names.index('pro')]) if 'pro' in label_names else 0.0,
            'score_anti': float(scores[label_names.index('anti')]) if 'anti' in label_names else 0.0,

            'score_neither': float(scores[label_names.index('neither')]) if 'neither' in label_names else 0.0,
            'threshold_pro': optimal_thresholds.get('pro', 0.5),
            'threshold_anti': optimal_thresholds.get('anti', 0.5),

            'threshold_neither': optimal_thresholds.get('neither', 0.5)
        })
    
    return all_examples



def evaluate_trained_models_on_test_data():
    """Evaluate trained pro-anti models on test data"""
    print("Starting evaluation...")
    try:
        # Dictionary to store all examples (both correct and incorrect)
        all_examples = {}
        
        # Load test data
        test_data = load_pro_anti_test_data()
        if not test_data:
            print("No test data found!")
            return
        
        # Load validation data for threshold optimization
        validation_data = load_pro_anti_validation_data()
        if not validation_data:
            print("No validation data found! Will use default thresholds.")
            validation_data = {}
        
        total_test = sum(len(sector_data) for sector_data in test_data.values())
        total_validation = sum(len(sector_data) for sector_data in validation_data.values()) if validation_data else 0
        print(f"Loaded: {total_test} test samples, {total_validation} validation samples")
        
        # Look for trained models (prioritize RoBERTa models)
        model_dirs = glob.glob('models/pro_anti_*_roberta_base')
        if not model_dirs:
            # Fallback to DistilBERT models if RoBERTa not found
            model_dirs = glob.glob('models/pro_anti_*_distilbert_base_uncased')
        print(f"Found model directories: {model_dirs}")
        
        if not model_dirs:
            print("No trained models found! Please train models first.")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        all_test_results = {}
        
        for model_dir in model_dirs:
            # Extract sector name from path like "pro_anti_food_distilbert_base_uncased"
            sector = model_dir.split('_')[2]  # Get the sector name (food, housing, transport)
            print(f"\n{'='*50}")
            print(f"Evaluating {sector.upper()} model")
            print(f"{'='*50}")
            
            # Load best model
            best_model_path = os.path.join(model_dir, 'best_model', 'model.pt')
            if not os.path.exists(best_model_path):
                print(f"Best model not found at {best_model_path}")
                continue
            
            # Load model checkpoint
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model_name = checkpoint.get('model_name', 'distilbert-base-uncased')
            label_names = checkpoint['label_names']
            
            print(f"Model: {model_name}")
            print(f"Labels: {label_names}")
            print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
            combined_score = checkpoint.get('combined_accuracy_f1', 'unknown')
            if isinstance(combined_score, (int, float)):
                print(f"Best Combined Score: {combined_score:.3f}")
            else:
                print(f"Best Combined Score: {combined_score}")
            
            # Initialize model
            model = ProAntiClassifier(model_name, len(label_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # Load tokenizer
            tokenizer = get_tokenizer(model_name)
            tokenizer_path = os.path.join(model_dir, 'best_model')
            if os.path.exists(tokenizer_path):
                tokenizer = tokenizer.from_pretrained(tokenizer_path)
            
            # Get optimal thresholds from checkpoint (from training)
            checkpoint_thresholds = checkpoint.get('optimal_thresholds', {})
            if checkpoint_thresholds:
                print(f"Checkpoint optimal thresholds: {checkpoint_thresholds}")
            
            # Optimize thresholds on validation data if available
            if validation_data and sector in validation_data:
                optimized_thresholds = optimize_thresholds_on_validation(
                    model, tokenizer, label_names, validation_data, sector, device
                )
                if optimized_thresholds:
                    optimal_thresholds = optimized_thresholds
                    print(f"Using validation-optimized thresholds: {optimal_thresholds}")
                else:
                    optimal_thresholds = checkpoint_thresholds
                    print(f"Using checkpoint thresholds: {optimal_thresholds}")
            else:
                optimal_thresholds = checkpoint_thresholds
                print(f"Using checkpoint thresholds: {optimal_thresholds}")
            
            # Evaluate on test data
            if sector in test_data and test_data[sector]:
                # Debug: Print the optimal thresholds being used
                print(f"DEBUG: Using optimal thresholds for {sector}: {optimal_thresholds}")
                
                # Collect all examples (both correct and incorrect)
                sector_examples = collect_all_examples(
                    model, tokenizer, label_names, test_data, sector, optimal_thresholds, device
                )
                all_examples[sector] = sector_examples
                
                # Count mislabeled examples
                mislabeled_count = sum(1 for ex in sector_examples if not ex['is_correct'])
                print(f"Found {mislabeled_count} mislabeled examples out of {len(sector_examples)} total for {sector}")
                
                # Run 3-label evaluation
                test_metrics = evaluate_pro_anti_model(
                    model, tokenizer, label_names, test_data, sector, optimal_thresholds, device
                )
                
                # Generate confusion matrix
                if test_metrics and 'all_predictions' in test_metrics and 'all_true_labels' in test_metrics:
                    confusion_matrix_data = generate_confusion_matrix(
                        test_metrics['all_true_labels'], 
                        test_metrics['all_predictions'], 
                        label_names, 
                        sector
                    )
                    print(f"Confusion matrix generated for {sector} sector")
                
                if test_metrics:
                    print(f"\n{sector.upper()} TEST RESULTS:")
                    print(f"CombinedScore={test_metrics['combined_accuracy_f1']:.3f} "
                          f"MacroF1={test_metrics['macro_f1_score']:.3f} "
                          f"F1={test_metrics['f1']:.3f} "
                          f"Acc={test_metrics['accuracy']:.3f} "
                          f"P={test_metrics['precision']:.3f} "
                          f"R={test_metrics['recall']:.3f} "
                          f"WeightedF1={test_metrics['weighted_f1']:.3f} "
                          f"W-P={test_metrics['weighted_precision']:.3f} "
                          f"W-R={test_metrics['weighted_recall']:.3f} "
                          f"n={test_metrics['num_samples']}")
                    
                    # Store results
                    all_test_results[sector] = {
                        'model_name': model_name,
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'best_train_combined_score': checkpoint.get('combined_accuracy_f1', 'unknown'),
                        'optimal_thresholds': optimal_thresholds,
                        **convert_numpy_types(test_metrics)
                    }
                    
                    # Print per-class metrics
                    print(f"\nPer-class metrics:")
                    for label in label_names:
                        print(f"  {label}: P={test_metrics['per_class_precision'][label]:.3f} "
                              f"R={test_metrics['per_class_recall'][label]:.3f} "
                              f"F1={test_metrics['per_class_f1'][label]:.3f} "
                              f"Support={test_metrics['per_class_support'][label]}")
                else:
                    print(f"No test metrics available for {sector}")
            else:
                print(f"No test data available for {sector}")
        
        # Save all test results
        if all_test_results:
            os.makedirs('results', exist_ok=True)
            
            # Save 3-label results
            test_results_path = 'results/pro_anti_test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(all_test_results, f, indent=2)
            print(f"\n3-label test results saved to {test_results_path}")
        
        # Save all examples to CSV
        if all_examples:
            total_examples = sum(len(examples) for examples in all_examples.values())
            print(f"\nSaving {total_examples} total examples to CSV...")
            
            # Combine all examples from all sectors into one list
            all_examples_combined = []
            for sector, examples in all_examples.items():
                all_examples_combined.extend(examples)
            
            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(all_examples_combined)
            
            # Save to paper4data directory
            csv_path = 'paper4data/pro_anti_all_examples.csv'
            df.to_csv(csv_path, index=False)
            print(f"All examples saved to {csv_path}")
            
            # Print summary of examples by sector
            print(f"\nExamples by sector:")
            for sector, examples in all_examples.items():
                mislabeled_count = sum(1 for ex in examples if not ex['is_correct'])
                correct_count = sum(1 for ex in examples if ex['is_correct'])
                print(f"  {sector.upper()}: {len(examples)} total ({correct_count} correct, {mislabeled_count} mislabeled)")
                
                # Show some examples of misclassifications
                mislabeled_examples = [ex for ex in examples if not ex['is_correct']]
                if mislabeled_examples:
                    print(f"    Sample misclassifications:")
                    for i, example in enumerate(mislabeled_examples[:3]):  # Show first 3
                        print(f"      {i+1}. True: {example['true_label']} -> Predicted: {example['predicted_label']}")
                        print(f"         Comment: {example['comment'][:100]}...")
            
            # Print comprehensive summary
            print(f"\n{'='*80}")
            print(f"3-LABEL TEST RESULTS SUMMARY")
            print(f"{'='*80}")
            print(f"{'Sector':<12} {'CombinedScore':<13} {'MacroF1':<10} {'WeightedF1':<12} {'Accuracy':<10} {'n':<5}")
            print(f"{'-'*80}")
            
            for sector in all_test_results.keys():
                results = all_test_results[sector]
                print(f"{sector.upper():<12} "
                      f"{results['combined_accuracy_f1']:.3f} "
                      f"{results['macro_f1_score']:.3f} "
                      f"{results['weighted_f1']:.3f} "
                      f"{results['accuracy']:.3f} "
                      f"{results['num_samples']}")
                
            print(f"{'-'*80}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run test evaluation
    evaluate_trained_models_on_test_data() 