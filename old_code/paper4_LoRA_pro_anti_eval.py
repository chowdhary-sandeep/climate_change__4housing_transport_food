"""
Evaluate LoRA-adapter pro/anti/neither models saved by paper4_LoRA_pro_anti.py

Loads sector-specific adapters from models_lora/pro_anti_<sector>_lora_<base>/
and evaluates on test data. Thresholds are optimized on validation data
using macro-F1 + precision objective (coordinate descent), mirroring the
baseline evaluation script semantics.
"""

from __future__ import annotations

import os
import glob
import json
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Reuse data IO and labels
from paper4_BERT_finetuning_pro_anti import (
    load_pro_anti_test_data,
    load_pro_anti_validation_data,
    PRO_ANTI_LABELS,
)

from paper4_accuracy_f1_threshold_optimizer import (
    coordinate_descent_thresholds_macro_f1,
    apply_optimal_thresholds,
)


@torch.no_grad()
def predict_probs(text: str, model, tokenizer, device: torch.device, max_length: int = 128) -> List[float]:
    """Return softmax probabilities for labels in PRO_ANTI_LABELS order."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.detach().cpu().numpy()[0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs.tolist()


def optimize_thresholds_on_validation(model, tokenizer, label_names, validation_data, sector, device):
    """Coordinate descent on validation data to pick per-label thresholds."""
    if sector not in validation_data or not validation_data[sector]:
        return {}
    texts = list(validation_data[sector].keys())
    true_lists = list(validation_data[sector].values())

    all_scores = []
    all_true = []
    for text, true_labels in zip(texts, true_lists):
        probs = predict_probs(text, model, tokenizer, device)
        all_scores.append(probs)
        true_vec = [1 if ln in true_labels else 0 for ln in label_names]
        all_true.append(true_vec)

    thresholds, _ = coordinate_descent_thresholds_macro_f1(
        all_scores,
        all_true,
        step=0.05,
        tol=1e-3,
        max_iter=15,
        f1_weight=0.8,
        precision_weight=0.2,
        verbose=False,
    )

    return {label: thr for label, thr in zip(label_names, thresholds)}


def generate_confusion_matrix(all_true_labels, all_predictions, label_names, sector, save_dir='paper4figs_lora'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_true_labels, all_predictions, labels=label_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {sector.upper()} (LoRA)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_lora_{sector}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def evaluate_sector(model, tokenizer, label_names, test_data, sector, optimal_thresholds, device) -> Dict:
    if sector not in test_data or not test_data[sector]:
        return {}
    texts = list(test_data[sector].keys())
    true_lists = list(test_data[sector].values())

    y_true_idx = []
    y_pred_idx = []

    for text, true_labels in zip(texts, true_lists):
        probs = predict_probs(text, model, tokenizer, device)
        # Apply thresholds; fallback to argmax if nothing selected
        if optimal_thresholds:
            pred_binary = apply_optimal_thresholds(probs, {sector: optimal_thresholds}, sector, label_names)
        else:
            pred_binary = [1 if p >= 0.5 else 0 for p in probs]
        if sum(pred_binary) == 0:
            pred_binary[int(np.argmax(probs))] = 1
        pred_idx = int(np.argmax(pred_binary))

        # True label as single index (argmax, default to 'neither')
        true_vec = [1 if ln in true_labels else 0 for ln in label_names]
        true_idx = int(np.argmax(true_vec)) if sum(true_vec) > 0 else label_names.index('neither')

        y_true_idx.append(true_idx)
        y_pred_idx.append(pred_idx)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(range(len(label_names))), average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(range(len(label_names))), average='weighted', zero_division=0
    )

    per_class = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(range(len(label_names))), average=None, zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = per_class

    # Map to label names
    per_class_precision = {ln: float(per_class_precision[i]) for i, ln in enumerate(label_names)}
    per_class_recall = {ln: float(per_class_recall[i]) for i, ln in enumerate(label_names)}
    per_class_f1 = {ln: float(per_class_f1[i]) for i, ln in enumerate(label_names)}
    per_class_support = {ln: int(per_class_support[i]) for i, ln in enumerate(label_names)}

    # Also create confusion matrix figure
    cm_path = generate_confusion_matrix(
        [label_names[i] for i in y_true_idx],
        [label_names[i] for i in y_pred_idx],
        label_names,
        sector,
    )

    return {
        'accuracy': float(acc),
        'macro_f1': float(f1_macro),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'weighted_f1': float(f1_weighted),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'num_samples': len(texts),
        'confusion_matrix_path': cm_path,
    }


def load_lora_model_for_sector(save_dir: str, base_model_name: str, num_labels: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    base = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    adapter_dir = os.path.join(save_dir, 'adapter')
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_lora_models():
    print("Evaluating LoRA adapters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    validation_data = load_pro_anti_validation_data()
    test_data = load_pro_anti_test_data()
    label_names = PRO_ANTI_LABELS

    # Find trained adapters
    model_dirs = glob.glob('models_lora/pro_anti_*_lora_*')
    if not model_dirs:
        print("No LoRA adapter directories found under models_lora/")
        return

    all_results: Dict[str, Dict] = {}

    for save_dir in model_dirs:
        # Expect dir names like: models_lora/pro_anti_food_lora_roberta_base
        try:
            parts = os.path.basename(save_dir).split('_')
            sector = parts[2]
        except Exception:
            print(f"Skipping unrecognized path: {save_dir}")
            continue

        # Infer base model name from tail if possible; fallback to roberta-base
        tail = os.path.basename(save_dir)
        if 'roberta_base' in tail:
            base_model_name = 'roberta-base'
        elif 'distilbert_base_uncased' in tail:
            base_model_name = 'distilbert-base-uncased'
        elif 'bert_base_uncased' in tail:
            base_model_name = 'bert-base-uncased'
        else:
            base_model_name = 'roberta-base'

        print(f"\nEvaluating {sector.upper()} (adapter: {save_dir}, base: {base_model_name})")
        try:
            model, tokenizer = load_lora_model_for_sector(save_dir, base_model_name, num_labels=len(label_names), device=device)
        except Exception as e:
            print(f"Failed to load model for {sector}: {e}")
            continue

        # Optimize thresholds on validation (sector-specific)
        optimal_thresholds = {}
        if validation_data and sector in validation_data:
            try:
                optimal_thresholds = optimize_thresholds_on_validation(model, tokenizer, label_names, validation_data, sector, device)
                print(f"Optimal thresholds ({sector}): {optimal_thresholds}")
            except Exception as e:
                print(f"Threshold optimization failed for {sector}: {e}")

        # Evaluate on test
        metrics = evaluate_sector(model, tokenizer, label_names, test_data, sector, optimal_thresholds, device)
        if metrics:
            print(
                f"{sector.upper()} LoRA TEST => Acc={metrics['accuracy']:.3f} MacroF1={metrics['macro_f1']:.3f} "
                f"WeightedF1={metrics['weighted_f1']:.3f} n={metrics['num_samples']}"
            )
        else:
            print(f"No test data for {sector}")
        all_results[sector] = metrics

    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = 'results/pro_anti_lora_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved LoRA evaluation results to {out_path}")


if __name__ == "__main__":
    evaluate_lora_models()


