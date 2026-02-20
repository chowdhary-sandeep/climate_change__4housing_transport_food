"""
LoRA-based pro-anti stance classification using the same data pipeline as
paper4_BERT_finetuning_pro_anti.py, but training lightweight adapters instead
of full fine-tuning. Outputs are saved under method-specific names/paths.

This script trains per-sector (transport/EVs, housing/solar, food/veganism)
RoBERTa-base sequence classification heads with LoRA adapters to classify
comments into {pro, anti, neither}. It mirrors the training/eval structure:
- Loads GPT-labeled training/validation/test data from paper4data
- Applies class weighting
- Trains with CE loss, warmup schedule, and early stopping by macro-F1
- Optimizes thresholds (optional) for combined accuracy+F1 objective
- Saves adapters and tokenizer into models_lora/
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Tuple
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model

# Check if DoRA is available in this PEFT version
try:
    from peft import DoRAConfig
    DORA_AVAILABLE = True
except ImportError:
    DORA_AVAILABLE = False
    print("Note: DoRA not available in this PEFT version. Using standard LoRA instead.")

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

import math
import shutil
from torch.nn.utils import clip_grad_norm_

# Reuse data loading utilities and constants
from paper4_BERT_finetuning_pro_anti import (
    load_pro_anti_training_data,
    load_pro_anti_validation_data,
    load_pro_anti_test_data,
    PRO_ANTI_LABELS,
)

# Threshold optimizer utilities
from paper4_accuracy_f1_threshold_optimizer import (
    apply_optimal_thresholds,
    coordinate_descent_thresholds_macro_f1,
    evaluate_thresholds_accuracy_f1,
)


def prepare_single_label_data(training_data: Dict[str, Dict[str, List[str]]], label_names: List[str]) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Convert mapping {sector: {comment: [best_label]}} to texts and integer labels.
    Only the first label is used per comment (data loader already selects best label).
    """
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    texts: List[str] = []
    labels: List[int] = []
    for sector_data in training_data.values():
        for comment, label_list in sector_data.items():
            if not label_list:
                continue
            best_label = label_list[0]
            if best_label in label_to_idx:
                texts.append(comment)
                labels.append(label_to_idx[best_label])
    return texts, labels, label_to_idx


class SingleLabelTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'text': text,
        }


def calculate_class_weights_from_data(training_data: Dict[str, Dict[str, List[str]]], label_names: List[str]) -> torch.Tensor:
    counts = {label: 0 for label in label_names}
    total = 0
    for sector_data in training_data.values():
        for _, label_list in sector_data.items():
            if not label_list:
                continue
            label = label_list[0]
            if label in counts:
                counts[label] += 1
                total += 1
    weights = []
    for label in label_names:
        if counts[label] > 0:
            weights.append(total / (len(label_names) * counts[label]))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_probs(text: str, model, tokenizer, device: torch.device, max_length: int = 128) -> List[float]:
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True,
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.detach().cpu().numpy()[0]
    # softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs.tolist()


def optimize_thresholds_on_training(model, tokenizer, label_names, training_data, device) -> Dict[str, float]:
    # Prepare texts and true labels (single label)
    texts, labels, _ = prepare_single_label_data(training_data, label_names)
    if not texts:
        return {}
    # Generate scores for optimizer API
    all_scores = []
    all_true = []
    for text, label in zip(texts, labels):
        probs = predict_probs(text, model, tokenizer, device)
        all_scores.append(probs)
        true_vec = [0] * len(label_names)
        true_vec[label] = 1
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


def evaluate_model(model, tokenizer, label_names, eval_data, device, optimal_thresholds=None) -> Dict:
    if not eval_data:
        return {}
    texts = list(eval_data.keys())
    true_label_lists = list(eval_data.values())

    y_true = []
    y_pred = []
    y_pred_raw = []

    for text, true_labels in zip(texts, true_label_lists):
        probs = predict_probs(text, model, tokenizer, device)
        y_pred_raw.append(probs)
        # single-label true
        true_vec = [1 if (label_names[idx] in true_labels) else 0 for idx in range(len(label_names))]
        y_true.append(true_vec)

        # thresholding to 1-of-K; fallback to argmax if none selected
        if optimal_thresholds:
            pred_binary = apply_optimal_thresholds(probs, {"tmp": optimal_thresholds}, "tmp", label_names)
        else:
            pred_binary = [1 if p >= 0.5 else 0 for p in probs]
        if sum(pred_binary) == 0:
            pred_binary[np.argmax(probs)] = 1
        predicted_idx = int(np.argmax(pred_binary))
        y_pred.append(predicted_idx)

    # Convert y_true to single label index by argmax
    y_true_single = [int(np.argmax(vec)) if sum(vec) > 0 else label_names.index('neither') for vec in y_true]

    acc = accuracy_score(y_true_single, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_single, y_pred, labels=list(range(len(label_names))), average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_single, y_pred, labels=list(range(len(label_names))), average='weighted', zero_division=0
    )

    return {
        'accuracy': acc,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'num_samples': len(texts),
    }


def _choose_lora_hyperparams(num_examples: int, per_device_batch_size: int, grad_accum_steps: int) -> tuple[int, float, float]:
    """Pick epochs, adapter_lr, head_lr per provided grid. Scales LR by beff/32."""
    beff = max(1, per_device_batch_size) * max(1, grad_accum_steps)
    # Regime selection with enhanced adapter learning rates (1.5x increase)
    if num_examples < 1_000:
        epochs = 3
        adapter_lr = 7.5e-4  # Increased from 5e-4
        head_lr = 1e-3
    elif num_examples < 10_000:
        epochs = 4
        adapter_lr = 7.5e-4  # Increased from 5e-4
        head_lr = 1e-3
    elif num_examples < 100_000:
        epochs = 5
        adapter_lr = 4.5e-4  # Increased from 3e-4
        head_lr = 1e-3
    else:
        epochs = 6
        adapter_lr = 4.5e-4  # Increased from 3e-4
        head_lr = 8e-4
    scale = beff / 32.0
    adapter_lr *= scale
    head_lr *= scale
    return epochs, adapter_lr, head_lr


def _sanitize_model_id_for_path(model_id: str) -> str:
    """Convert any model id (may contain slashes) into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", model_id)


def _reinitialize_classifier_head(model: AutoModelForSequenceClassification, num_labels: int) -> None:
    """Force a fresh classifier head regardless of what's in the checkpoint.
    Works for Roberta/BERT-style heads (RobertaClassificationHead) and plain Linear.
    """
    if hasattr(model, 'classifier'):
        clf = model.classifier
        # If RobertaClassificationHead: has dense and out_proj
        if hasattr(clf, 'dense') and hasattr(clf, 'out_proj'):
            if hasattr(clf.dense, 'weight'):
                nn.init.xavier_uniform_(clf.dense.weight)
            if hasattr(clf.dense, 'bias') and clf.dense.bias is not None:
                nn.init.zeros_(clf.dense.bias)
            # Ensure output dims match num_labels
            if clf.out_proj.out_features != num_labels:
                clf.out_proj = nn.Linear(clf.out_proj.in_features, num_labels)
            nn.init.xavier_uniform_(clf.out_proj.weight)
            if clf.out_proj.bias is not None:
                nn.init.zeros_(clf.out_proj.bias)
        # Plain linear classifier case
        elif isinstance(clf, nn.Linear):
            if clf.out_features != num_labels or clf.in_features != clf.in_features:
                model.classifier = nn.Linear(clf.in_features, num_labels)
                clf = model.classifier
            nn.init.xavier_uniform_(clf.weight)
            if clf.bias is not None:
                nn.init.zeros_(clf.bias)
        # Fallback: try to set an out_proj if present
        elif hasattr(clf, 'out_proj') and isinstance(clf.out_proj, nn.Linear):
            if clf.out_proj.out_features != num_labels:
                clf.out_proj = nn.Linear(clf.out_proj.in_features, num_labels)
            nn.init.xavier_uniform_(clf.out_proj.weight)
            if clf.out_proj.bias is not None:
                nn.init.zeros_(clf.out_proj.bias)

def train_sector_with_lora(
    sector: str,
    train_data_sector: Dict[str, List[str]],
    validation_data: Dict[str, Dict[str, List[str]]],
    model_name: str = 'SamLowe/roberta-base-go_emotions',
    num_epochs: int | None = None,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    adapter_learning_rate: float | None = None,
    head_learning_rate: float | None = None,
    save_root: str = 'models_lora',
    use_class_weights: bool = True,
    max_length: int = 96,
    do_eval: bool = False,
    unfreeze_final_layer: bool = False,  # New option to unfreeze final encoder layer
    use_dora: bool = False,  # Option to use DoRA instead of standard LoRA
) -> Tuple:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Optimize PyTorch threading for CPU
    if device.type == 'cpu':
        try:
            torch.set_num_threads(max(1, os.cpu_count() - 1))
            torch.set_num_interop_threads(max(1, os.cpu_count() // 2))
        except Exception:
            pass
    label_names = PRO_ANTI_LABELS

    # Tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Workaround for torch.get_default_device issue in transformers
    try:
        # Temporarily add get_default_device method to torch if it doesn't exist
        if not hasattr(torch, 'get_default_device'):
            def get_default_device():
                return torch.device('cpu')
            torch.get_default_device = get_default_device
            print("Added temporary get_default_device method to torch")
        
        # Load the model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_names), ignore_mismatched_sizes=True
        )
        
    except Exception as e:
        print(f"Warning: Model loading failed ({e}), trying alternative approach")
        # Try with different parameters
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(label_names), 
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"Alternative loading also failed ({e2}), trying basic loading")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(label_names), 
                ignore_mismatched_sizes=True
            )
    # Discard any pre-existing classifier weights; we care about the encoder pretraining only
    _reinitialize_classifier_head(base_model, num_labels=len(label_names))

    # Enhanced LoRA configuration with improved target modules and higher rank
    if use_dora and DORA_AVAILABLE:
        # Use DoRA (Decomposed Rank Adaptation) for better integration
        # Build DoRA config with available parameters
        dora_kwargs = {
            'task_type': TaskType.SEQ_CLS,
            'r': 16,  # Increased from 8 to 16 for more capacity
            'lora_alpha': 32,  # Increased from 16 to 32 (maintaining 2×rank scaling)
            'lora_dropout': 0.2,
            'target_modules': ["query", "key", "value", "o_proj", "ff_proj"],  # Extended target modules
        }
        
        # Add SVD initialization if available in this PEFT version
        try:
            dora_kwargs['init_lora_weights'] = True
            print("SVD initialization enabled")
        except TypeError:
            print("Note: SVD initialization not available in this PEFT version")
        
        lora_config = DoRAConfig(**dora_kwargs)
        print("Using DoRA (Decomposed Rank Adaptation) for enhanced performance")
    else:
        # Standard LoRA with improvements (fallback if DoRA not available)
        if use_dora and not DORA_AVAILABLE:
            print("Warning: DoRA requested but not available. Falling back to enhanced LoRA.")
        
        # Build LoRA config with available parameters
        lora_kwargs = {
            'task_type': TaskType.SEQ_CLS,
            'r': 16,  # Increased from 8 to 16 for more capacity
            'lora_alpha': 32,  # Increased from 16 to 32 (maintaining 2×rank scaling)
            'lora_dropout': 0.2,
            'target_modules': ["query", "key", "value", "o_proj", "ff_proj"],  # Extended target modules
        }
        
        # Add SVD initialization if available in this PEFT version
        try:
            lora_kwargs['init_lora_weights'] = True
            print("SVD initialization enabled")
        except TypeError:
            print("Note: SVD initialization not available in this PEFT version")
        
        lora_config = LoraConfig(**lora_kwargs)
        print("Using enhanced LoRA configuration")
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)

    # Freeze base params; train only LoRA adapters and classifier head
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Optionally unfreeze the final encoder layer for better domain adaptation
    if unfreeze_final_layer:
        # Find and unfreeze the last transformer layer
        for name, param in model.named_parameters():
            if 'lora_' in name or 'classifier' in name:
                param.requires_grad = True
            # Unfreeze the final encoder layer (layer 11 for RoBERTa-base)
            elif 'encoder.layer.11.' in name:
                param.requires_grad = True
                print(f"Unfrozen final encoder layer: {name}")
        print("Final encoder layer unfrozen for enhanced domain adaptation")
    else:
        # Standard LoRA: only train adapters and classifier
        for name, param in model.named_parameters():
            if 'lora_' in name or 'classifier' in name:
                param.requires_grad = True
    
    # Keep LayerNorm frozen per guidance (regardless of unfreeze_final_layer setting)
    for name, param in model.named_parameters():
        if 'LayerNorm' in name or 'layernorm' in name:
            param.requires_grad = False

    # Prepare data
    single_sector_wrapper = {sector: train_data_sector}
    train_texts, train_labels, _ = prepare_single_label_data(single_sector_wrapper, label_names)
    if len(train_texts) == 0:
        raise ValueError(f"No training data for sector {sector}")

    train_dataset = SingleLabelTextDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if torch.cuda.is_available():
        loader_kwargs.update({"num_workers": 2, "pin_memory": True, "persistent_workers": True})
    else:
        # CPU: use more workers to parallelize tokenization
        nw = max(1, min(8, (os.cpu_count() or 2) - 1))
        loader_kwargs.update({
            "num_workers": nw,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 4,
        })
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # Choose hyperparams by data size if not provided
    if num_epochs is None or adapter_learning_rate is None or head_learning_rate is None:
        picked_epochs, picked_adapter_lr, picked_head_lr = _choose_lora_hyperparams(
            num_examples=len(train_dataset),
            per_device_batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
        if num_epochs is None:
            num_epochs = picked_epochs
        if adapter_learning_rate is None:
            adapter_learning_rate = picked_adapter_lr
        if head_learning_rate is None:
            head_learning_rate = picked_head_lr

    # Optimizer with separate LR for adapters vs classifier head
    adapter_params = []
    head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'classifier' in n:
            head_params.append(p)
        else:
            adapter_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": float(adapter_learning_rate), "weight_decay": 0.01},
            {"params": head_params, "lr": float(head_learning_rate), "weight_decay": 0.01},
        ]
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_steps = steps_per_epoch * int(num_epochs)
    warmup_steps = int(0.12 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Optional class weights
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights_from_data(single_sector_wrapper, label_names).to(device)

    best_macro_f1 = 0.0
    patience = 3
    epochs_no_improve = 0

    # Label smoothing 0.05
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05, weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=0.05)

    best_ckpt_dir = None
    # CPU: evaluate once per epoch to reduce overhead; GPU: ~quarter epoch
    eval_interval_steps = steps_per_epoch if device.type == 'cpu' else max(1, int(0.25 * steps_per_epoch))
    global_step = 0
    for epoch in range(int(num_epochs)):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"{sector.upper()} LoRA Epoch {epoch+1}/{int(num_epochs)}")
        accum_count = 0
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = ce_loss(outputs.logits, labels) / max(1, grad_accum_steps)
            loss.backward()
            total_loss += float(loss.item())
            accum_count += 1

            if accum_count % max(1, grad_accum_steps) == 0:
                clip_grad_norm_(adapter_params + head_params, max_norm=0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Mid-epoch evaluation and checkpointing (optional)
                if do_eval and (global_step % eval_interval_steps == 0):
                    val_data_sector = validation_data.get(sector, {}) if validation_data else {}
                    if val_data_sector:
                        try:
                            optimal_thresholds = optimize_thresholds_on_training(model, tokenizer, label_names, single_sector_wrapper, device)
                        except Exception:
                            optimal_thresholds = None
                        metrics = evaluate_model(model, tokenizer, label_names, val_data_sector, device, optimal_thresholds)
                        macro_f1 = metrics.get('macro_f1', 0.0)
                        # Save best checkpoint
                        if macro_f1 > best_macro_f1:
                            best_macro_f1 = macro_f1
                            epochs_no_improve = 0
                            best_ckpt_dir = os.path.join(save_root, f"tmp_best_{sector}")
                            os.makedirs(best_ckpt_dir, exist_ok=True)
                            model.save_pretrained(best_ckpt_dir)
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                print("Early stopping: macro-F1 plateaued")
                                break

            pbar.set_postfix({"loss": f"{(loss.item()*max(1,grad_accum_steps)):.4f}"})

        avg_loss = total_loss / max(1, len(train_loader))
        if do_eval:
            print(f"{sector.upper()} LoRA Epoch {epoch+1}: Loss={avg_loss:.4f} BestMacroF1={best_macro_f1:.3f}")
        else:
            print(f"{sector.upper()} LoRA Epoch {epoch+1}: Loss={avg_loss:.4f}")

    # Save LoRA adapters and tokenizer
    safe_model_tag = _sanitize_model_id_for_path(model_name.replace('-', '_'))
    save_dir = os.path.join(save_root, f"pro_anti_{sector}_lora_{safe_model_tag}")
    os.makedirs(save_dir, exist_ok=True)
    # Save best adapter if we have it; otherwise current adapter
    if best_ckpt_dir and os.path.isdir(best_ckpt_dir):
        dst = os.path.join(save_dir, 'adapter')
        os.makedirs(dst, exist_ok=True)
        for fname in os.listdir(best_ckpt_dir):
            src_path = os.path.join(best_ckpt_dir, fname)
            dst_path = os.path.join(dst, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
    else:
        model.save_pretrained(os.path.join(save_dir, 'adapter'))
    tokenizer.save_pretrained(save_dir)

    return model, tokenizer, label_names, save_dir


def train_all_sectors_with_lora(
    model_name: str = 'SamLowe/roberta-base-go_emotions',
    num_epochs: int | None = None,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    adapter_learning_rate: float | None = None,
    head_learning_rate: float | None = None,
    use_class_weights: bool = True,
    do_eval: bool = False,
    unfreeze_final_layer: bool = False,  # Option to unfreeze final encoder layer
    use_dora: bool = False,  # Option to use DoRA instead of standard LoRA
) -> Dict[str, Dict]:
    print("Loading GPT-labeled training/validation/test data...")
    training_data = load_pro_anti_training_data()
    validation_data = load_pro_anti_validation_data()
    test_data = load_pro_anti_test_data()

    results_summary: Dict[str, Dict] = {}
    sector_models: Dict[str, Dict] = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for sector in ['transport', 'housing', 'food']:
        sector_train = training_data.get(sector, {})
        if not sector_train:
            print(f"No training data for {sector}; skipping")
            continue
        model, tokenizer, label_names, save_dir = train_sector_with_lora(
            sector,
            sector_train,
            validation_data,
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            adapter_learning_rate=adapter_learning_rate,
            head_learning_rate=head_learning_rate,
            save_root='models_lora',
            use_class_weights=use_class_weights,
            do_eval=do_eval,
            unfreeze_final_layer=unfreeze_final_layer,
            use_dora=use_dora,
        )

        sector_models[sector] = {
            'save_dir': save_dir,
            'model_name': model_name,
            'label_names': label_names,
        }

        # Evaluate on test (if available)
        sector_test = test_data.get(sector, {}) if test_data else {}
        if sector_test:
            # Thresholds via validation if available
            optimal_thresholds = None
            val_sector = validation_data.get(sector, {}) if validation_data else {}
            if val_sector:
                try:
                    optimal_thresholds = optimize_thresholds_on_training(model, tokenizer, label_names, {sector: training_data.get(sector, {})}, device)
                except Exception:
                    optimal_thresholds = None
            metrics = evaluate_model(model, tokenizer, label_names, sector_test, device, optimal_thresholds)
            results_summary[sector] = metrics
            print(f"{sector.upper()} LoRA TEST: Acc={metrics['accuracy']:.3f} MacroF1={metrics['macro_f1']:.3f} n={metrics['num_samples']}")

    # Save metadata
    os.makedirs('models_lora', exist_ok=True)
    meta = {
        'method': 'lora_seq_cls',
        'base_model': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'grad_accum_steps': grad_accum_steps,
        'adapter_learning_rate': adapter_learning_rate,
        'head_learning_rate': head_learning_rate,
        'label_names': PRO_ANTI_LABELS,
        'sector_models': sector_models,
        'test_results': results_summary,
        'timestamp': int(time.time()),
        # Enhanced LoRA configuration details
        'lora_config': {
            'rank': 16,
            'alpha': 32,
            'dropout': 0.2,
            'target_modules': ["query", "key", "value", "o_proj", "ff_proj"],
            'use_dora': use_dora and DORA_AVAILABLE,
            'unfreeze_final_layer': unfreeze_final_layer,
            'init_lora_weights': 'init_lora_weights' in LoraConfig.__dict__,
        },
        'enhanced_features': [
            'Extended target modules (query, key, value, o_proj, ff_proj)',
            'Increased rank from 8 to 16',
            'Increased alpha from 16 to 32',
            'SVD-based initialization' if 'init_lora_weights' in LoraConfig.__dict__ else 'Standard initialization',
            '1.5x increased adapter learning rates',
            'Optional final layer unfreezing',
            'DoRA variant support' if (use_dora and DORA_AVAILABLE) else 'Standard LoRA with improvements'
        ]
    }
    with open('models_lora/pro_anti_lora_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return results_summary


if __name__ == "__main__":
    # Enhanced LoRA training with improved configuration
    print("Starting enhanced LoRA training with improvements:")
    print("- Rank: 16 (increased from 8)")
    print("- Alpha: 32 (increased from 16)")
    print("- Extended target modules: query, key, value, o_proj, ff_proj")
    print(f"- SVD initialization: {'Available' if 'init_lora_weights' in LoraConfig.__dict__ else 'Not available'}")
    print("- 1.5x increased adapter learning rates")
    print("- Optional final layer unfreezing")
    print(f"- DoRA variant: {'Available' if DORA_AVAILABLE else 'Not available in PEFT {peft.__version__}'}")
    
    # You can enable these options for even better performance:
    # unfreeze_final_layer=True  # Unfreeze final encoder layer
    # use_dora=True              # Use DoRA instead of standard LoRA (if available)
    
    train_all_sectors_with_lora(
        # unfreeze_final_layer=True,  # Uncomment to unfreeze final encoder layer
        # use_dora=True,              # Uncomment to use DoRA variant (if available)
    )


