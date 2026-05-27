import json
import logging
import os
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)

def _forward(model, model_type: str, batch: dict):
    """Mirrors UniversalTrainer._forward without labels (test has none sometimes)."""
    if model_type in ("vilt", "clip"):
        return model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )
    elif model_type in ("bert", "gpt2"):
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )
    elif model_type in ("clip_encoder"):
        return model(
            pixel_values=batch["pixel_values"],
            labels=batch.get("labels"),
        )

    elif model_type == 'vit':
        if 'pixel_values' not in batch.keys():
            batch['pixel_values'] = batch['image']
    
        return model(
                batch['pixel_values'],
                batch['labels']
                )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

def _compute_binary_metrics(labels: list, preds: list, probs: list):
    prob_pos = [p[1] for p in probs]   # probability of class 1
 
    f1_per_class = f1_score(labels, preds, average=None, zero_division=np.nan).tolist()
    macro_f1     = f1_score(labels, preds, average="macro",    zero_division=np.nan)
    weighted_f1  = f1_score(labels, preds, average="weighted", zero_division=np.nan)
 
    try:
        auc = roc_auc_score(labels, prob_pos)
    except ValueError:
        auc = None   # only one class present in labels
 
    report = classification_report(labels, preds, output_dict=True, zero_division=np.nan)
 
    return {
        "f1_class_0":  round(f1_per_class[0], 6) if len(f1_per_class) > 0 else None,
        "f1_class_1":  round(f1_per_class[1], 6) if len(f1_per_class) > 1 else None,
        "f1_macro":    round(macro_f1, 6),
        "f1_weighted": round(weighted_f1, 6),
        "auc_roc":     round(auc, 6) if auc is not None else None,
        "classification_report": report,
    }

def evaluate_test_set(
    model: torch.nn.Module,
    model_type: str,
    test_loader,
    device: str,
    save_dir: str,
    run_name: str,
    test_type: str,
    filename_key: str,
    data_type: str):

    model.eval()
    all_probs, all_preds, all_labels, all_filenames = [], [], [], []
    metrics_probs, metrics_preds, metrics_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            if 'labels' not in batch.keys() and 'label' in batch.keys():
                batch['labels'] = batch['label']

            filenames = batch.get(filename_key, [""] * len(batch["labels"]))
            labels = batch["labels"].to(device)

            tensor_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != filename_key
            }
            out = _forward(model, model_type, tensor_batch)
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(tensor_batch["labels"].cpu().tolist())
            all_filenames.extend(
                filenames if not isinstance(filenames, torch.Tensor)
                else filenames.cpu().tolist()
            )

            valid_mask = (labels != -1)
            if valid_mask.any():
                metrics_probs.extend(probs[valid_mask].cpu().tolist())
                metrics_preds.extend(preds[valid_mask].cpu().tolist())
                metrics_labels.extend(labels[valid_mask].cpu().tolist())

    if len(metrics_labels) > 0:
        metrics = _compute_binary_metrics(metrics_labels, metrics_preds, metrics_probs)
        logger.info(f"Test metrics (computed on {len(metrics_labels)} samples): %s", metrics)
    else:
        metrics = {}
        logger.warning("No valid labels (not -1) found for metric computation.")


    #metrics = _compute_binary_metrics(all_labels, all_preds, all_probs)
    #logger.info("Test metrics: %s", metrics)

    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for fname, prob, pred, label in zip(all_filenames, all_probs, all_preds, all_labels):
        rows.append({
            "filename":       fname,
            "prob_class_0":   round(prob[0], 6),
            "prob_class_1":   round(prob[1], 6),
            "predicted_label": int(pred),
            "original_label":  int(label),
        })

    out_path = os.path.join(save_dir, f"{run_name}_{data_type}_{model_type}_{test_type}_predictions.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Test predictions saved → %s", out_path)

    return metrics

