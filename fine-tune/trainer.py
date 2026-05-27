import os
import json
import logging
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm
import wandb
from losses import *
from evaluate import *

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Stop training when *both* val_loss and val_f1 stop improving."""

    def __init__(self, patience: int, min_delta: float,
                 restore_best_weights: bool, loss_config: dict, test_loader):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.counter = 0
        self.best_weights = None
        self.test_loader = test_loader
        self.loss_fn = get_loss_fn(loss_config)

    def __call__(self, val_loss: float, val_f1: float, model: torch.nn.Module) -> bool:
        loss_improved = val_loss < (self.best_loss - self.min_delta)
        f1_improved   = val_f1   > (self.best_f1   + self.min_delta)

        if loss_improved or f1_improved:
            if loss_improved:
                self.best_loss = val_loss
            if f1_improved:
                self.best_f1 = val_f1
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(
                    f"Early stopping. Restored best weights "
                    f"(loss={self.best_loss:.4f}, f1={self.best_f1:.4f})"
                )
            return True
        return False

def build_layerwise_optimizer(model: torch.nn.Module, base_lr: float, head_lr: float, weight_decay: float):
    """
    Three-group LR scheme:
      embedding layers  → base_lr × 0.1
      early encoder     → base_lr × 0.5
      late encoder      → base_lr
      classifier head   → head_lr  (no weight decay)
    """
    embedding_params, early_params, late_params, head_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if 'embed' in lname:
            embedding_params.append(param)
        elif 'classifier' in lname or 'projection' in lname or 'pooler' in lname:
            head_params.append(param)
        elif 'layer' in lname or 'layers' in lname:
            try:
                # Extract layer index from the name
                for part in name.split('.'):
                    idx = int(part)
                    if idx < 6:
                        early_params.append(param)
                    else:
                        late_params.append(param)
                    break
            except (ValueError, StopIteration):
                late_params.append(param)
        else:
            late_params.append(param)

    groups = []
    if embedding_params:
        groups.append({'params': embedding_params, 'lr': base_lr * 0.1,
                       'weight_decay': weight_decay})
    if early_params:
        groups.append({'params': early_params, 'lr': base_lr * 0.5,
                       'weight_decay': weight_decay})
    if late_params:
        groups.append({'params': late_params, 'lr': base_lr,
                       'weight_decay': weight_decay})
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': 0.0})

    return AdamW(groups)


class UniversalTrainer:
    """
    Forward-pass routing is based on model_type so that each model receives only the inputs it expects.
    """
    def __init__(self, model: torch.nn.Module, model_type: str, train_loader, val_loader, save_dir: str, device: str,  learning_rate: float, weight_decay: float, num_epochs: int, patience: int, freeze: bool, unfreeze_every: int, test_loader, loss_config, memegate_adult_loader, memegate_youth_loader, mix_test_loader, data_type:str, fold:int):
        self.model = model.to(device)
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = os.path.join(save_dir, model_type)
        self.unfreeze_every = unfreeze_every
        self.test_loader = test_loader
        self.loss_fn = get_loss_fn(loss_config)
        self.freeze = freeze
        self.adult_loader = memegate_adult_loader
        self.youth_loader = memegate_youth_loader
        self.data_type = data_type
        self.mix_test_loader = mix_test_loader
        self.fold=fold

        self.optimizer = build_layerwise_optimizer(
            model,
            base_lr=learning_rate,
            head_lr=learning_rate * 10,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = total_steps // 10
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=0.001, restore_best_weights=True, loss_config=loss_config, test_loader=test_loader
        )

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0


    def _forward(self, batch: dict) -> dict:
        if self.model_type == 'vilt':
            return self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
            )

        elif self.model_type == 'clip':
            return self.model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
            )

        elif self.model_type == 'bert':
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
            )

        elif self.model_type == 'gpt2':
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
            )

        elif self.model_type == 'clip_encoder':
            return self.model(
                pixel_values=batch['pixel_values'],
                labels=batch.get('labels'),
            )

        elif self.model_type == 'vit':
            if 'pixel_values' not in batch.keys():
                batch['pixel_values'] = batch['image']

            return self.model(
                    batch['pixel_values'],
                    batch.get('labels')
                    )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")



    def _train_epoch(self, epoch: int, unfreeze_every: int):
        self.model.train()
        if not self.freeze:
            stage = min(epoch // unfreeze_every, 4)
            if hasattr(self.model, 'unfreeze_stage'):
                self.model.unfreeze_stage(stage)

        total_loss, correct, total = 0.0, 0, 0

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Train epoch {epoch+1}")):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != 'filename'
            }

            if self.model_type == 'vit':
                batch['labels'] = batch['label']

            self.optimizer.zero_grad()
            out = self._forward(batch)
            logits = out['logits']
            loss = self.loss_fn(logits, batch['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total += batch['labels'].size(0)
            correct += (preds == batch['labels']).sum().item()

            if step % 40 == 0:
                wandb.log({'train_batch_loss': loss.item()})

        return total_loss / len(self.train_loader), correct / total

    def _validate(self, run_name: str = None):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs, all_filenames = 0.0, [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if self.model_type == 'vit':
                    batch['labels'] = batch['label']

                filenames = batch.get('filename', [''] * len(batch['labels']))
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k != 'filename'
                }

                out = self._forward(batch)
                logits = out['logits']
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=1)

                total_loss += self.loss_fn(logits, batch['labels']).item()
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
                all_filenames.extend(
                    filenames if not isinstance(filenames, torch.Tensor)
                    else filenames.cpu().tolist()
                )

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average='macro')
        prec, rec, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1, prec, rec, all_filenames, all_preds, all_probs, all_labels


    def train(self):
        logger.info(f"Starting training: model={self.model_type}, epochs={self.num_epochs}")
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(epoch, self.unfreeze_every)
            val_loss, val_acc, val_f1, val_prec, val_rec, val_fnames, val_preds, val_probs, val_labels = self._validate()

            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'learning_rate': self.optimizer.param_groups[-1]['lr'],
            }
            wandb.log(metrics)

            logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_loss
                


            if self.early_stopping(val_loss, val_f1, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                last_val_fnames, last_val_preds, last_val_probs, last_val_labels = val_fnames, val_preds, val_probs, val_labels
                break

            else:
                last_val_fnames, last_val_preds, last_val_probs, last_val_labels = val_fnames, val_preds, val_probs, val_labels

        #run_name = wandb.run.name if wandb.run else self.model_type
        run_name = wandb.run.name if wandb.run else self.model_type
        if self.fold is not None:
            run_name = f"fold{self.fold}_{run_name}"

        os.makedirs(self.save_dir, exist_ok=True)

        val_rows = []
        for fname, pred, prob, label in zip(last_val_fnames, last_val_preds, last_val_probs, last_val_labels):
            val_rows.append({
            "img_path": fname,
            "predicted_label": int(pred),
            "prob_class_0": round(prob[0], 6),
            "prob_class_1": round(prob[1], 6),
            "original_label": int(label),
        })

        pd.DataFrame(val_rows).to_csv(
                os.path.join(self.save_dir, f"{run_name}_{self.data_type}_{self.model_type}_VAL_predictions.csv"), index=False
        )
        logger.info(f"Val predictions saved → {run_name}_{self.data_type}_{self.model_type}_val_predictions.csv")


        logger.info("TESTING")
        self.test_metrics = evaluate_test_set(
                model=self.model,
                model_type=self.model_type,
                test_loader=self.test_loader,
                device=self.device,
                save_dir=self.save_dir,
                run_name=run_name,
                test_type = "test-set",
                filename_key='filename',
                data_type=self.data_type,
        )
        wandb.log({f"test_{k}": v for k, v in self.test_metrics.items()
                   if isinstance(v, (int, float))})

        logger.info("MEMEGATE-ADULT")

        self.adult_metrics = evaluate_test_set(
                model=self.model,
                model_type=self.model_type,
                test_loader = self.adult_loader,
                device=self.device,
                save_dir=self.save_dir,
                run_name=run_name,
                filename_key='filename',
                test_type = "adult",
                data_type=self.data_type
                )

        wandb.log({f"adult_test_{k}": v for k, v in self.adult_metrics.items() if isinstance(v, (int, float))})
        
        logger.info("MEMEGATE-YOUTH")

        self.youth_metrics = evaluate_test_set(
                model=self.model,
                model_type=self.model_type,
                test_loader=self.youth_loader,
                device=self.device,
                save_dir=self.save_dir,
                run_name=run_name,
                filename_key='filename',
                test_type = "youth",
                data_type=self.data_type
                )
        
        wandb.log({f"youth_test_{k}": v for k,v in self.youth_metrics.items() if isinstance(v, (int, float))})
        logger.info("MIX-DATASET-TEST-SET")

        self.mix_metrics = evaluate_test_set(
            model=self.model,
            model_type=self.model_type,
            test_loader=self.mix_test_loader,
            device=self.device,
            save_dir=self.save_dir,
            run_name=run_name,
            filename_key='filename',
            test_type='mix',
            data_type=self.data_type
        )

        wandb.log({f"mix_test_{k}": v for k,v in self.mix_metrics.items() if isinstance(v, (int, float))})

       
