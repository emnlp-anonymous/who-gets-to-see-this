import re
import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import ast 
import logging
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm
from collections import Counter
logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class MultimodalCollator:
    """Collates multimodal (image + text) batches using a HuggingFace processor."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        if "pixel_values" in batch[0]:
            filenames = [b.pop("filename", "") for b in batch]
            labels    = torch.stack([b.pop("label") for b in batch])
            collated  = {
                k: torch.stack([b[k] for b in batch])
                for k in batch[0]
            }
            collated["labels"]   = labels
            collated["filename"] = filenames
            return collated
        else:
            images, texts, labels = zip(*[(b["image"], b["text"], b["label"]) for b in batch])
            inputs = self.processor(
                text=list(texts),
                images=list(images),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs["labels"] = torch.tensor(labels, dtype=torch.long)
            return inputs

class BaseMemeDataset(Dataset):
    """
    A unified meme dataset class that handles all three dataset formats:
      - Hateful Memes   (JSONL: {img, text, label})
      - Harmful Memes   (JSONL: {image, text, label_bin})
      - Misogyny Memes  (TSV:   {file_name, text, label})

    Args:
        records   : list of dicts, each with keys `img_path`, `text`, `label`
        transform : torchvision transform (used when no processor is given)
        processor : HuggingFace processor (CLIP, ViLT, …); takes priority over transform
    """

    def __init__(self, records: list[dict], transform=None, processor=None):
        self.records   = records
        self.processor = processor
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        #print(rec["img_path"])
        image = Image.open(rec["img_path"]).convert("RGB")
        text  = str(rec.get("text", ""))
        label = int(rec["label"])

        if self.processor:
            image = image.resize((224, 224))
            inputs = self.processor(
                text=text, images=image,
                return_tensors="pt", padding="max_length", truncation=True,
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["label"] = torch.tensor(label, dtype=torch.long)
            inputs["filename"] = rec["img_path"]
            return inputs

        return {
            "image": self.transform(image),
            "text":  text,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": rec["img_path"],
        }


def _records_from_jsonl(jsonl_path: str, img_dir: str,img_key="img", label_key="label", harmful=None) -> list[dict]:
    """Parse a JSONL file into a list of unified record dicts."""
    records = []
    labels = []
    with open(jsonl_path, 'r') as f:
        content = list(f)
        for line in content:
            if not line.strip():
                continue

            line = re.sub(r"\\/", '/', line)

            entry = json.loads(line)
            if isinstance(entry, str):
                entry = ast.literal_eval(entry)

            if harmful:
                labels += entry['labels']
                if entry['labels'][0] == 'not harmful':
                    entry[label_key] = 0
                else:
                    entry[label_key] = 1

            #print(entry)

            records.append({
                "img_path": os.path.join(img_dir, entry[img_key]),
                "text":     str(entry.get("text", "")),
                "label":    int(entry[label_key]),
            })


    labels2 = [rec['label'] for rec in records]
    print(f"\nDURING LOADING TIME for {jsonl_path}")
    print(f"LABEL DISTRO: {Counter(labels2)}")

    return records

def _records_from_tsv(tsv_path: str, img_dir: str, img_col="file_name", label_col="label") -> list[dict]:
    """Parse a TSV file into a list of unified record dicts."""
    df = pd.read_csv(tsv_path, sep="\t")
    return [
        {
            "img_path": os.path.join(img_dir, row[img_col]),
            "text":     str(row.get("text", "")),
            "label":    int(row[label_col]),
        }
        for _, row in df.iterrows()
    ]

def _load_hateful(base_path: str, split: str) -> list[dict]:
    """Facebook Hateful Memes — JSONL with keys: img, text, label."""
    fname = {"train": "train_new", "val": "val_new", "test": "test_new"}[split]
    return _records_from_jsonl(
        jsonl_path=os.path.join(base_path, f"{fname}.jsonl"),
        img_dir=os.path.join(base_path, ""),
    )


def _load_harmful(base_path: str, category: str, split: str) -> list[dict]:
    """Harm-P / Harm-C — JSONL with keys: image, text, label_bin."""
    split_file = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}[split]
    return _records_from_jsonl(
        jsonl_path=os.path.join(base_path, category, "annotations", split_file),
        img_dir=os.path.join(base_path, category, "images"),
        img_key="image",
        label_key="label_bin",
        harmful=True
    )


def _load_misogyny(base_path: str, split: str) -> list[dict]:
    """SemEval Misogyny Memes — TSV with keys: file_name, text, label."""
    fname = {"train": "train", "val": "validation", "test": "test"}[split]
    return _records_from_tsv(
        tsv_path=os.path.join(base_path, f"{fname}.tsv"),
        img_dir=os.path.join(base_path, "images"),
    )


class MemeDatasetManager:
    """
    Builds train / val / test DataLoaders by merging all three meme datasets.

    Args:
        paths : dict with keys:
                  "hateful"   → root dir of Hateful Memes dataset
                  "harmful"   → root dir of Harmful Memes dataset (Harm-P & Harm-C inside)
                  "misogyny"  → root dir of Misogyny dataset
        processor  : HuggingFace processor (optional; used for transformer-based models)
        transform  : torchvision transform (optional; used when no processor is given)
        batch_size : DataLoader batch size
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, paths: dict, dataset_type: str, model_type: str, batch_size: int, processor=None, tokenizer=None, transform=None):
        self.paths      = paths
        self.dataset_type = dataset_type
        self.processor  = processor
        self.tokenizer    = tokenizer
        self.transform  = transform
        self.batch_size = batch_size
        self.model_type = model_type

        if model_type in ('bert', 'gpt2'):
            self.collator = TextCollator()
        else:
            self.collator   = MultimodalCollator(processor) if processor else None

    def _build_hateful_records(self, split: str) -> list[dict]:
        return _load_hateful(self.paths["hateful"], split)

    def _build_harmful_records(self, split: str) -> list[dict]:
        records = []
        records += _load_harmful(self.paths["harmful"], "Harm-P", split)
        records += _load_harmful(self.paths["harmful"], "Harm-C", split)
        return records

    def _build_misogyny_records(self, split: str) -> list[dict]:
        return _load_misogyny(self.paths["misogyny"], split)

    def _built_all_records(self, split: str) -> list[dict]:
        records = []
        records += _load_hateful(self.paths["hateful"], split)
        records += _load_harmful(self.paths["harmful"], "Harm-P", split)
        records += _load_harmful(self.paths["harmful"], "Harm-C", split)
        records += _load_misogyny(self.paths["misogyny"], split)
        return records

    def _get_records(self, split: str):
        if self.dataset_type == 'hate':
            return self._build_hateful_records(split)
        elif self.dataset_type == 'harm':
            return self._build_harmful_records(split)
        elif self.dataset_type == 'misogyny':
            return self._build_misogyny_records(split)
        else:
            return self._built_all_records(split)


    def _make_dataset(self, split: str) -> BaseMemeDataset:
        records = self._get_records(split)

        if self.model_type in ('bert', 'gpt2'):
            return TextMemeDataset(
                records=records,
                tokenizer=self.tokenizer,
            )
        else:
            return BaseMemeDataset(
                records=records,
                transform=self.transform,
                processor=self.processor,
            )



    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._make_dataset(split),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=2,
            collate_fn=self.collator,
        )

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_loader, val_loader, test_loader)."""
        return (
            self._make_loader("train", shuffle=True),
            self._make_loader("val",   shuffle=False),
            self._make_loader("test",  shuffle=False),
        )

    def get_split(self, split: str) -> DataLoader:
        """Return a single DataLoader for the given split ('train'/'val'/'test')."""
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}")
        return self._make_loader(split, shuffle=(split == "train"))


class MemeGateDataset(Dataset):
    def __init__(self, dataframe, image_base_path, label_type, model_type,
                 processor=None, tokenizer=None, transform=None):
        self.model_type      = model_type
        self.image_base_path = image_base_path
        self.transform       = transform if transform is not None else DEFAULT_TRANSFORM
        self.label_type      = label_type
        self.processor       = processor
        self.tokenizer       = tokenizer
        self.label_map       = {'yes': 1, 'no': 0, 'contested': -1}

        # Filter: remove rows with blank/null text for all model types
        df = dataframe.copy()
        df = df.fillna({'text':'empty'})
        df = df[df['text'].astype(str).str.strip() != '']
        self.dataframe = df.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row['text']).strip()

        if self.label_type == 'adult':
            label_value = row['limit_adult_maj']
        elif self.label_type == 'youth':
            label_value = row['limit_youth_maj']
        else:
            raise ValueError("label_type must be 'adult' or 'youth'")

        label = torch.tensor(self.label_map.get(label_value, -1), dtype=torch.long)
        img_path = os.path.join(self.image_base_path, row['img_path'])

        if self.model_type in ('bert', 'gpt2'):
            encoded = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids":      encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label":          label,
                "filename":       row["img_path"],
            }

        if self.model_type in ('vit', 'clip_encoder'):
            image = Image.open(img_path).convert('RGB')
            return {
                "pixel_values": self.transform(image),
                "label":        label,
                "filename":     row["img_path"],
            }

        
        image = Image.open(img_path).convert('RGB').resize((224, 224))
        inputs = self.processor(
            text=text, images=image,
            return_tensors="pt", padding="max_length", truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["label"]    = label
        inputs["filename"] = row["img_path"]
        return inputs

def memegate_make_loader(data, model_type, processor, batch_size, shuffle: bool) -> DataLoader:
    if model_type in ('bert', 'gpt2'):
        collator = TextCollator()
    elif model_type in ('vit', 'clip_encoder'):
        collator = TextCollator()
    else:
        collator = MultimodalCollator(processor)

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
        collate_fn=collator,
    )


class TextMemeDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = str(rec.get("text", ""))
        label = int(rec["label"])

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
            "filename":       rec["img_path"],  # kept for consistency in CSV saving
        }

class TextCollator:
    def __call__(self, batch):
        filenames = [b.pop("filename", "") for b in batch]
        labels    = torch.stack([b.pop("label") for b in batch])
        collated  = {
            k: torch.stack([b[k] for b in batch])
            for k in batch[0]
        }
        collated["labels"]   = labels
        collated["filename"] = filenames
        return collated

class ImageCollator:
    def __call__(self, batch):
        filenames = [b.pop("filename", "") for b in batch]
        labels    = torch.stack([b.pop("label") for b in batch])
        images    = torch.stack([b["image"] for b in batch])
        return {
            "pixel_values": images,
            "label":       labels,
            "filename":     filenames,
        }
