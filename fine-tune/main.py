from utils import *
from trainer import *
from losses import *
import wandb
import argparse
from transformers import CLIPProcessor, ViltProcessor, BertTokenizerFast, GPT2TokenizerFast
from collections import Counter
import numpy as np
import random
import yaml

PATHS = {
    "hateful":   '../datasets/Hateful-Memes/',
    "harmful":   '../datasets/Harmful-Memes/',
    "misogyny":  '../datasets/Misogyny-Memes/',
}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5]},
        "weight_decay": {"values": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]},
        "batch_size": {"values": [16, 32, 64]},
        "num_epochs": {"values": [2, 3, 5, 7]},
        "patience":   {"values": [3, 5, 7]},
        "unfreeze_every": {"values": [1, 2, 3]},
        "loss_type":  {"values": ["cross_entropy", "focal"]},
        "focal_gamma":{"distribution": "uniform", "min": 1.0, "max": 3.0},
        "freeze": {"values": [True, False]},
        "dropout": {"values": [0.1, 0.01, 0.001]},
        "num_classes": {"values": [2]}
    },
}


def get_fold_loaders(data_type, model_type, fold_idx, fold_dir, processor, tokenizer, batch_size):
    fold_path = os.path.join(fold_dir, f"{data_type}.json")
    
    if not os.path.exists(fold_path):
        raise FileNotFoundError(
            f"Fold file not found: {fold_path}\n"
            f"Run: python generate_folds.py --data_type {data_type}"
        )

    with open(fold_path, "r") as f:
        fold_data = json.load(f)
 
    fold_key = f"fold_{fold_idx}"
    if fold_key not in fold_data:
        n = fold_data["meta"]["n_folds"]
        raise ValueError(f"fold_idx={fold_idx} not found in {fold_path} (n_folds={n})")
 
    all_records   = fold_data["records"]          # full ordered list of dicts
    train_indices = fold_data[fold_key]["train"]
    val_indices   = fold_data[fold_key]["val"]
 
    train_records = [all_records[i] for i in train_indices]
    val_records   = [all_records[i] for i in val_indices]
 
    print(f"\n[CV fold {fold_idx}] train={len(train_records)}, val={len(val_records)}")
 
    # ── Build datasets using your existing classes (nothing new here) ────────
    if model_type in ('bert', 'gpt2'):
        from utils import TextMemeDataset, TextCollator
        train_ds = TextMemeDataset(records=train_records, tokenizer=tokenizer)
        val_ds   = TextMemeDataset(records=val_records,   tokenizer=tokenizer)
        collator = TextCollator()
    elif model_type in ('vit'):
        from utils import BaseMemeDataset, MultimodalCollator
        train_ds = BaseMemeDataset(records=train_records, processor=processor)
        val_ds   = BaseMemeDataset(records=val_records,   processor=processor)
        collator = ImageCollator()

    else:
        from utils import BaseMemeDataset, MultimodalCollator
        train_ds = BaseMemeDataset(records=train_records, processor=processor)
        val_ds   = BaseMemeDataset(records=val_records,   processor=processor)
        collator = MultimodalCollator(processor)
 
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # shuffle within the fold's training split
        pin_memory=True,
        num_workers=2,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        collate_fn=collator,
    )
 
    return train_loader, val_loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_type')
    parser.add_argument('--model_type', default='vilt')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sweep_count', type=int, default=20)
    parser.add_argument('--reproduce', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--fold', default=0)
    parser.add_argument('--fold_dir', default='folds')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda_manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\nEVERYTHING SET TO SEED = {seed}\n")

def train_sweep():
    args = get_args()
    run = wandb.init()
    cfg = wandb.config

    processor  = None
    tokenizer  = None

    seed_everything(args.seed)

    if args.model_type == 'vilt':
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    elif args.model_type in ('clip', 'clip_encoder'):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif args.model_type == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.padding_side = "left"   # required for causal LM
        tokenizer.pad_token    = tokenizer.eos_token

    manager = MemeDatasetManager(
        paths=PATHS,
        dataset_type=args.data_type,
        model_type=args.model_type,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.batch_size,
    )

    train_loader, val_loader = get_fold_loaders(
            data_type=args.data_type,
            model_type=args.model_type,
            fold_idx=args.fold,
            fold_dir=args.fold_dir,
            processor=processor,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
        )

    test_loader = manager.get_split("test")

    mix_manager = MemeDatasetManager(
        paths=PATHS,
        dataset_type='mix',
        model_type=args.model_type,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.batch_size,
    )
    
    mix_test_loader = mix_manager._make_loader("test",  shuffle=False)

    imgur_imgs = "../datasets/MemeGate-Imgur/"
    memegate = pd.read_csv("../datasets/MemeGate-Imgur/imgur-data.csv")

    memegate_adult_dataset = MemeGateDataset(dataframe=memegate, image_base_path=imgur_imgs, label_type='adult', processor=processor, model_type=args.model_type, tokenizer=tokenizer,)
    memegate_youth_dataset = MemeGateDataset(dataframe=memegate, image_base_path=imgur_imgs, label_type='youth', processor=processor, model_type=args.model_type, tokenizer=tokenizer,)

    memegate_adult_loader = memegate_make_loader(data=memegate_adult_dataset, model_type=args.model_type, processor=processor, batch_size=cfg.batch_size, shuffle=False)
    memegate_youth_loader = memegate_make_loader(data=memegate_youth_dataset, model_type=args.model_type, processor=processor, batch_size=cfg.batch_size, shuffle=False)

    print("ALL DATA LOADED")
    print()

    from models import build_model
    model = build_model(args.model_type, num_classes=cfg.num_classes, dropout=cfg.dropout,)

    loss_config = {
        "loss_type":   cfg.loss_type,
        "focal_gamma": cfg.focal_gamma,
    }

    save_dir = f'{args.save_dir}-seed-{args.seed}-fold-{args.fold}'

    trainer = UniversalTrainer(
        model=model,
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        device=args.device,
        learning_rate=cfg.get("learning_rate"),
        weight_decay=cfg.get("weight_decay"),
        num_epochs=cfg.get("num_epochs"),
        patience=cfg.get("patience"),
        unfreeze_every=cfg.get("unfreeze_every"),
        loss_config=loss_config,
        freeze=cfg.get("freeze"),
        memegate_adult_loader=memegate_adult_loader,
        memegate_youth_loader=memegate_youth_loader,
        data_type=args.data_type,
        mix_test_loader=mix_test_loader,
        fold=args.fold
    )
    trainer.train()
    
    save_path = os.path.join(f"../spinning-storage/kverma/{save_dir}", args.model_type, f"{args.model_type}_best_train={args.data_type}_{wandb.run.name}")
    os.makedirs(save_path, exist_ok=True)

    model = trainer.model
    
    if args.reproduce:

        if args.model_type == 'vilt':
            model.vilt.save_pretrained(save_path)
            processor.save_pretrained(save_path)

        elif args.model_type == 'clip':
            model.clip.save_pretrained(save_path)
            processor.save_pretrained(save_path)

        elif args.model_type == 'clip_encoder':
            model.vision_model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

        elif args.model_type == 'bert':
            model.bert.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        elif args.model_type == 'gpt2':
            model.gpt2.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        else:
            print(f"{args.model_type} is INVALID !!")

        torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier_head.pt"))
    
        print(f"MODEL & CLASSIFICATION HEAD SAVED AT: {save_path}")

if __name__ == "__main__":
    
    args = get_args()

    if args.reproduce:
        with open(args.config, "r") as f:
            sweep_config = yaml.safe_load(f)
        
        sweep_id = wandb.sweep(
                sweep_config,
                project=f"meme-reproduce-{args.data_type}-{args.fold}-{args.model_type}-{args.fold}"
                )
        wandb.agent(sweep_id, function=train_sweep, count=1)

    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=f"meme-seed-{args.seed}-{args.data_type}-{args.fold}-{args.model_type}")
        wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)


