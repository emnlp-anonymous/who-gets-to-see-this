import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, SiglipVisionModel
import argparse
import json

def get_extensions(folder):
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.JPEG')
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)

def get_img_paths():
    imgur_path    = get_extensions('../datasets/MemeGate-Imgur/blur_img_only/')
    hate_path     = get_extensions('../datasets/Hateful-Memes/img/')
    harmc_path    = get_extensions('../datasets/Harmful-Memes/Harm-C/images/')
    harmp_path    = get_extensions('../datasets/Harmful-Memes/Harm-P/images/')
    misogyny_path = get_extensions('../datasets/Misogyny-Memes/images/')
    return imgur_path, hate_path, harmc_path, harmp_path, misogyny_path

def load_model(modelname, model_type):
    if model_type == 'siglip':
        processor = AutoProcessor.from_pretrained(modelname)
        model = SiglipVisionModel.from_pretrained(modelname).eval().cuda()
    else:
        processor = AutoImageProcessor.from_pretrained(modelname)
        model = AutoModel.from_pretrained(modelname).eval().cuda()
    print(f"{modelname} loaded")
    return processor, model

def extract_embeddings(image_paths, processor, model, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch = []
        for p in image_paths[i:i+batch_size]:
            try:
                batch.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Skipping corrupt image {p}: {e}")
        if not batch:
            continue
        inputs = processor(images=batch, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
            feats = out.last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.vstack(embeddings)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=['dino', 'siglip'])
    parser.add_argument("--compare", required=True, choices=['hate', 'harmc', 'harmp', 'misogyny'])
    parser.add_argument("--k",       required=True, type=int)
    parser.add_argument("--out_dir", default="./results")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imgur_path, hate_path, harmc_path, harmp_path, misogyny_path = get_img_paths()

    siglip_name = "google/siglip-base-patch16-224"
    dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    if args.model == 'dino':
        processor, model = load_model(dinov3_name, 'dino')
    elif args.model == 'siglip':
        processor, model = load_model(siglip_name, 'siglip')

    compare_map = {
        'hate':     hate_path,
        'harmc':    harmc_path,
        'harmp':    harmp_path,
        'misogyny': misogyny_path,
    }

    target_paths = compare_map[args.compare]
    hate_paths = compare_map['hate']
    harmc_paths = compare_map['harmc']
    harmp_paths = compare_map['harmp']
    mis_paths = compare_map['misogyny']

    ref_paths    = imgur_path

    print(f"Found {len(ref_paths)} reference (Imgur) images.")
    print(f"Found {len(target_paths)} target ({args.compare}) images.")

    if len(ref_paths) == 0 or len(target_paths) == 0:
        raise ValueError("One of the directories is empty!")

    print("\n--- Processing Imgur (reference) ---")
    X_ref = extract_embeddings(ref_paths, processor, model)

    print("\n--- Processing Target ---")
    Y_target = extract_embeddings(target_paths, processor, model)
    
    nbrs = NearestNeighbors(n_neighbors=args.k, metric='cosine').fit(Y_target)
    
    dist_unseen, _ = nbrs.kneighbors(X_ref)
    dist_unseen = dist_unseen[:, -1]

    print(f"\n{'='*50}")
    print(f"  Distance Analysis: {args.compare.upper()} vs Imgur  [{args.model.upper()}]")
    print(f"{'='*50}")
    print(f"  n={len(dist_unseen)}")
    print(f"    mean:   {dist_unseen.mean():.4f}")
    print(f"    median: {np.median(dist_unseen):.4f}")
    print(f"    std:    {dist_unseen.std():.4f}")
    print(f"    min:    {dist_unseen.min():.4f}")
    print(f"    max:    {dist_unseen.max():.4f}")
    print(f"    p25:    {np.percentile(dist_unseen, 25):.4f}")
    print(f"    p75:    {np.percentile(dist_unseen, 75):.4f}")
    print(f"    p95:    {np.percentile(dist_unseen, 95):.4f}")

    # Save distances — named by arguments for easy identification
    save_name = f"dist_{args.compare}_{args.model}_k{args.k}"
    
    # .npy for fast reloading in numpy
    np.save(os.path.join(args.out_dir, f"{save_name}.npy"), dist_unseen)
    
    # .json for readability / plotting in other tools
    out = {
        "model":   args.model,
        "compare": args.compare,
        "k":       args.k,
        "n":       len(dist_unseen),
        "distances": dist_unseen.tolist()
    }
    with open(os.path.join(args.out_dir, f"{save_name}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {save_name}.npy / .json")
