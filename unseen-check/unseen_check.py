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

def compute_knn_distances(query_emb, pool_emb, k):
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(pool_emb)
    dist, _ = nbrs.kneighbors(query_emb)
    return dist[:, -1]

def print_stats(dist, label):
    print(f"  {label}  n={len(dist)}")
    print(f"    mean:   {dist.mean():.4f}")
    print(f"    median: {np.median(dist):.4f}")
    print(f"    std:    {dist.std():.4f}")
    print(f"    min:    {dist.min():.4f}")
    print(f"    max:    {dist.max():.4f}")
    print(f"    p25:    {np.percentile(dist, 25):.4f}")
    print(f"    p75:    {np.percentile(dist, 75):.4f}")
    print(f"    p95:    {np.percentile(dist, 95):.4f}")

def save_distances(dist, name, out_dir, meta):
    np.save(os.path.join(out_dir, f"{name}.npy"), dist)
    out = {**meta, "n": len(dist), "distances": dist.tolist()}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {name}.npy / .json")

def get_or_extract(name, paths, out_dir, processor, model, reuse):
    emb_path = os.path.join(out_dir, f"emb_{name}.npy")
    if reuse and os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"Loaded cached embeddings: {name} {emb.shape}")
        return emb
    print(f"\n--- Extracting: {name} ({len(paths)} images) ---")
    emb = extract_embeddings(paths, processor, model)
    np.save(emb_path, emb)
    print(f"Saved embeddings -> emb_{name}.npy")
    return emb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=['dino', 'siglip'])
    parser.add_argument("--k",       required=True, type=int)
    parser.add_argument("--out_dir", default="./results")
    parser.add_argument("--reuse",   action="store_true",
                        help="Reuse cached .npy embeddings if they exist")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    siglip_name = "google/siglip-base-patch16-224"
    dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    imgur_path, hate_path, harmc_path, harmp_path, misogyny_path = get_img_paths()

    compare_map = {
        'hate':     hate_path,
        'harmc':    harmc_path,
        'harmp':    harmp_path,
        'misogyny': misogyny_path,
    }

    # Load model once
    model_name = dinov3_name if args.model == 'dino' else siglip_name
    processor, model = load_model(model_name, args.model)

    # Extract Imgur once — reused across all comparisons
    X_imgur = get_or_extract('imgur', imgur_path, args.out_dir, processor, model, args.reuse)

    # Extract all target datasets
    embeddings = {}
    for ds_name, ds_paths in compare_map.items():
        embeddings[ds_name] = get_or_extract(ds_name, ds_paths, args.out_dir, processor, model, args.reuse)

    # Run comparisons
    print(f"\n{'='*60}")
    print(f"  KNN Distance Analysis  [model={args.model.upper()}  k={args.k}]")
    print(f"{'='*60}")

    summary = {}
    for ds_name, Y_target in embeddings.items():

        print(f"\n--- Imgur vs {ds_name.upper()} ---")

        # Imgur distances to this dataset (main claim)
        dist_imgur = compute_knn_distances(X_imgur, Y_target, args.k)
        print_stats(dist_imgur, f"Imgur -> {ds_name.upper()}")
        save_distances(dist_imgur,
                       f"dist_{ds_name}_{args.model}_k{args.k}",
                       args.out_dir,
                       {"model": args.model, "compare": ds_name, "k": args.k})

        # Internal baseline: random split of target dataset vs itself
        n_sample = min(len(X_imgur), len(Y_target) // 2)
        idx = np.random.choice(len(Y_target), size=n_sample, replace=False)
        Y_sample = Y_target[idx]
        Y_rest   = np.delete(Y_target, idx, axis=0)
        dist_internal = compute_knn_distances(Y_sample, Y_rest, args.k)
        print_stats(dist_internal, f"{ds_name.upper()} internal baseline")
        save_distances(dist_internal,
                       f"dist_internal_{ds_name}_{args.model}_k{args.k}",
                       args.out_dir,
                       {"model": args.model, "compare": f"{ds_name}_internal", "k": args.k})

        delta = dist_imgur.mean() - dist_internal.mean()
        verdict = "OOD SUPPORTED" if delta > 0 else "OOD WEAKENED"
        print(f"\n  Delta (Imgur - internal): {delta:+.4f}  -> {verdict}")

        summary[ds_name] = {
            "imgur_mean":    round(dist_imgur.mean(), 4),
            "internal_mean": round(dist_internal.mean(), 4),
            "delta":         round(delta, 4),
            "verdict":       verdict,
        }

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY  [model={args.model.upper()}  k={args.k}]")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Imgur mean':>12} {'Internal mean':>14} {'Delta':>8}  Verdict")
    print(f"  {'-'*58}")
    for ds_name, s in summary.items():
        print(f"  {ds_name:<12} {s['imgur_mean']:>12.4f} {s['internal_mean']:>14.4f} "
              f"{s['delta']:>+8.4f}  {s['verdict']}")
