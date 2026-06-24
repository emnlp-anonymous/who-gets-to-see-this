This directory contains the code used to verify that MemeGate contains images that are visually distinct from existing meme corpora, following the non-parametric out-of-distribution (OOD) detection framework of Sun et al. (2022).


## Running the Analysis
### Step 1: Compute k-NN Distances
Run:
```
python knn_unseen.py --model siglip --compare {harmp, harmc, hate, misogyny} --k {1, 2, 3, 4, 5} --out_dir {output-directory}
```

Running the command above produces files such as: 
```
dist_harmp_siglip_k1.npy
dist_harmp_siglip_k1.json

The JSON file contains:

{
  "model": "siglip",
  "compare": "harmp",
  "k": 1,
  "n": 5000,
  "distances": [...]
}
```

### Step 2: 
