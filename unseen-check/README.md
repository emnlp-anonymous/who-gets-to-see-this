This directory contains the code used to verify that MemeGate contains images that are visually distinct from existing meme corpora, following the non-parametric out-of-distribution (OOD) detection framework of Sun et al. (2022).
This directory is discussed in Section 5.1 and Appendix E


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

### Step 2: Analyse the k-NN Distances

Run:
```
for k in 1 2 3 4 5; do
    python unseen_check.py --model siglip --k $k --out_dir results --reuse
```

### Note:
`unseen_check.py` also computes the **OOD signal (Δ)**:

$$
\Delta = \mathbb{E}[d_{\text{Imgur} \rightarrow \text{corpus}}] - \mathbb{E}[d_{\text{internal}}]
$$

where Δ > 0 indicates OOD support, and Δ ≤ 0 indicates weak or no OOD evidence.
