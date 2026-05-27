# Installation

1. Create Environment
```
conda create -n meme-training python=3.10
conda activate meme-training
```

2. Install Dependencies
```
pip install torch torchvision torchaudio
pip install transformers
pip install wandb
pip install scikit-learn pandas tqdm pyyaml
```

# Dataset

3. Dataset Structure
```
datasets/
│
├── Hateful-Memes/
├── Harmful-Memes/
├── Misogyny-Memes/
├── MemeGate-Imgur/
│   ├── imgur-data.csv
│   └── images
```
# Training

4. Training
```
python main.py \
    --seed <enter-seed> \
    --model_type <models = vilt, clip, clip_encoder, bert> \
    --data_type <data = hate, harm , misogyny, mix> \
    --save_dir <save-dir> \
    --sweep_count <count> \
    --fold <fold-num> \
    --fold_dir <fold-dir>
```

5. Arguments
```
Argument	Description
--seed	Random seed
--model_type	Model architecture (vilt, clip, bert, etc.)
--data_type	Dataset type (misogyny, hate, harm, mix)
--save_dir	Output directory
--sweep_count	Number of W&B sweep runs
--fold	Cross-validation fold index
--fold_dir	Directory containing fold JSON files
--device	Training device (cuda or cpu)
--reproduce	Enable reproduction mode
--config	YAML sweep config for reproduction
```
