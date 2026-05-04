### Pipeline

Two stages, split between Kaggle (run on GPU) and a local DVC pipeline (run on CPU)

**Kaggle notebook** (`kaggle_extract_embeddings.ipynb`):
1. Read `RUFRcorr.csv` (12 target words)
2. For every speaker (19 in total), concatenate per-recording `_words.csv` files into one sequential word table
3. Find every occurrence of each target by matching token sequences
4. Run `facebook/wav2vec2-base` once per wav, slice the last hidden state by word boundaries, mean-pool over time
5. Save `embeddings.npz` in float64 (1361 vectors)

**Local DVC pipeline** (`dvc.yaml`):
- `analyse` (`src/analyse.py`) — converts the float64 reference to float32, float16 and int8, computes pairwise cosine distance matrices, splits pairs into intra-speaker and inter-speaker groups (same word in both cases), measures storage and compute time. Outputs `summary.csv`, `metrics.json`, `distances.npz`
- `visualise` (`src/visualise.py`) — produces `distributions.png` and `comparison.png` from the cached distances

Reproducible end-to-end: `dvc repro` re-runs only stages whose dependencies changed