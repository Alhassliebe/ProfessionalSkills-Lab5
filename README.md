# Does Precision Matter? Quantising wav2vec 2.0 Word Embeddings

Lab for the Professional Skills course, Université Paris Cité, 2026.

Speech embeddings take a lot of space. Can we store them in lower precision without losing the information that matters? I extract word-level embeddings with `facebook/wav2vec2-base` from a corpus of 19 speakers (12 target words, 1,361 tokens), store them in float64, float32, float16 and int8, and check whether the distances between them still behave the same way. The test: the same word said by the same speaker should be closer than the same word said by different speakers.

## Results

| Precision | Intra-speaker distance | Inter-speaker distance | Ratio | Storage |
|---|---|---|---|---|
| float64 | 0.0487 | 0.1067 | 2.192 | 8.0 MB |
| float32 | 0.0487 | 0.1067 | 2.192 | 4.0 MB |
| float16 | 0.0487 | 0.1067 | 2.192 | 2.0 MB |
| int8 | 0.0488 | 0.1068 | 2.188 | 1.0 MB |

Going down to int8 cuts storage by 8 times and leaves the intra/inter speaker structure practically unchanged (ratio 2.19 vs 2.19). Mean cosine distances are shown in the table.

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