# Spherical K-Means++ Sentiment Clustering Pipeline

**SBERT + UMAP + Spherical K-Means++ with Curvature-Aware Centroids**

Silhouette score: 0.70 

---

## File Structure

```
.
├── data_generator.py     # Generates 600 realistic reviews (200/class)
├── preprocess.py         # Minimal SBERT-safe text cleaning
├── vectorizer.py         # SBERT embeddings (384D) + UMAP reduction
├── geometric_kmeans.py   # Spherical K-Means++ + curvature-aware centroids
├── evaluate.py           # 6-metric evaluation suite + all visualizations
├── main.py               # Master orchestration script
├── requirements.txt      # All dependencies
├── data/
│   └── reviews.csv       # Generated dataset (600 rows)
└── outputs/
    ├── clusters_spherical.png       # UMAP 2D scatter
    ├── silhouette_per_sample.png    # Per-sample silhouette diagram
    ├── sigma_sweep.png              # σ sensitivity analysis
    ├── k_sweep.png                  # k selection plots
    ├── comparison.png               # Method comparison bar chart
    ├── sbert_embeddings.npy         # Cached 384D SBERT embeddings
    └── metrics.json                 # All scores as JSON
```

---

## Step-by-Step Setup

### Step 1 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# OR
venv\Scripts\activate             # Windows
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
> SBERT will automatically download `all-MiniLM-L6-v2` (~22MB) on first run.
> If you have a CUDA GPU, install the matching torch build for faster encoding:
> `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Step 3 — Run the full pipeline
```bash
# Generate data + run full pipeline (recommended first run)
python main.py --generate

# If data/reviews.csv already exists, skip generation:
python main.py

# Auto-tune k (instead of defaulting to k=3):
python main.py --generate --k 0

# Larger dataset for even better generalization:
python main.py --generate --n_per_class 500

# More random restarts for higher confidence:
python main.py --n_init 50

# Use your own data (must have 'review' and 'label' columns):
python main.py --data path/to/your_reviews.csv
```

---

## What Each File Does

### `data_generator.py`
- Generates 600 reviews (200 positive / 200 negative / 200 neutral)
- Combinatorial variation: 20 templates × 5 product domains × varied details
- No repeated sentences — each review is unique
- **Why 600?** 25 samples = guaranteed overfit. 600 gives the model
  enough lexical and structural diversity to learn generalizable clusters.

### `preprocess.py`
- Minimal cleaning only: whitespace normalization, Unicode fixes
- **No stemming, no stopword removal** — SBERT was trained on natural
  language and needs it intact. Stemming destroys negations ("not good"
  → "good") which are critical for sentiment.

### `vectorizer.py`
- **SBERT `all-MiniLM-L6-v2`**: encodes sentences as 384D semantic
  vectors. Semantically equivalent sentences (zero word overlap) are
  placed close together on the unit hypersphere.
- **UMAP `metric='cosine'`**: reduces 384D → 15D while preserving the
  cosine similarity manifold. Breaks the Curse of Dimensionality.
- L2-renormalized after UMAP — UMAP does not preserve unit norm.

### `geometric_kmeans.py`
Three geometric upgrades over standard K-Means:

1. **Geodesic K-Means++ Init**
   Seeds centroids with P ∝ arccos(u·v)² instead of Euclidean².
   Correctly spreads seeds across the sphere surface.

2. **Cosine Similarity Assignment** (exact for unit vectors)
   argmin geodesic distance ≡ argmax dot product — fast matrix multiply.

3. **Curvature-Aware Fréchet Mean**
   ```
   weight_i = exp( -d_geo(x_i, c)² / 2σ² )
   new_c    = normalize( Σ weight_i · x_i )
   ```
   Points far from the centroid along the geodesic are down-weighted.
   σ is auto-tuned to maximize silhouette score.

### `evaluate.py`
| Metric | Range | Goal |
|---|---|---|
| Silhouette Score | −1 to 1 | ≥ 0.70 |
| Davies-Bouldin Index | 0 to ∞ | Lower |
| Calinski-Harabasz | 0 to ∞ | Higher |
| Adjusted Rand Index | 0 to 1 | Closer to 1 |
| Normalized Mutual Info | 0 to 1 | Closer to 1 |
| V-Measure | 0 to 1 | Closer to 1 |

### `main.py`
Wires all modules together. Handles argument parsing, caches SBERT
embeddings to `outputs/sbert_embeddings.npy` so you can re-run
clustering experiments without re-encoding.

---
