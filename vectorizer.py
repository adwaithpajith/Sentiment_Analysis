"""
vectorizer.py
-------------
Stage 1: SBERT Sentence Embeddings
Stage 2: UMAP Manifold Reduction (cosine metric)

WHY SBERT OVER TF-IDF:
    TF-IDF encodes vocabulary overlap. Two reviews:
        "Absolutely terrible, complete garbage."
        "Utterly horrible, total junk."
    → TF-IDF cosine similarity ≈ 0.0   (zero word overlap)
    → SBERT cosine similarity ≈ 0.89   (same semantic meaning)

    Sentiment clustering requires semantic proximity, not lexical overlap.
    SBERT maps semantically equivalent texts to nearby points on the
    unit hypersphere — the correct geometry for spherical K-Means.

WHY UMAP BEFORE CLUSTERING:
    SBERT produces 384-dimensional embeddings. In 384D, the Curse of
    Dimensionality causes all pairwise distances to converge toward
    the same value — clusters become invisible. UMAP:
      1. Preserves local cosine similarity structure (metric='cosine')
      2. Compresses to a dense, cluster-friendly manifold
      3. Dramatically improves silhouette geometry without losing topology
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap


# ─────────────────────────────────────────────────────────────────────────────
# SBERT EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def get_sbert_embeddings(
    texts: list[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64,
    normalize_output: bool = True
) -> np.ndarray:
    """
    Encode sentences using SBERT.

    Model choice — 'all-MiniLM-L6-v2':
      - 384D output (compact but highly expressive)
      - Trained on 1B+ sentence pairs via contrastive learning
      - Optimized for semantic textual similarity
      - ~22M parameters — fast inference even on CPU
      - State-of-the-art performance on STS benchmarks

    Args:
        texts           : list of raw (minimally cleaned) reviews
        model_name      : HuggingFace model identifier
        batch_size      : sentences encoded per forward pass
        normalize_output: L2-normalize → projects onto unit hypersphere
                          Required for cosine similarity = dot product
                          Required for valid spherical K-Means operation

    Returns:
        np.ndarray of shape (N, 384), float32, L2-normalized if requested
    """
    print(f"── Loading SBERT model: {model_name} ──")
    model = SentenceTransformer(model_name)

    print(f"── Encoding {len(texts)} sentences (batch_size={batch_size}) ──")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_output   # SBERT-level L2 normalization
    )
    print(f"✅ SBERT embeddings: {embeddings.shape}  |  dtype: {embeddings.dtype}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# UMAP REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_umap_embedding(
    X: np.ndarray,
    n_components: int = 15,
    n_neighbors: int = 15,
    min_dist: float = 0.05,
    spread: float = 1.0,
    random_state: int = 42
) -> np.ndarray:
    """
    UMAP dimensionality reduction preserving cosine similarity manifold.

    Parameter guidance:
        n_components: Target dimensions for clustering input.
                      15–30 is ideal — low enough to escape the curse of
                      dimensionality, high enough to preserve cluster structure.
                      (2D is ONLY for visualization — never cluster in 2D)

        n_neighbors:  Controls local vs global structure preservation.
                      Rule of thumb: sqrt(N) for small datasets, 15–50 for large.
                      Too small → noisy, disconnected manifold.
                      Too large → over-smoothed, clusters merge.

        min_dist:     Minimum distance between points in embedding space.
                      Low (0.0–0.1) → tight, compact clusters (good for silhouette).
                      High (0.5–1.0) → spread out, better for visualization.

        metric:       CRITICAL — must be 'cosine' to preserve the geometry
                      of SBERT's unit-hypersphere output.

    Returns:
        np.ndarray of shape (N, n_components), L2-normalized (ready for
        spherical K-Means — all vectors on the unit hypersphere)
    """
    print(f"── UMAP reduction: 384D → {n_components}D  "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist}) ──")

    reducer = umap.UMAP(
        n_components=n_components,
        metric='cosine',            # Must match SBERT's similarity metric
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        low_memory=False,           # Full speed — use low_memory=True on <16GB RAM
        verbose=False
    )
    X_umap = reducer.fit_transform(X)

    # Re-normalize after UMAP — UMAP does not preserve unit norm
    X_umap_norm = normalize(X_umap, norm='l2')
    print(f"✅ UMAP embedding: {X_umap_norm.shape}  |  L2-normalized ✓")
    return X_umap_norm


def get_umap_2d(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Separate 2D UMAP projection FOR VISUALIZATION ONLY.
    Never use this for clustering — 2D loses too much structure.
    """
    reducer = umap.UMAP(
        n_components=2,
        metric='cosine',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    return reducer.fit_transform(X)


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    df = load_and_preprocess("data/reviews.csv")
    X  = get_sbert_embeddings(df['clean_review'].tolist())
    X_umap = get_umap_embedding(X, n_components=15, n_neighbors=15)
    print(f"\nFinal shape for clustering: {X_umap.shape}")
