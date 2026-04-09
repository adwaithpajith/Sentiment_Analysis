"""
geometric_kmeans.py
-------------------
Spherical K-Means++ with Curvature-Aware Centroid Updates.

MATHEMATICAL FOUNDATION
━━━━━━━━━━━━━━━━━━━━━━
After L2 normalization, all data vectors lie on S^(d-1) — the surface of
the d-dimensional unit hypersphere — which is a Riemannian manifold with
constant positive curvature.

PROBLEM WITH STANDARD K-MEANS ON THIS MANIFOLD:
    Centroid ← arithmetic mean of assigned vectors
    The arithmetic mean of unit vectors does NOT lie on the unit sphere.
    Projecting it back (renormalization) is an approximation that ignores
    the manifold's intrinsic geometry.

OUR APPROACH — Three geometric corrections:

1. INITIALIZATION: Geodesic K-Means++
   Standard K-Means++ seeds centroids with probability ∝ d_euclidean².
   We replace Euclidean distance with geodesic (arc-length) distance:
       d_geo(u, v) = arccos(u · v)   ∈ [0, π]
   This correctly spreads seeds across the sphere surface.

2. ASSIGNMENT: Cosine Similarity (exact for unit vectors)
   argmin_c d_geo(x, c) ≡ argmax_c (x · c)   when ‖x‖=‖c‖=1
   So assignment is just a matrix multiplication — fast and exact.

3. UPDATE: Curvature-Aware Fréchet Mean
   The Fréchet mean on S^(d-1) is the point that minimizes the sum of
   squared geodesic distances to all assigned vectors. It is approximated
   by a geodesic Gaussian-weighted mean:

       weight_i = exp( -d_geo(x_i, c)² / 2σ² )
       new_c    = normalize( Σ weight_i · x_i )

   Points far along the geodesic from the current centroid are
   DOWN-WEIGHTED. This:
     - Reduces pull from cluster-boundary/outlier points
     - Respects the sphere's curvature (contribution decays with arc length)
     - Converges to true Fréchet mean as σ → 0
     - Degenerates to standard spherical centroid as σ → ∞

   σ (sigma) is a hyperparameter controlling kernel width. It is tuned
   automatically via silhouette-score-guided sweep in this file.
"""

import numpy as np
from sklearn.metrics import silhouette_score


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def geodesic_distances_to_centroid(X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Compute arc-length distances from all rows of X to a single centroid.
    Both X and centroid must be L2-normalized (on unit hypersphere).

    d_geo(x, c) = arccos( clip(x · c, -1, 1) )

    The clip is essential — floating point errors can push dot products
    slightly outside [-1, 1], causing arccos to return NaN.

    Returns: np.ndarray of shape (N,) in [0, π]
    """
    dot_products = np.clip(X @ centroid, -1.0, 1.0)
    return np.arccos(dot_products)


def curvature_aware_centroid(
    vectors: np.ndarray,
    current_centroid: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Geodesic Gaussian-weighted Fréchet Mean.

    Args:
        vectors         : (M, d) — L2-normalized vectors assigned to this cluster
        current_centroid: (d,)   — current centroid (L2-normalized)
        sigma           : kernel width in radians (tuned externally)

    Returns: (d,) — updated centroid, L2-normalized (back on sphere)
    """
    geodesic_dists = geodesic_distances_to_centroid(vectors, current_centroid)
    weights = np.exp(-(geodesic_dists ** 2) / (2.0 * sigma ** 2))  # (M,)

    # Weighted mean in embedding space
    weighted_mean = (weights[:, None] * vectors).sum(axis=0)
    norm = np.linalg.norm(weighted_mean)

    if norm < 1e-12:
        # Degenerate: all weights collapsed — fall back to uniform Fréchet mean
        fallback = vectors.mean(axis=0)
        norm_fb = np.linalg.norm(fallback)
        if norm_fb < 1e-12:
            return current_centroid.copy()  # Keep old centroid unchanged
        return fallback / norm_fb

    return weighted_mean / norm   # Project back onto S^(d-1)


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS++ INITIALIZATION (GEODESIC)
# ─────────────────────────────────────────────────────────────────────────────

def geodesic_kmeans_pp_init(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    K-Means++ seeding adapted for the unit hypersphere.

    Seeding strategy:
        c_1 ~ Uniform(X)
        c_i ~ Categorical( d_geo(x, nearest_c)² / Z )  for i = 2..k

    Using geodesic² (instead of Euclidean²) correctly captures angular
    separation between candidate centroids on the sphere surface.

    Returns: (k, d) array of initial centroids, all L2-normalized
    """
    n = len(X)
    first_idx = rng.integers(n)
    centroids = [X[first_idx].copy()]

    for _ in range(k - 1):
        # (N, current_k) geodesic distance matrix to all current centroids
        dist_matrix = np.column_stack([
            geodesic_distances_to_centroid(X, c) for c in centroids
        ])
        # Distance to nearest centroid for each point
        min_dists = dist_matrix.min(axis=1)           # (N,)
        probs     = min_dists ** 2
        probs    /= probs.sum()
        next_idx  = rng.choice(n, p=probs)
        centroids.append(X[next_idx].copy())

    return np.array(centroids)   # (k, d)


# ─────────────────────────────────────────────────────────────────────────────
# SPHERICAL K-MEANS++ — MAIN ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def run_spherical_kmeans(
    X: np.ndarray,
    k: int = 3,
    sigma: float = 0.5,
    max_iter: int = 500,
    n_init: int = 20,
    tol: float = 1e-6,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Spherical K-Means++ with Curvature-Aware Centroid Updates.

    Args:
        X            : (N, d) L2-normalized feature matrix (on unit sphere)
        k            : number of clusters
        sigma        : geodesic kernel width for curvature-aware centroid
        max_iter     : max EM iterations per initialization
        n_init       : number of random restarts (best silhouette kept)
        tol          : convergence threshold on centroid movement (radians)
        random_state : base seed (incremented per init run)

    Returns:
        best_labels    : (N,) cluster assignments from best run
        best_centroids : (k, d) final centroids from best run
        best_score     : silhouette score (cosine metric) of best run
    """
    # Safety: enforce unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)

    best_labels, best_centroids, best_score = None, None, -1.0

    for run in range(n_init):
        rng        = np.random.default_rng(random_state + run)
        centroids  = geodesic_kmeans_pp_init(X, k, rng)
        labels     = np.full(len(X), -1, dtype=int)

        for iteration in range(max_iter):
            # ── E-Step: Assign each point to nearest centroid by cosine sim
            # (equivalent to geodesic distance for unit vectors)
            similarities = X @ centroids.T          # (N, k) — fast matmul
            new_labels   = np.argmax(similarities, axis=1)

            # ── Convergence check
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # ── M-Step: Update centroids with curvature-aware Fréchet mean
            new_centroids = np.empty_like(centroids)
            for c_idx in range(k):
                mask = labels == c_idx
                if mask.sum() == 0:
                    # Empty cluster: reinitialize to random point
                    new_centroids[c_idx] = X[rng.integers(len(X))]
                else:
                    new_centroids[c_idx] = curvature_aware_centroid(
                        X[mask], centroids[c_idx], sigma=sigma
                    )

            # ── Centroid movement check (in geodesic radians)
            movements = geodesic_distances_to_centroid(
                new_centroids,
                centroids[0]  # placeholder — we check per centroid below
            )
            max_movement = max(
                geodesic_distances_to_centroid(
                    new_centroids[c:c+1], centroids[c]
                )[0]
                for c in range(k)
            )
            centroids = new_centroids
            if max_movement < tol:
                break

        # ── Score this run
        n_unique = len(np.unique(labels))
        if n_unique == k:
            score = silhouette_score(X, labels, metric='cosine')
            if score > best_score:
                best_score     = score
                best_labels    = labels.copy()
                best_centroids = centroids.copy()
        elif n_unique < k and best_labels is None:
            # Degenerate but record to avoid returning None
            best_labels    = labels.copy()
            best_centroids = centroids.copy()
            best_score     = -1.0

    cluster_sizes = {i: int((best_labels == i).sum()) for i in range(k)}
    print(f"✅ Spherical K-Means++ | k={k} | σ={sigma:.3f} | "
          f"Silhouette={best_score:.4f} | Cluster sizes: {cluster_sizes}")
    return best_labels, best_centroids, best_score


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_sigma(
    X: np.ndarray,
    k: int = 3,
    sigma_range: tuple = (0.05, 2.0),
    n_steps: int = 25,
    n_init: int = 10,
    random_state: int = 42
) -> tuple[float, list, list]:
    """
    Grid search over sigma values, maximizing silhouette score.
    Uses fewer n_init per step for speed; final run uses full n_init.

    Returns:
        best_sigma : float
        sigmas     : list of sigma values tested
        scores     : corresponding silhouette scores
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_steps)
    scores = []

    print(f"── Tuning σ over {n_steps} values in [{sigma_range[0]}, {sigma_range[1]}] ──")
    for sigma in sigmas:
        _, _, score = run_spherical_kmeans(
            X, k=k, sigma=sigma, n_init=n_init, random_state=random_state
        )
        scores.append(score)

    best_sigma = float(sigmas[np.argmax(scores)])
    print(f"✅ Best σ: {best_sigma:.4f}  (Silhouette: {max(scores):.4f})\n")
    return best_sigma, sigmas.tolist(), scores


def tune_k(
    X: np.ndarray,
    k_range: range = range(2, 8),
    sigma: float = 0.5,
    n_init: int = 10,
    random_state: int = 42
) -> tuple[int, list, list]:
    """
    Sweep over k values to find optimal number of clusters.
    Uses silhouette score (higher = better separation).
    Also collect inertia (lower = tighter clusters) for elbow analysis.

    Note: For sentiment analysis with positive/negative/neutral labels,
    k=3 is ground truth. This sweep validates that assumption.
    """
    silhouette_scores = []
    inertias = []

    print(f"── Tuning k over {list(k_range)} ──")
    for k in k_range:
        _, centroids, score = run_spherical_kmeans(
            X, k=k, sigma=sigma, n_init=n_init, random_state=random_state
        )
        silhouette_scores.append(score)

        # Compute inertia: sum of squared geodesic distances to nearest centroid
        dist_matrix = np.column_stack([
            geodesic_distances_to_centroid(X, c) for c in centroids
        ])
        min_dists = dist_matrix.min(axis=1)
        inertias.append(float((min_dists ** 2).sum()))

    best_k = int(k_range[np.argmax(silhouette_scores)])
    print(f"✅ Best k: {best_k}  (Silhouette: {max(silhouette_scores):.4f})\n")
    return best_k, list(k_range), silhouette_scores, inertias
