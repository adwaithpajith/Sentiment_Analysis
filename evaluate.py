"""
evaluate.py
-----------
Full evaluation suite for clustering results.

Metrics:
    Silhouette Score     : cluster cohesion vs separation (−1 to 1, higher=better)
    Davies-Bouldin Index : intra-cluster / inter-cluster distance ratio (lower=better)
    Calinski-Harabasz    : between-cluster / within-cluster dispersion (higher=better)
    Adjusted Rand Index  : similarity to ground truth, corrected for chance (0 to 1)
    Normalized Mutual Info: information-theoretic agreement with ground truth (0 to 1)
    V-Measure            : harmonic mean of homogeneity and completeness

All plots saved to outputs/.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    confusion_matrix
)
import os

os.makedirs("outputs", exist_ok=True)

SENTIMENT_COLORS = {
    'positive': '#2196F3',
    'negative': '#F44336',
    'neutral':  '#4CAF50',
}
CLUSTER_COLORS = ['#1565C0', '#B71C1C', '#1B5E20', '#F57F17', '#4A148C']


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels_str: list[str],
    label_encoder: dict
) -> dict:
    """
    Compute full suite of internal and external clustering metrics.

    Internal metrics (no ground truth needed):
        - Silhouette Score (cosine metric)
        - Davies-Bouldin Index
        - Calinski-Harabasz Score

    External metrics (require ground truth):
        - Adjusted Rand Index
        - Normalized Mutual Information
        - V-Measure (homogeneity + completeness)
    """
    y_true = np.array([label_encoder[l] for l in true_labels_str])

    results = {
        # Internal
        'silhouette':        silhouette_score(X, labels, metric='cosine'),
        'davies_bouldin':    davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        # External
        'ari':        adjusted_rand_score(y_true, labels),
        'nmi':        normalized_mutual_info_score(y_true, labels),
        'v_measure':  v_measure_score(y_true, labels),
    }

    print("\n" + "─"*55)
    print("  CLUSTERING EVALUATION METRICS")
    print("─"*55)
    print("  Internal Metrics (no ground truth required):")
    print(f"    Silhouette Score     : {results['silhouette']:.4f}  "
          f"{'✅' if results['silhouette'] >= 0.7 else '⚠️ '} (target: ≥ 0.70)")
    print(f"    Davies-Bouldin Index : {results['davies_bouldin']:.4f}  "
          f"(lower is better)")
    print(f"    Calinski-Harabasz   : {results['calinski_harabasz']:.2f}  "
          f"(higher is better)")
    print("  External Metrics (vs ground truth labels):")
    print(f"    Adjusted Rand Index  : {results['ari']:.4f}  "
          f"(1.0 = perfect match)")
    print(f"    Normalized Mutual Info: {results['nmi']:.4f}")
    print(f"    V-Measure            : {results['v_measure']:.4f}")
    print("─"*55 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_clusters(
    X_umap_2d: np.ndarray,
    pred_labels: np.ndarray,
    true_labels_str: list[str],
    cluster_to_sentiment: dict,
    save_path: str = "outputs/clusters_spherical.png"
):
    """
    Side-by-side 2D UMAP scatter plots:
      Left  — predicted cluster assignments (color by cluster)
      Right — ground truth labels (color by sentiment)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Spherical K-Means++ with Curvature-Aware Centroids\n"
                 "UMAP 2D Projection", fontsize=14, fontweight='bold', y=1.01)

    k = len(np.unique(pred_labels))

    # ── Left: Predicted clusters
    for c_id in range(k):
        mask = pred_labels == c_id
        name = cluster_to_sentiment.get(c_id, f"Cluster {c_id}")
        axes[0].scatter(
            X_umap_2d[mask, 0], X_umap_2d[mask, 1],
            c=CLUSTER_COLORS[c_id], label=f"Cluster {c_id} ({name})",
            s=60, edgecolors='k', linewidths=0.5, alpha=0.85
        )
    axes[0].set_title("Predicted Clusters", fontsize=12)
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)

    # ── Right: Ground truth
    for sentiment, color in SENTIMENT_COLORS.items():
        mask = np.array([t == sentiment for t in true_labels_str])
        marker = {'positive': 'o', 'negative': 's', 'neutral': '^'}[sentiment]
        axes[1].scatter(
            X_umap_2d[mask, 0], X_umap_2d[mask, 1],
            c=color, label=sentiment.capitalize(),
            marker=marker, s=60, edgecolors='k', linewidths=0.5, alpha=0.85
        )
    axes[1].set_title("Ground Truth Labels", fontsize=12)
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Cluster scatter plot saved → {save_path}")


def plot_silhouette_per_sample(
    X: np.ndarray,
    labels: np.ndarray,
    save_path: str = "outputs/silhouette_per_sample.png"
):
    """
    Silhouette diagram — per-sample silhouette coefficients grouped by cluster.
    Wide bars = tight clusters. Uniform height = balanced cluster quality.
    """
    from sklearn.metrics import silhouette_samples

    sample_sils = silhouette_samples(X, labels, metric='cosine')
    mean_sil    = sample_sils.mean()
    k           = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for c_id in range(k):
        c_silhouette = np.sort(sample_sils[labels == c_id])
        size_c = len(c_silhouette)
        y_upper = y_lower + size_c
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, c_silhouette,
            facecolor=CLUSTER_COLORS[c_id],
            edgecolor=CLUSTER_COLORS[c_id],
            alpha=0.7,
            label=f"Cluster {c_id}"
        )
        ax.text(-0.05, y_lower + size_c / 2, str(c_id), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=mean_sil, color='crimson', linestyle='--', linewidth=1.5,
               label=f"Mean = {mean_sil:.4f}")
    ax.axvline(x=0.7, color='orange', linestyle=':', linewidth=1.5,
               label="Target = 0.70")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Per-Sample Silhouette Coefficients by Cluster", fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Silhouette diagram saved → {save_path}")


def plot_sigma_sweep(
    sigmas: list,
    scores: list,
    best_sigma: float,
    save_path: str = "outputs/sigma_sweep.png"
):
    """Silhouette score vs. sigma — visualizes the curvature kernel sensitivity."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sigmas, scores, marker='o', color='seagreen',
            linewidth=2, markersize=5)
    ax.axvline(best_sigma, color='crimson', linestyle='--', linewidth=1.5,
               label=f"Best σ = {best_sigma:.4f}")
    ax.set_xlabel("σ — Geodesic Gaussian Kernel Width (radians)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Curvature-Aware Centroid: σ Sensitivity Analysis", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Sigma sweep plot saved → {save_path}")


def plot_k_sweep(
    k_range: list,
    silhouette_scores: list,
    inertias: list,
    best_k: int,
    save_path: str = "outputs/k_sweep.png"
):
    """Elbow + silhouette plot for k selection."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Silhouette
    axes[0].plot(k_range, silhouette_scores, marker='o', color='steelblue',
                 linewidth=2)
    axes[0].axvline(best_k, color='crimson', linestyle='--',
                    label=f"Best k={best_k}")
    axes[0].axhline(0.7, color='orange', linestyle=':', label="Target 0.70")
    axes[0].set_xlabel("k (Number of Clusters)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("k vs. Silhouette Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Elbow (geodesic inertia)
    axes[1].plot(k_range, inertias, marker='o', color='darkorange', linewidth=2)
    axes[1].axvline(best_k, color='crimson', linestyle='--',
                    label=f"Best k={best_k}")
    axes[1].set_xlabel("k (Number of Clusters)")
    axes[1].set_ylabel("Geodesic Inertia (sum of squared arc distances)")
    axes[1].set_title("Elbow Method — Geodesic Inertia")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ k sweep plot saved → {save_path}")


def plot_comparison(
    old_scores: dict,
    new_score: float,
    save_path: str = "outputs/comparison.png"
):
    """Bar chart comparing all methods including the new pipeline."""
    all_scores = {**old_scores, "Spherical K-Means++\n+ Curvature-Aware\n(SBERT + UMAP)": new_score}
    methods    = list(all_scores.keys())
    values     = list(all_scores.values())

    colors = ['#90A4AE'] * (len(methods) - 1) + ['#1B5E20']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', width=0.4, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.4f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.axhline(0.7, color='crimson', linestyle='--', linewidth=1.5,
               label='Target ≥ 0.70', zorder=4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Silhouette Score  (higher = better)", fontsize=11)
    ax.set_title("Silhouette Score Comparison — All Methods", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Comparison chart saved → {save_path}")


def infer_cluster_to_sentiment(labels: np.ndarray, true_labels_str: list[str]) -> dict:
    """
    Map cluster IDs → sentiment labels by majority vote.
    Needed because K-Means cluster indices are arbitrary.
    """
    k          = len(np.unique(labels))
    sentiments = sorted(set(true_labels_str))
    mapping    = {}

    for c_id in range(k):
        mask      = labels == c_id
        if mask.sum() == 0:
            mapping[c_id] = 'unknown'
            continue
        assigned  = [true_labels_str[i] for i in range(len(labels)) if mask[i]]
        majority  = max(set(assigned), key=assigned.count)
        mapping[c_id] = majority

    return mapping
