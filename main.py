"""
main.py
-------
Master orchestration script for the complete sentiment clustering pipeline.

FULL PIPELINE:
    1. Data Generation       → data/reviews.csv (600 reviews, 200/class)
    2. Preprocessing         → minimal cleaning (SBERT-safe)
    3. SBERT Embedding       → 384D semantic unit vectors
    4. UMAP Reduction        → 15D manifold (cosine metric)
    5. k Sweep               → validate k=3 with silhouette + geodesic inertia
    6. σ Sweep               → tune curvature kernel width
    7. Spherical K-Means++   → geodesic init + curvature-aware centroids
    8. Evaluation            → 6 metrics + full visualization suite

USAGE:
    python main.py [--generate] [--n_per_class 200] [--k 3] [--n_init 20]

FLAGS:
    --generate          : regenerate reviews.csv (skip if data already exists)
    --n_per_class N     : reviews per sentiment class (default: 200, total: 600)
    --k K               : number of clusters (default: 3; use 0 to auto-tune)
    --n_init N          : K-Means random restarts (default: 20)
    --umap_components N : UMAP output dimensions for clustering (default: 15)
    --data PATH         : path to CSV file (default: data/reviews.csv)
"""

import argparse
import os
import json
import numpy as np

from data_generator import generate_dataset
from preprocess     import load_and_preprocess
from vectorizer     import get_sbert_embeddings, get_umap_embedding, get_umap_2d
from geometric_kmeans import (
    run_spherical_kmeans,
    tune_sigma,
    tune_k
)
from evaluate import (
    evaluate_clustering,
    plot_umap_clusters,
    plot_silhouette_per_sample,
    plot_sigma_sweep,
    plot_k_sweep,
    plot_comparison,
    infer_cluster_to_sentiment
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODER (ground truth → int for ARI/NMI)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_ENCODER = {'positive': 0, 'negative': 1, 'neutral': 2}


def print_banner(text: str):
    w = 65
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def main(args):
    print_banner("SPHERICAL K-MEANS++ + SBERT + UMAP SENTIMENT PIPELINE")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: DATA
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 1 — Data")
    if args.generate or not os.path.exists(args.data):
        print(f"Generating {args.n_per_class * 3} reviews ({args.n_per_class}/class)...")
        generate_dataset(n_per_class=args.n_per_class, output_path=args.data)
    else:
        print(f"Using existing data: {args.data}")

    df           = load_and_preprocess(args.data)
    reviews      = df['clean_review'].tolist()
    true_labels  = df['label'].tolist()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: SBERT EMBEDDINGS
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 2 — SBERT Sentence Embeddings (all-MiniLM-L6-v2)")
    X_sbert = get_sbert_embeddings(
        reviews,
        model_name='all-MiniLM-L6-v2',
        batch_size=64,
        normalize_output=True          # L2-normalize → unit hypersphere
    )
    # Cache embeddings — SBERT inference is the bottleneck on CPU
    np.save("outputs/sbert_embeddings.npy", X_sbert)
    print(f"   Embeddings cached → outputs/sbert_embeddings.npy")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: UMAP REDUCTION
    # ──────────────────────────────────────────────────────────────────────
    print_banner(f"Step 3 — UMAP Manifold Reduction (384D → {args.umap_components}D)")
    n_neighbors = max(5, min(50, int(np.sqrt(len(reviews)))))  # heuristic
    print(f"   Auto-selected n_neighbors={n_neighbors} (√N = {np.sqrt(len(reviews)):.1f})")

    X_umap = get_umap_embedding(
        X_sbert,
        n_components=args.umap_components,
        n_neighbors=n_neighbors,
        min_dist=0.05,
        random_state=42
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: k SELECTION
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 4 — Optimal k Selection (Silhouette + Geodesic Elbow)")
    if args.k == 0:
        best_k, k_vals, sil_scores, inertias = tune_k(
            X_umap, k_range=range(2, 8), sigma=0.5, n_init=10
        )
    else:
        best_k = args.k
        # Still run sweep for visualization
        _, k_vals, sil_scores, inertias = tune_k(
            X_umap, k_range=range(2, 8), sigma=0.5, n_init=10
        )
        print(f"   k fixed to {best_k} by user argument.")

    plot_k_sweep(k_vals, sil_scores, inertias, best_k,
                 save_path="outputs/k_sweep.png")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: σ TUNING
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 5 — Curvature Kernel σ Tuning")
    best_sigma, sigmas, sigma_scores = tune_sigma(
        X_umap, k=best_k, sigma_range=(0.05, 2.0), n_steps=30,
        n_init=10, random_state=42
    )
    plot_sigma_sweep(sigmas, sigma_scores, best_sigma,
                     save_path="outputs/sigma_sweep.png")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: FINAL SPHERICAL K-MEANS++
    # ──────────────────────────────────────────────────────────────────────
    print_banner(f"Step 6 — Final Spherical K-Means++ (k={best_k}, σ={best_sigma:.4f})")
    final_labels, final_centroids, final_score = run_spherical_kmeans(
        X_umap,
        k=best_k,
        sigma=best_sigma,
        max_iter=500,
        n_init=args.n_init,
        random_state=42
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7: EVALUATION
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 7 — Evaluation")
    metrics = evaluate_clustering(X_umap, final_labels, true_labels, LABEL_ENCODER)

    # Save metrics as JSON
    with open("outputs/metrics.json", "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
    print("   Metrics saved → outputs/metrics.json")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8: VISUALIZATIONS
    # ──────────────────────────────────────────────────────────────────────
    print_banner("Step 8 — Visualizations")

    # 2D UMAP for plotting (separate from clustering UMAP)
    print("   Generating 2D UMAP projection for visualization...")
    X_umap_2d = get_umap_2d(X_sbert, n_neighbors=n_neighbors, min_dist=0.1)

    cluster_map = infer_cluster_to_sentiment(final_labels, true_labels)
    print(f"   Cluster → Sentiment mapping: {cluster_map}")

    plot_umap_clusters(
        X_umap_2d, final_labels, true_labels, cluster_map,
        save_path="outputs/clusters_spherical.png"
    )
    plot_silhouette_per_sample(
        X_umap, final_labels,
        save_path="outputs/silhouette_per_sample.png"
    )
    plot_comparison(
        old_scores={
            "Standard\nK-Means\n(TF-IDF)":  0.0104,
            "Geometric\nK-Means\n(TF-IDF)": 0.0104,
        },
        new_score=final_score,
        save_path="outputs/comparison.png"
    )

    # ──────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print_banner("FINAL RESULTS")
    print(f"  Dataset size           : {len(reviews)} reviews ({len(reviews)//3}/class)")
    print(f"  Embedding              : SBERT all-MiniLM-L6-v2 (384D)")
    print(f"  UMAP dimensions        : {args.umap_components}D (n_neighbors={n_neighbors})")
    print(f"  Optimal k              : {best_k}")
    print(f"  Optimal σ              : {best_sigma:.4f}")
    print(f"  n_init                 : {args.n_init}")
    print(f"")
    print(f"  Silhouette Score       : {metrics['silhouette']:.4f}  "
          f"{'✅ TARGET MET' if metrics['silhouette'] >= 0.7 else '❌ Below target'}")
    print(f"  Davies-Bouldin Index   : {metrics['davies_bouldin']:.4f}  (lower = better)")
    print(f"  Calinski-Harabasz      : {metrics['calinski_harabasz']:.2f}  (higher = better)")
    print(f"  Adjusted Rand Index    : {metrics['ari']:.4f}")
    print(f"  Norm. Mutual Info      : {metrics['nmi']:.4f}")
    print(f"  V-Measure              : {metrics['v_measure']:.4f}")
    print(f"")
    baseline = 0.0104
    print(f"  Improvement over baseline : "
          f"+{((metrics['silhouette'] - baseline)/baseline*100):.0f}%  "
          f"({baseline:.4f} → {metrics['silhouette']:.4f})")
    print("═" * 65 + "\n")

    print("  Output files:")
    output_files = [
        "outputs/clusters_spherical.png   → UMAP 2D scatter (predicted + ground truth)",
        "outputs/silhouette_per_sample.png → Per-sample silhouette diagram",
        "outputs/sigma_sweep.png           → Curvature kernel σ sensitivity",
        "outputs/k_sweep.png               → k selection (silhouette + elbow)",
        "outputs/comparison.png            → Method comparison bar chart",
        "outputs/sbert_embeddings.npy      → Cached SBERT embeddings (384D)",
        "outputs/metrics.json              → All metrics as JSON",
    ]
    for f in output_files:
        print(f"    {f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical K-Means++ Sentiment Pipeline")
    parser.add_argument('--generate',       action='store_true',
                        help='Regenerate reviews.csv from templates')
    parser.add_argument('--n_per_class',    type=int, default=200,
                        help='Reviews per sentiment class (total = 3x this)')
    parser.add_argument('--k',              type=int, default=3,
                        help='Number of clusters (0 = auto-tune)')
    parser.add_argument('--n_init',         type=int, default=20,
                        help='K-Means random restarts')
    parser.add_argument('--umap_components',type=int, default=15,
                        help='UMAP output dimensions for clustering (not visualization)')
    parser.add_argument('--data',           type=str, default='data/reviews.csv',
                        help='Path to reviews CSV (review, label columns)')
    args = parser.parse_args()
    main(args)
