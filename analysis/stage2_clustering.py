#!/usr/bin/env python3
"""
Stage-2 clustering on the UNION of selected Stage-1 clusters.

Input:
  Stage-1 CSV (typically abs_clustered_utterances.csv) containing:
    - id, raw_text, y_true, cluster
    - yhat__<Method> columns (per-method predictions)
    - <Method> columns (per-method abs errors, if your stage-1 was abs mode)
Optionally:
  - engineered features CSV (merged by id)

Feature spaces:
  --feature-space abs_error  -> use per-method abs error vector (method error cols)
  --feature-space yhat       -> use per-method prediction vector (yhat__* cols)
  --feature-space engineered -> use engineered features from --engineered-csv
  --feature-space concat     -> concatenate yhat + engineered (or abs_error + engineered if you want)

Outputs (single pooled run):
  - stage2_clustered_union.csv
  - stage2_cluster_summary_union.csv
  - stage2_cluster_centroids_union.csv
  - stage2_pca_union.png
  - stage2_tsne_union.png (optional)

Example:
  python stage2_clustering_union.py \
    --in-csv /scratch/.../abs_clustered_utterances.csv \
    --out-dir /scratch/.../stage2_union \
    --clusters 0 3 \
    --k2 4 \
    --feature-space abs_error \
    --standardize \
    --tsne
"""

import os
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CORE_COLS = {"id", "raw_text", "y_true", "cluster"}


def infer_error_method_cols(df: pd.DataFrame) -> list[str]:
    """Method abs-error cols: numeric cols that are not core and not yhat__*."""
    cols = []
    for c in df.columns:
        if c in CORE_COLS:
            continue
        if c.startswith("yhat__"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def infer_yhat_cols(df: pd.DataFrame) -> list[str]:
    """Per-method prediction cols: yhat__*"""
    return [c for c in df.columns if c.startswith("yhat__") and pd.api.types.is_numeric_dtype(df[c])]


def safe_k(n_samples: int, requested_k: int) -> int:
    if n_samples < 3:
        return 0
    k = int(requested_k)
    if k < 2:
        k = 2
    if k >= n_samples:
        k = max(2, n_samples - 1)
    return k


def run_kmeans(X: np.ndarray, k: int, seed: int = 42, standardize: bool = True):
    X_used = X.copy()
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X_used)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(X_used)
    return labels, km, scaler, X_used

def elbow_curve(
    Xs: np.ndarray,
    out_dir: str,
    k_min: int = 2,
    k_max: int = 12,
    seed: int = 42,
):
    """
    Saves:
      - <prefix>_elbow.csv: columns [k, inertia]
      - <prefix>_elbow.png: inertia vs k
    """
    os.makedirs(out_dir, exist_ok=True)

    n = Xs.shape[0]
    if n < 3:
        print("[elbow] Too few samples to compute elbow curve.")
        return None

    k_max_eff = min(k_max, n - 1)
    if k_max_eff < k_min:
        print("[elbow] Too few samples for requested k range.")
        return None

    ks, inertias = [], []
    for k in range(k_min, k_max_eff + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        km.fit(Xs)
        ks.append(k)
        inertias.append(float(km.inertia_))

    df = pd.DataFrame({"k": ks, "inertia": inertias})
    df.to_csv(os.path.join(out_dir, f"_elbow.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plot_title = "Stage 2 Clustering Elbow Curve (MOSEI)"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"_elbow.png"), dpi=180)
    plt.close()

    print(f"[elbow] Saved elbow.csv and elbow.png")
    return df



def plot_pca_tsne(X_used: np.ndarray, labels: np.ndarray, out_dir: str, do_tsne: bool):
    os.makedirs(out_dir, exist_ok=True)
    n = X_used.shape[0]
    if n < 2:
        return

    # PCA
    try:
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(X_used)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(Xp[:, 0], Xp[:, 1], c=labels, s=12, alpha=0.8)
        plt.title("Stage-2 (union) — PCA scatter")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(sc, label="cluster2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "stage2_pca_union.png"), dpi=180)
        plt.close()
    except Exception as e:
        print(f"[plot] PCA failed: {e}")

    # t-SNE
    if do_tsne and n >= 10:
        try:
            perp = int(min(30, max(5, (n - 1) // 3)))
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                random_state=42,
                n_iter=1000,
            )
            Xt = tsne.fit_transform(X_used)
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(Xt[:, 0], Xt[:, 1], c=labels, s=12, alpha=0.8)
            plt.title("Stage-2 (union) — t-SNE scatter")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(sc, label="cluster2")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "stage2_tsne_union.png"), dpi=180)
            plt.close()
        except Exception as e:
            print(f"[plot] t-SNE failed: {e}")


def summarize(df_sub: pd.DataFrame, err_cols_for_bestworst: list[str], labels2: np.ndarray) -> pd.DataFrame:
    """
    Summary per stage-2 cluster:
      - size/pct
      - y_true mean/std
      - most common best/worst model (based on abs error cols, if available)
    """
    out = df_sub.copy()
    out["cluster2"] = labels2

    rows = []
    n_total = len(out)

    for c2 in sorted(out["cluster2"].unique()):
        sub = out[out["cluster2"] == c2]

        rec = {
            "cluster2": int(c2),
            "size": int(len(sub)),
            "pct": float(len(sub) / n_total),
            "y_true_mean": float(sub["y_true"].mean()),
            "y_true_std": float(sub["y_true"].std(ddof=0)),
        }

        if err_cols_for_bestworst:
            err = sub[err_cols_for_bestworst].to_numpy(dtype=np.float32)
            best_idx = np.argmin(err, axis=1)
            worst_idx = np.argmax(err, axis=1)
            best_methods = [err_cols_for_bestworst[i] for i in best_idx]
            worst_methods = [err_cols_for_bestworst[i] for i in worst_idx]
            bc = Counter(best_methods)
            wc = Counter(worst_methods)
            rec["most_common_best_method"] = bc.most_common(1)[0][0] if bc else None
            rec["most_common_worst_method"] = wc.most_common(1)[0][0] if wc else None
        else:
            rec["most_common_best_method"] = None
            rec["most_common_worst_method"] = None

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("cluster2").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, required=True, help="Stage-1 abs_clustered_utterances.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder for stage-2 results")
    ap.add_argument("--clusters", nargs="+", type=int, required=True,
                    help="Stage-1 cluster ids to pool together (e.g. 0 3)")
    ap.add_argument("--k2", type=int, default=4, help="Stage-2 KMeans K")
    ap.add_argument("--standardize", action="store_true", help="Standardize features before KMeans")
    ap.add_argument("--tsne", action="store_true", help="Also plot t-SNE")

    ap.add_argument("--feature-space", type=str, default="abs_error",
                    choices=["abs_error", "yhat", "engineered", "concat"],
                    help="Which feature space to cluster in")

    ap.add_argument("--engineered-csv", type=str, default=None,
                    help="CSV with engineered features, must include 'id' column")
    ap.add_argument("--engineered-join", type=str, default="inner", choices=["inner", "left"],
                    help="How to join engineered features onto stage-1 subset")
    ap.add_argument("--engineered-cols", nargs="*", default=None,
                    help="Which engineered columns to use (default: all numeric except id/cluster/y_true/raw_text)")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise SystemExit(f"Missing input CSV: {in_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "cluster" not in df.columns:
        raise SystemExit("Input CSV must have a 'cluster' column (stage-1 cluster).")

    # Pool requested stage-1 clusters
    df_sub = df[df["cluster"].isin(args.clusters)].copy()
    print(f"[info] loaded stage-1 rows={len(df)}")
    print(f"[info] pooled stage-1 clusters={args.clusters} -> n={len(df_sub)}")

    if df_sub.empty:
        raise SystemExit("No rows left after filtering to requested stage-1 clusters.")

    # Ensure required cols
    for c in ["id", "y_true", "raw_text"]:
        if c not in df_sub.columns:
            raise SystemExit(f"Missing required column in stage-1 CSV: {c}")

    # Identify columns available
    err_cols = infer_error_method_cols(df_sub)
    yhat_cols = infer_yhat_cols(df_sub)

    # Prepare engineered features if needed
    eng_df = None
    eng_cols = []
    if args.feature_space in {"engineered", "concat"}:
        if not args.engineered_csv:
            raise SystemExit("--engineered-csv is required for feature-space engineered/concat")

        eng_path = Path(args.engineered_csv)
        if not eng_path.exists():
            raise SystemExit(f"Missing engineered CSV: {eng_path}")

        eng_df = pd.read_csv(eng_path)
        if "id" not in eng_df.columns:
            raise SystemExit("Engineered CSV must contain an 'id' column")

        # infer engineered cols
        if args.engineered_cols and len(args.engineered_cols) > 0:
            eng_cols = list(args.engineered_cols)
        else:
            exclude = {"id", "raw_text", "y_true", "cluster", "cluster2", "cluster12"}
            eng_cols = [c for c in eng_df.columns
                        if c not in exclude and pd.api.types.is_numeric_dtype(eng_df[c])]

        if not eng_cols:
            raise SystemExit("No engineered numeric columns found/selected.")

        # merge
        how = args.engineered_join
        df_sub = df_sub.merge(eng_df[["id"] + eng_cols], on="id", how=how)
        print(f"[info] merged engineered features ({how}) -> n={len(df_sub)} rows, eng_cols={len(eng_cols)}")

    # Build X according to feature space
    if args.feature_space == "abs_error":
        if not err_cols:
            raise SystemExit("No abs-error method columns detected in stage-1 CSV.")
        feat_cols = err_cols

    elif args.feature_space == "yhat":
        if not yhat_cols:
            raise SystemExit("No yhat__* columns detected in stage-1 CSV.")
        feat_cols = yhat_cols

    elif args.feature_space == "engineered":
        feat_cols = eng_cols

    elif args.feature_space == "concat":
        # default: yhat + engineered (most interpretable)
        if not yhat_cols:
            raise SystemExit("concat requested but no yhat__* columns found in stage-1 CSV.")
        if not eng_cols:
            raise SystemExit("concat requested but no engineered cols found (check engineered CSV/cols).")
        feat_cols = yhat_cols + eng_cols

    else:
        raise SystemExit("Unknown feature space (should be impossible due to argparse choices).")

    # Coerce numeric and drop NAs
    for c in feat_cols:
        df_sub[c] = pd.to_numeric(df_sub[c], errors="coerce")
    df_sub["y_true"] = pd.to_numeric(df_sub["y_true"], errors="coerce")
    df_sub = df_sub.dropna(subset=["id", "y_true"] + feat_cols).copy()

    if df_sub.empty:
        raise SystemExit("No rows remain after dropping NaNs in selected feature columns.")

    X = df_sub[feat_cols].to_numpy(dtype=np.float32)
    k = safe_k(len(df_sub), args.k2)
    if k == 0:
        raise SystemExit("Too few samples in pooled subset for KMeans.")

    print(f"[info] feature-space={args.feature_space} | X shape={X.shape} | k2={k}")

    labels2, km, scaler, X_used = run_kmeans(X, k, seed=42, standardize=args.standardize)
  
    elbow_curve(X, out_dir=args.out_dir, k_min=2, k_max=12)  

    # Attach stage2 label
    df_sub["cluster2"] = labels2
    df_sub["cluster12"] = df_sub["cluster"].astype(str) + "_" + df_sub["cluster2"].astype(str)

    # Save full
    out_full = Path(args.out_dir) / "stage2_clustered_union.csv"
    keep_cols = ["id", "raw_text", "y_true", "cluster", "cluster2", "cluster12"]
    # keep predictions too if present
    keep_cols += [c for c in df_sub.columns if c.startswith("yhat__")]
    # keep error cols too if present
    keep_cols += [c for c in err_cols if c in df_sub.columns]
    # keep engineered if used
    keep_cols += [c for c in eng_cols if c in df_sub.columns]
    # de-dup preserve order
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    df_sub.sort_values(["cluster2", "id"]).to_csv(out_full, index=False, columns=keep_cols, encoding="utf-8-sig")

    # Summary (best/worst only makes sense if err cols exist)
    sum_df = summarize(df_sub, err_cols, labels2)
    out_sum = Path(args.out_dir) / "stage2_cluster_summary_union.csv"
    sum_df.to_csv(out_sum, index=False, encoding="utf-8-sig")

    # Centroids
    if scaler is not None:
        cent = scaler.inverse_transform(km.cluster_centers_)
    else:
        cent = km.cluster_centers_
    cent_df = pd.DataFrame(cent, columns=feat_cols)
    cent_df.insert(0, "cluster2", np.arange(len(cent_df)))
    out_cent = Path(args.out_dir) / "stage2_cluster_centroids_union.csv"
    cent_df.to_csv(out_cent, index=False, encoding="utf-8-sig")

    # Plots
    plot_pca_tsne(X_used=X_used, labels=labels2, out_dir=args.out_dir, do_tsne=args.tsne)

    print("[saved]")
    print(" ", out_full)
    print(" ", out_sum)
    print(" ", out_cent)
    print("\n[done] Stage-2 (union) clustering complete.")


if __name__ == "__main__":
    main()
