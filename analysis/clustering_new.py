#!/usr/bin/env python3
"""
ABS-ONLY: Cluster utterances by *absolute error patterns* across models from saved preds_test.json files.

Expected JSON format in each experiment folder (e.g. .../kernel_fusion_run3/preds_test.json):
{
  "split": "test",
  "exp_name": "...",
  "threshold": 0.0,
  "ids": [...],
  "raw_texts": [...],
  "y_true": [...],
  "yhat": [...]
}

Pipeline:
1) Scan experiment folders for preds_test.json under *_{run_suffix}
2) Load predictions across methods
3) Align utterances by id
4) Build ABS error matrix per utterance across methods
5) (NEW) Elbow plot: inertia vs K to justify number of clusters
6) Run KMeans on standardized error matrix
7) Save cluster outputs + plots

Usage:
  python cluster_abs_errors.py \
    --root /hpctmp/scratch/e0968015/chsims/FusionRuns \
    --out-dir /hpctmp/scratch/e0968015/chsims/FusionRuns/cluster_analysis_abs \
    --run-suffix run3 \
    --dataset chsims \
    --audio-dim 35 \
    --n-clusters 4
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")  # safe for HPC/headless
import matplotlib.pyplot as plt

# clustering / dimensionality reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config: canonical method names
# -----------------------------
METHOD_ALIASES = {
    "early_fusion": "Early Fusion",
    "trimodal_attention": "Tri-Modal Attention",
    "tri_modal_attention": "Tri-Modal Attention",
    "tri-modal_attention": "Tri-Modal Attention",
    "text_guided_attn": "Text-Guided Attention",
    "text_guided_attention": "Text-Guided Attention",
    "kernel_fusion": "Kernel Fusion",
    "graph_fusion": "Graph Fusion",
    "tensor_fusion": "Tensor Fusion",
    "bimodal_fusion": "Bi-modal Fusion",
    "bi_modal_fusion": "Bi-modal Fusion",
    "bi-modal_fusion": "Bi-modal Fusion",
    "audio_guided_attn": "Audio-Guided Attention",
    "audio_guided_attention": "Audio-Guided Attention",
    "vision_guided_attn": "Vision-Guided Attention",
    "vision_guided_attention": "Vision-Guided Attention",
    "lowrank_fusion": "Low-Rank Fusion",
    "low_rank_fusion": "Low-Rank Fusion",
    "low-rank_fusion": "Low-Rank Fusion",
    "tfn": "Tensor Fusion",
    "lmf": "Low-Rank Fusion",
}


# ---------------------------------
# Utilities: parsing and file scan
# ---------------------------------
def infer_dataset_from_path(path_str: str) -> str:
    p = path_str.lower()
    if "/mosei/" in p or p.endswith("/mosei") or "mosei" in p:
        return "mosei"
    if "/chsims/" in p or "chsims" in p:
        return "chsims"
    return "unknown"


def infer_audio_dim_from_path_or_json(path_str: str, data: dict) -> str:
    """
    Best-effort inference. If not obvious, returns 'unknown' (not forced to 35).
    """
    p = path_str.lower()

    m = re.search(r'[_\-](16|35)[_\-]', p)
    if m:
        return m.group(1)

    m = re.search(r'\ba(16|35)\b', p)
    if m:
        return m.group(1)
    m = re.search(r'audio[_\-]?(16|35)\b', p)
    if m:
        return m.group(1)

    exp_name = str(data.get("exp_name", "")).lower()
    m = re.search(r'[_\-](16|35)[_\-]', exp_name)
    if m:
        return m.group(1)

    return "unknown"


def normalise_method_name(folder_name: str) -> str:
    name = folder_name.lower()
    name = re.sub(r"_run\d+$", "", name)

    candidates = []
    for k, v in METHOD_ALIASES.items():
        if name == k or name.startswith(k) or k in name:
            candidates.append((len(k), v))
    if candidates:
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

    name = name.replace("_", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in name.split())


def find_prediction_jsons(root: str, run_suffix: str = "run3", dataset_filter=None, audio_dim_filter=None):
    """
    Scan recursively for preds_test.json under folders ending in *_{run_suffix}
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    pred_files = []
    for p in root_path.rglob("preds_test.json"):
        parent = p.parent.name.lower()

        if run_suffix:
            if not parent.endswith(f"_{run_suffix.lower()}"):
                continue

        if dataset_filter:
            ds_guess = infer_dataset_from_path(str(p))
            if ds_guess != dataset_filter.lower():
                continue

        # only filter on explicit dim tokens if present in path
        if audio_dim_filter is not None:
            path_l = str(p).lower()
            explicit_dim = None
            m = re.search(r'[_\-](16|35)[_\-]', path_l)
            if m:
                explicit_dim = m.group(1)
            m2 = re.search(r'\ba(16|35)\b', path_l)
            if m2:
                explicit_dim = explicit_dim or m2.group(1)

            if explicit_dim is not None and str(audio_dim_filter) != explicit_dim:
                continue

        pred_files.append(str(p))

    return sorted(pred_files)


# ---------------------------------
# Loading / alignment
# ---------------------------------
def load_preds_json(fp: str) -> pd.DataFrame:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["ids", "raw_texts", "y_true", "yhat"]
    for k in required:
        if k not in data:
            raise ValueError(f"{fp} missing required key: {k}")

    ids = data["ids"]
    raws = data["raw_texts"]
    y_true = data["y_true"]
    yhat = data["yhat"]

    n = len(ids)
    if not (len(raws) == len(y_true) == len(yhat) == n):
        raise ValueError(
            f"{fp} length mismatch: ids={len(ids)}, raw_texts={len(raws)}, "
            f"y_true={len(y_true)}, yhat={len(yhat)}"
        )

    folder_name = Path(fp).parent.name
    method = normalise_method_name(folder_name)
    dataset = infer_dataset_from_path(fp)
    audio_dim = infer_audio_dim_from_path_or_json(fp, data)

    rows = []
    for i in range(n):
        yt = float(y_true[i])
        yh = float(yhat[i])
        rows.append(
            {
                "id": str(ids[i]),
                "raw_text": "" if raws[i] is None else str(raws[i]),
                "y_true": yt,
                "yhat": yh,
                "error_abs": abs(yh - yt),
                "method": method,
                "source_file": fp,
                "folder_name": folder_name,
                "dataset": dataset,
                "audio_dim": audio_dim,
            }
        )

    df = pd.DataFrame(rows)

    # safety: drop NaNs / inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y_true", "yhat", "error_abs"]).reset_index(drop=True)
    return df


def combine_all_predictions(pred_files):
    if not pred_files:
        raise ValueError("No preds_test.json files found with the given filters.")

    dfs = []
    for fp in pred_files:
        try:
            df = load_preds_json(fp)
            dfs.append(df)
            print(f"[load] {fp} -> {df['method'].iloc[0]} ({len(df)} rows)")
        except Exception as e:
            print(f"[skip] Failed to load {fp}: {e}")

    if not dfs:
        raise ValueError("No valid prediction files could be loaded.")

    all_df = pd.concat(dfs, ignore_index=True)

    print("\n[summary] Loaded methods by dataset/audio_dim:")
    inv = (
        all_df[["dataset", "audio_dim", "method", "source_file"]]
        .drop_duplicates()
        .groupby(["dataset", "audio_dim"])["method"]
        .apply(list)
        .reset_index()
    )
    for _, r in inv.iterrows():
        print(f"  - {r['dataset']} A{r['audio_dim']}: {len(r['method'])} methods -> {r['method']}")

    return all_df


# ---------------------------------
# Feature construction for clustering
# ---------------------------------
def build_utterance_matrix_abs(group_df: pd.DataFrame, require_all_methods: bool = True):
    """
    Returns:
      wide_df: one row per utterance with columns:
         id, raw_text, y_true, <method error cols>, yhat__<method>...
      X: np.ndarray (N, num_methods)  <-- ABS error matrix used for clustering
      methods: list[str]             <-- method names (error columns)
    """
    if "error_abs" not in group_df.columns:
        raise ValueError("Missing error_abs in group_df. Did you compute it in load_preds_json()?")

    meta = (
        group_df.sort_values(["id", "method"])
        .groupby("id", as_index=False)
        .agg(
            raw_text=("raw_text", "first"),
            y_true=("y_true", "first"),
            dataset=("dataset", "first"),
            audio_dim=("audio_dim", "first"),
        )
    )

    pivot_err = group_df.pivot_table(index="id", columns="method", values="error_abs", aggfunc="first")
    methods = list(pivot_err.columns)

    pivot_yhat = group_df.pivot_table(index="id", columns="method", values="yhat", aggfunc="first")
    pivot_yhat.columns = [f"yhat__{c}" for c in pivot_yhat.columns]

    if require_all_methods:
        before = len(pivot_err)
        pivot_err = pivot_err.dropna(axis=0, how="any")
        after = len(pivot_err)
        dropped = before - after
        if dropped > 0:
            print(f"[align] Dropped {dropped} utterances with missing method predictions (abs errors).")
        pivot_yhat = pivot_yhat.loc[pivot_err.index]
    else:
        pivot_err = pivot_err.copy()
        for c in pivot_err.columns:
            pivot_err[c] = pivot_err[c].fillna(pivot_err[c].median())
        pivot_yhat = pivot_yhat.loc[pivot_err.index]

    wide = meta.merge(pivot_err.reset_index(), on="id", how="inner")
    wide = wide.merge(pivot_yhat.reset_index(), on="id", how="left")

    err_cols = [c for c in wide.columns if c not in {"id", "raw_text", "y_true", "dataset", "audio_dim"} and not c.startswith("yhat__")]
    X = wide[err_cols].to_numpy(dtype=np.float32)

    # safety: ensure finite
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf after alignment. Check input preds_test.json files.")

    return wide, X, err_cols


# ---------------------------------
# KMeans + elbow
# ---------------------------------
def standardize_X(X: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def elbow_curve(
    Xs: np.ndarray,
    out_dir: str,
    dataset: str,
    prefix: str,
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
    df.to_csv(os.path.join(out_dir, f"{prefix}_elbow.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plot_title = "Elbow curve (MOSEI)" if dataset == "mosei" else "Elbow curve (CHSIMS)"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_elbow.png"), dpi=180)
    plt.close()

    print(f"[elbow] Saved {prefix}_elbow.csv and {prefix}_elbow.png")
    return df


# ---------------------------------
# Summaries + outputs
# ---------------------------------
def cluster_summary_table(df_clustered: pd.DataFrame, err_cols: list[str]):
    rows = []
    for c in sorted(df_clustered["cluster"].unique()):
        sub = df_clustered[df_clustered["cluster"] == c].copy()

        err_mat = sub[err_cols].to_numpy(dtype=np.float32)
        best_idx = np.argmin(err_mat, axis=1)
        worst_idx = np.argmax(err_mat, axis=1)

        best_methods = [err_cols[i] for i in best_idx]
        worst_methods = [err_cols[i] for i in worst_idx]

        best_counts = Counter(best_methods)
        worst_counts = Counter(worst_methods)

        rows.append({
            "cluster": int(c),
            "size": int(len(sub)),
            "pct": float(len(sub) / len(df_clustered)),
            "y_true_mean": float(sub["y_true"].mean()),
            "y_true_std": float(sub["y_true"].std(ddof=0)),
            "most_common_best_method": best_counts.most_common(1)[0][0] if best_counts else None,
            "most_common_best_method_count": best_counts.most_common(1)[0][1] if best_counts else 0,
            "most_common_worst_method": worst_counts.most_common(1)[0][0] if worst_counts else None,
            "most_common_worst_method_count": worst_counts.most_common(1)[0][1] if worst_counts else 0,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def save_cluster_artifacts(
    df_wide: pd.DataFrame,
    err_cols: list[str],
    labels: np.ndarray,
    group_out_dir: str,
    prefix: str = "abs",
):
    os.makedirs(group_out_dir, exist_ok=True)

    df = df_wide.copy()
    df["cluster"] = labels

    # Summary
    summary_df = cluster_summary_table(df, err_cols)
    summary_df.to_csv(os.path.join(group_out_dir, f"{prefix}_cluster_summary.csv"),
                      index=False, encoding="utf-8-sig")

    # Full table
    yhat_cols = [c for c in df.columns if c.startswith("yhat__")]

    full_cols = ["id", "raw_text", "y_true", "cluster"] + yhat_cols + err_cols
    df.sort_values(["cluster", "id"]).to_csv(
        os.path.join(group_out_dir, f"{prefix}_clustered_utterances.csv"),
        index=False,
        columns=full_cols,
        encoding="utf-8-sig"
    )

    # Per-cluster clip list
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c].copy()
        clip_cols = ["id", "raw_text", "y_true", "cluster"] + yhat_cols
        sub.sort_values("id").to_csv(
            os.path.join(group_out_dir, f"{prefix}_cluster{c}_clips_with_predictions.csv"),
            index=False,
            columns=clip_cols,
            encoding="utf-8-sig"
        )

    return df, summary_df


# ---------------------------------
# Plotting (PCA + t-SNE)
# ---------------------------------
def _safe_perplexity(n_samples: int) -> int:
    if n_samples <= 5:
        return 2
    return int(min(30, max(5, (n_samples - 1) // 3)))


def plot_cluster_scatter(Xs: np.ndarray, cluster_labels: np.ndarray, out_dir: str, prefix: str = "", title_suffix: str = "", try_tsne: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    n = Xs.shape[0]
    if n < 2:
        print("[plot] Skipping scatter plots (need at least 2 samples).")
        return

    # PCA scatter
    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(Xs)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=12, alpha=0.8)
        plt.title(title_suffix)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(sc, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_clusters_pca.png"), dpi=180)
        plt.close()

        # explained variance ratio
        plt.figure(figsize=(6, 4))
        evr = pca.explained_variance_ratio_
        plt.bar([1, 2], evr)
        plt.xticks([1, 2], ["PC1", "PC2"])
        plt.ylabel("Explained variance ratio")
        plt.title(title_suffix)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_pca_variance.png"), dpi=180)
        plt.close()

    except Exception as e:
        print(f"[plot] PCA plotting failed: {e}")

    # t-SNE scatter (visualisation only; clusters are still KMeans clusters)
    if try_tsne and n >= 10:
        try:
            perp = _safe_perplexity(n)
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                random_state=42,
                n_iter=1000,
            )
            X_tsne = tsne.fit_transform(Xs)

            plt.figure(figsize=(8, 6))
            sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, s=12, alpha=0.8)
            plt.title(f"t-SNE cluster scatter ({prefix}) {title_suffix}".strip())
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(sc, label="Cluster")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{prefix}_clusters_tsne.png"), dpi=180)
            plt.close()
        except Exception as e:
            print(f"[plot] t-SNE plotting failed: {e}")
    else:
        print(f"[plot] Skipping t-SNE (n={n} too small or disabled).")


# ---------------------------------
# Main workflow
# ---------------------------------
def analyse_group_abs(
    group_df: pd.DataFrame,
    group_key: tuple,
    out_root: str,
    n_clusters: int,
    require_all_methods: bool,
    standardize_for_kmeans: bool,
    elbow_k_max: int,
):
    dataset, audio_dim = group_key
    group_name = f"{dataset}_A35"
    group_out_dir = os.path.join(out_root, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    print(f"\n[group] {group_name} | rows={len(group_df)}")
    print(f"[group] methods: {sorted(group_df['method'].unique().tolist())}")

    wide_df, X, err_cols = build_utterance_matrix_abs(group_df, require_all_methods=require_all_methods)
    print(f"  [matrix] X shape = {X.shape} | methods={len(err_cols)}")

    if X.shape[0] < 3:
        print("  [skip] Too few utterances for clustering.")
        return

    # Standardize (recommended)
    if standardize_for_kmeans:
        Xs, _ = standardize_X(X)
    else:
        Xs = X

    # (NEW) Elbow curve for justification
    elbow_curve(Xs, out_dir=group_out_dir, dataset=dataset, prefix="abs", k_min=2, k_max=elbow_k_max)

    # choose k safely
    k = int(n_clusters)
    if k < 2:
        k = 2
    if k >= Xs.shape[0]:
        k = max(2, Xs.shape[0] - 1)

    labels, kmeans = run_kmeans(Xs, n_clusters=k, seed=42)

    # Save artifacts
    df_clustered, summary_df = save_cluster_artifacts(
        df_wide=wide_df,
        err_cols=err_cols,
        labels=labels,
        group_out_dir=group_out_dir,
        prefix="abs",
    )

    # Method-level stats
    method_stats = []
    for m in err_cols:
        vals = df_clustered[m].to_numpy(dtype=np.float32)
        method_stats.append(
            {
                "method": m,
                "abs_error_mean": float(np.mean(vals)),
                "abs_error_std": float(np.std(vals)),
                "abs_error_median": float(np.median(vals)),
            }
        )
    pd.DataFrame(method_stats).sort_values("abs_error_mean").to_csv(
        os.path.join(group_out_dir, "abs_method_error_stats.csv"),
        index=False,
    )

    # Centroids in standardized space + (optionally) original space
    centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=err_cols)
    centroid_df.insert(0, "cluster", np.arange(len(centroid_df)))
    centroid_df.to_csv(os.path.join(group_out_dir, "abs_cluster_centroids_standardized.csv"), index=False)

    plot_title = "PCA Cluster Visualisation (MOSEI)" if dataset == "mosei" else "PCA Cluster Visualisation (CHSIMS)"
    # Scatter plots use the same Xs used for clustering
    plot_cluster_scatter(
        Xs=Xs,
        cluster_labels=labels,
        out_dir=group_out_dir,
        title_suffix=plot_title,
        try_tsne=True,
    )

    print(f"  [done] Saved outputs to {group_out_dir}")
    print(f"  [clusters] sizes:\n{summary_df[['cluster', 'size', 'pct']].to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description="ABS-ONLY: cluster utterances by multimodal model absolute error patterns.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory to scan (e.g. /hpctmp/scratch/e0968015 or .../FusionRuns)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for clustering artefacts")
    parser.add_argument("--run-suffix", type=str, default="run3",
                        help="Only include folders ending with _<run-suffix> (default: run3)")
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mosei", "chsims"],
                        help="Optional dataset filter")
    parser.add_argument("--audio-dim", type=int, default=None, choices=[16, 35],
                        help="Optional audio dim filter (best-effort from path naming)")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="K for KMeans (your chosen final K)")
    parser.add_argument("--elbow-k-max", type=int, default=12,
                        help="Max K to plot in elbow curve (min is 2). Will be capped at N-1.")
    parser.add_argument("--allow-missing-methods", action="store_true",
                        help="If set, keep utterances with missing methods (median-fill); else drop incomplete rows")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Disable feature standardization before KMeans (not recommended)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[scan] Finding preds_test.json files...")
    pred_files = find_prediction_jsons(
        root=args.root,
        run_suffix=args.run_suffix,
        dataset_filter=args.dataset,
        audio_dim_filter=args.audio_dim,
    )
    print(f"[scan] Found {len(pred_files)} files")
    for p in pred_files[:50]:
        print(f"  - {p}")
    if len(pred_files) > 50:
        print("  ...")

    if not pred_files:
        raise SystemExit("No prediction JSONs found. Check --root / --run-suffix / filters.")

    all_df = combine_all_predictions(pred_files)

    # Stronger filtering after load
    if args.dataset is not None:
        all_df = all_df[all_df["dataset"].str.lower() == args.dataset.lower()].copy()

    if args.audio_dim is not None:
        target = str(args.audio_dim)
        if (all_df["audio_dim"] == target).any():
            all_df = all_df[all_df["audio_dim"] == target].copy()
        else:
            print(f"[warn] No files explicitly inferred as audio_dim={target}; keeping current filtered set.")

    if all_df.empty:
        raise SystemExit("No rows remain after filtering.")

    grouped = all_df.groupby(["dataset", "audio_dim"], dropna=False)

    for group_key, group_df in grouped:
        analyse_group_abs(
            group_df=group_df.copy(),
            group_key=group_key,
            out_root=args.out_dir,
            n_clusters=args.n_clusters,
            require_all_methods=not args.allow_missing_methods,
            standardize_for_kmeans=not args.no_standardize,
            elbow_k_max=args.elbow_k_max,
        )

    print("\n[done] ABS clustering analysis complete.")


if __name__ == "__main__":
    main()
