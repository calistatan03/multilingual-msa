#!/usr/bin/env python3
"""
Cluster utterances by model error patterns from saved preds_test.json files.

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

This script:
1) Scans experiment folders (e.g. kernel_fusion_run3, text_guided_attn_run3, ...)
2) Loads predictions across methods
3) Aligns utterances by id
4) Builds error matrix per utterance across methods
5) Runs KMeans clustering (signed and/or absolute errors)
6) Saves cluster summaries / CSVs
7) Plots PCA and t-SNE scatterplots of clusters

Works even if you currently only have one dataset group (e.g. MOSEI A35).

Usage examples:
    python cluster_utterance_errors.py \
      --root /hpctmp/scratch/e0968015 \
      --dataset mosei \
      --audio-dim 35 \
      --out-dir /hpctmp/scratch/e0968015/analysis_clusters \
      --run-suffix run3

    python cluster_utterance_errors.py \
      --root /hpctmp/scratch/e0968015/mosei/FusionRuns \
      --out-dir /hpctmp/scratch/e0968015/mosei/FusionRuns/cluster_analysis \
      --run-suffix run3 \
      --mode both
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter

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
    # folder prefix -> pretty label
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
    if "/chsims/" in p or "chsims" in p or "chsims" in p:
        return "chsims"
    return "unknown"


def infer_audio_dim_from_path_or_json(path_str: str, data: dict) -> str:
    """
    Best-effort inference. If not obvious, returns 'unknown'.
    """
    p = path_str.lower()

    # Common naming patterns you used:
    # ... audio_guided_attn_35_fusion_run1 ...
    # ... vision_guided_attn_16_fusion_run1 ...
    m = re.search(r'[_\-](16|35)[_\-]', p)
    if m:
        return m.group(1)

    # If path contains explicit audio dim token like a35 / audio35
    m = re.search(r'\ba(16|35)\b', p)
    if m:
        return m.group(1)
    m = re.search(r'audio[_\-]?(16|35)\b', p)
    if m:
        return m.group(1)

    # Could also infer from exp_name if present
    exp_name = str(data.get("exp_name", "")).lower()
    m = re.search(r'[_\-](16|35)[_\-]', exp_name)
    if m:
        return m.group(1)

    return "35"


def normalise_method_name(folder_name: str) -> str:
    """
    Example:
      kernel_fusion_run3 -> Kernel Fusion
      text_guided_attn_fusion_run1 -> Text-Guided Attention (best effort)
    """
    name = folder_name.lower()

    # strip trailing _runX
    name = re.sub(r"_run\d+$", "", name)

    # best alias match by longest prefix/key contained
    candidates = []
    for k, v in METHOD_ALIASES.items():
        if name == k or name.startswith(k) or k in name:
            candidates.append((len(k), v))
    if candidates:
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

    # fallback prettify
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

        # require folder name to contain _runX suffix if requested
        if run_suffix:
            if not parent.endswith(f"_{run_suffix.lower()}"):
                continue

        # lightweight dataset filter using path
        if dataset_filter:
            ds_guess = infer_dataset_from_path(str(p))
            if ds_guess != dataset_filter.lower():
                continue

        # audio dim filter (best-effort) by folder path string if explicit
        if audio_dim_filter is not None:
            path_l = str(p).lower()
            # if explicit 16/35 token present and mismatched -> skip
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
def load_preds_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # FIX: yhat is required for error_abs / error_signed
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
                # FIX: compute error columns used by clustering
                "error_signed": yh - yt,
                "error_abs": abs(yh - yt),
                "method": method,
                "source_file": fp,
                "folder_name": folder_name,
                "dataset": dataset,
                "audio_dim": audio_dim,
            }
        )

    return pd.DataFrame(rows)

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

    # show quick inventory
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
def build_utterance_matrix(group_df: pd.DataFrame, mode: str = "abs", require_all_methods: bool = True):
    """
    Returns:
      wide_df: one row per utterance with columns:
         id, raw_text, y_true, <method error cols>, yhat__<method>...
      X: np.ndarray (N, num_methods)  <-- error matrix used for clustering
      methods: list[str]             <-- method names (error columns)
    """
    if mode not in {"abs", "signed"}:
        raise ValueError("mode must be 'abs' or 'signed'")

    val_col = "error_abs" if mode == "abs" else "error_signed"
    if val_col not in group_df.columns:
        raise ValueError(f"Missing {val_col} in group_df. Did you compute errors in load_preds_json?")

    # meta: one row per id
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

    # Error pivot (used for clustering)
    pivot_err = group_df.pivot_table(index="id", columns="method", values=val_col, aggfunc="first")
    methods = list(pivot_err.columns)

    # yhat pivot (for reporting)
    pivot_yhat = group_df.pivot_table(index="id", columns="method", values="yhat", aggfunc="first")
    pivot_yhat.columns = [f"yhat__{c}" for c in pivot_yhat.columns]

    # alignment policy (based on error pivot)
    if require_all_methods:
        before = len(pivot_err)
        pivot_err = pivot_err.dropna(axis=0, how="any")
        after = len(pivot_err)
        dropped = before - after
        if dropped > 0:
            print(f"[align] Dropped {dropped} utterances with missing method predictions (mode={mode})")

        pivot_yhat = pivot_yhat.loc[pivot_err.index]
    else:
        pivot_err = pivot_err.copy()
        for c in pivot_err.columns:
            pivot_err[c] = pivot_err[c].fillna(pivot_err[c].median())
        pivot_yhat = pivot_yhat.loc[pivot_err.index]

    wide = meta.merge(pivot_err.reset_index(), on="id", how="inner")
    wide = wide.merge(pivot_yhat.reset_index(), on="id", how="left")

    # error columns for X
    methods = [c for c in wide.columns if c not in {"id", "raw_text", "y_true", "dataset", "audio_dim"} and not c.startswith("yhat__")]

    X = wide[methods].to_numpy(dtype=np.float32)
    return wide, X, methods

def add_derived_columns(wide_df: pd.DataFrame, methods: list[str]):
    out = wide_df.copy()
    err_mat = out[methods].to_numpy(dtype=np.float32)

    # These columns are only meaningful if using abs errors.
    # But we'll compute generic ones from whatever matrix is passed.
    out["feature_mean"] = np.mean(err_mat, axis=1)
    out["feature_std"] = np.std(err_mat, axis=1)
    out["feature_min"] = np.min(err_mat, axis=1)
    out["feature_max"] = np.max(err_mat, axis=1)
    out["feature_range"] = out["feature_max"] - out["feature_min"]

    # "hardness" and disagreement heuristics
    out["hardness_score"] = out["feature_mean"]
    out["model_disagreement"] = out["feature_std"]

    # best / worst model on that utterance (based on current feature values)
    best_idx = np.argmin(err_mat, axis=1)
    worst_idx = np.argmax(err_mat, axis=1)
    out["best_method_on_utterance"] = [methods[i] for i in best_idx]
    out["worst_method_on_utterance"] = [methods[i] for i in worst_idx]

    return out


# ---------------------------------
# Clustering + summaries
# ---------------------------------
def run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42, standardize: bool = True):
    X_used = X.copy()
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X_used)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(X_used)
    return labels, kmeans, scaler, X_used

def cluster_summary_table(df_with_clusters: pd.DataFrame, methods: list[str]):
    rows = []
    for c in sorted(df_with_clusters["cluster"].unique()):
        sub = df_with_clusters[df_with_clusters["cluster"] == c].copy()

        # determine best/worst method per utterance by min/max error across methods
        err_mat = sub[methods].to_numpy(dtype=np.float32)
        best_idx = np.argmin(err_mat, axis=1)
        worst_idx = np.argmax(err_mat, axis=1)
        best_methods = [methods[i] for i in best_idx]
        worst_methods = [methods[i] for i in worst_idx]

        best_counts = Counter(best_methods)
        worst_counts = Counter(worst_methods)

        rows.append({
            "cluster": int(c),
            "size": int(len(sub)),
            "pct": float(len(sub) / len(df_with_clusters)),
            "y_true_mean": float(sub["y_true"].mean()),
            "y_true_std": float(sub["y_true"].std(ddof=0)),
            "most_common_best_method": best_counts.most_common(1)[0][0] if best_counts else None,
            "most_common_best_method_count": best_counts.most_common(1)[0][1] if best_counts else 0,
            "most_common_worst_method": worst_counts.most_common(1)[0][0] if worst_counts else None,
            "most_common_worst_method_count": worst_counts.most_common(1)[0][1] if worst_counts else 0,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

def save_cluster_artifacts(
    df_clustered: pd.DataFrame,
    methods: list[str],
    labels: np.ndarray,
    group_out_dir: str,
    prefix: str,
):
    os.makedirs(group_out_dir, exist_ok=True)

    df_clustered = df_clustered.copy()
    df_clustered["cluster"] = labels

    # Summary table
    summary_df = cluster_summary_table(df_clustered, methods)
    summary_path = os.path.join(group_out_dir, f"{prefix}_cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Full clustered table (keep error cols + yhat cols)
    yhat_cols = [c for c in df_clustered.columns if c.startswith("yhat__")]

    full_cols = (
        ["id", "raw_text", "y_true", "cluster"]
        + yhat_cols
        + methods  # these are error columns
    )
    full_cols = [c for c in full_cols if c in df_clustered.columns]
    full_path = os.path.join(group_out_dir, f"{prefix}_clustered_utterances.csv")
    df_clustered.sort_values(["cluster", "id"]).to_csv(
        full_path, index=False, columns=full_cols, encoding="utf-8-sig"
    )

    # NEW: per-cluster “clip list” CSV (all utterances in cluster)
    for c in sorted(df_clustered["cluster"].unique()):
        sub = df_clustered[df_clustered["cluster"] == c].copy()

        clip_cols = ["id", "raw_text", "y_true", "cluster"] + yhat_cols
        clip_cols = [x for x in clip_cols if x in sub.columns]

        out_path = os.path.join(group_out_dir, f"{prefix}_cluster{c}_clips_with_predictions.csv")
        sub.sort_values("id").to_csv(out_path, index=False, columns=clip_cols, encoding="utf-8-sig")

    return df_clustered, summary_df


# ---------------------------------
# Plotting
# ---------------------------------
def _safe_perplexity(n_samples: int) -> int:
    # TSNE requires perplexity < n_samples, typically > 1
    if n_samples <= 5:
        return 2
    # choose something conservative
    return int(min(30, max(5, (n_samples - 1) // 3)))


def plot_cluster_scatter(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: str,
    prefix: str,
    title_suffix: str = "",
    try_tsne: bool = True,
):
    """
    Creates PCA scatter and (optionally) t-SNE scatter.
    X should be (N, D), preferably already standardized or suitable for projection.
    """
    os.makedirs(out_dir, exist_ok=True)

    n = X.shape[0]
    if n < 2:
        print("[plot] Skipping scatter plots (need at least 2 samples).")
        return

    # PCA
    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=12, alpha=0.8)
        plt.title(f"PCA cluster scatter ({prefix}) {title_suffix}".strip())
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(sc, label="Cluster")
        plt.tight_layout()
        pca_path = os.path.join(out_dir, f"{prefix}_clusters_pca.png")
        plt.savefig(pca_path, dpi=180)
        plt.close()

        # variance plot (optional, useful)
        plt.figure(figsize=(6, 4))
        evr = pca.explained_variance_ratio_
        plt.bar([1, 2], evr)
        plt.xticks([1, 2], ["PC1", "PC2"])
        plt.ylabel("Explained variance ratio")
        plt.title(f"PCA variance ({prefix}) {title_suffix}".strip())
        plt.tight_layout()
        evr_path = os.path.join(out_dir, f"{prefix}_pca_variance.png")
        plt.savefig(evr_path, dpi=180)
        plt.close()

    except Exception as e:
        print(f"[plot] PCA plotting failed: {e}")

    # t-SNE
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
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(8, 6))
            sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, s=12, alpha=0.8)
            plt.title(f"t-SNE cluster scatter ({prefix}) {title_suffix}".strip())
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(sc, label="Cluster")
            plt.tight_layout()
            tsne_path = os.path.join(out_dir, f"{prefix}_clusters_tsne.png")
            plt.savefig(tsne_path, dpi=180)
            plt.close()
        except Exception as e:
            print(f"[plot] t-SNE plotting failed: {e}")
    else:
        print(f"[plot] Skipping t-SNE (n={n} too small or disabled).")


# ---------------------------------
# Main workflow
# ---------------------------------
def analyse_group(
    group_df: pd.DataFrame,
    group_key: tuple,
    out_root: str,
    n_clusters: int,
    mode: str,
    require_all_methods: bool,
    standardize_for_kmeans: bool,
):
    dataset, audio_dim = group_key
    group_name = f"{dataset}_A{audio_dim}"
    group_out_dir = os.path.join(out_root, group_name)
    os.makedirs(group_out_dir, exist_ok=True)

    print(f"\n[group] {group_name} | rows={len(group_df)}")
    print(f"[group] methods: {sorted(group_df['method'].unique().tolist())}")

    modes = ["abs", "signed"] if mode == "both" else [mode]

    for cluster_mode in modes:
        print(f"\n  [mode] {cluster_mode}")

        wide_df, X, methods = build_utterance_matrix(
            group_df, mode=cluster_mode, require_all_methods=require_all_methods
        )
        print('wide_df keys:', wide_df.keys())
        print(f"  [matrix] X shape = {X.shape} | methods={len(methods)}")
        if X.shape[0] == 0:
            print("  [skip] No aligned utterances for this group/mode.")
            continue

        # choose n_clusters safely
        k = min(n_clusters, max(2, X.shape[0] // 10)) if X.shape[0] < n_clusters else n_clusters
        if X.shape[0] < 3:
            print("  [skip] Too few utterances for clustering.")
            continue
        if k < 2:
            k = 2
        if k >= X.shape[0]:
            k = max(2, X.shape[0] - 1)

        labels, kmeans, scaler, X_used = run_kmeans(
            X, n_clusters=k, seed=42, standardize=standardize_for_kmeans
        )

        # Save artifacts
        df_clustered, summary_df = save_cluster_artifacts(
            df_clustered=wide_df,
            methods=methods,
            labels=labels,
            group_out_dir=group_out_dir,
            prefix=cluster_mode,
        )    

        # Save method-level aggregate stats (for this aligned subset)
        method_stats = []
        for m in methods:
            vals = wide_df[m].to_numpy(dtype=np.float32)
            method_stats.append(
                {
                    "method": m,
                    f"{cluster_mode}_feature_mean": float(np.mean(vals)),
                    f"{cluster_mode}_feature_std": float(np.std(vals)),
                    f"{cluster_mode}_feature_median": float(np.median(vals)),
                }
            )
        pd.DataFrame(method_stats).sort_values(f"{cluster_mode}_feature_mean").to_csv(
            os.path.join(group_out_dir, f"{cluster_mode}_method_feature_stats.csv"),
            index=False,
        )

        # Save centroid matrix in original feature space if standardized
        if scaler is not None:
            centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            centroids_orig = kmeans.cluster_centers_

        centroid_df = pd.DataFrame(centroids_orig, columns=methods)
        centroid_df.insert(0, "cluster", np.arange(len(centroid_df)))
        centroid_df.to_csv(os.path.join(group_out_dir, f"{cluster_mode}_cluster_centroids.csv"), index=False)

        # Plot scatter using the same matrix used for clustering (standardized if applicable)
        plot_cluster_scatter(
            X=X_used,
            cluster_labels=labels,
            out_dir=group_out_dir,
            prefix=cluster_mode,
            title_suffix=group_name,
            try_tsne=True,
        )

    
        print(f"  [done] Saved outputs to {group_out_dir}")
        print(f"  [clusters] sizes:\n{summary_df[['cluster', 'size', 'pct']].to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description="Cluster utterances by multimodal model error patterns.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory to scan. Can be /hpctmp/scratch/e0968015 or .../mosei/FusionRuns")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for clustering artefacts")
    parser.add_argument("--run-suffix", type=str, default="run3",
                        help="Only include folders ending with _<run-suffix>, e.g. run3")
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mosei", "chsims"],
                        help="Optional dataset filter")
    parser.add_argument("--audio-dim", type=int, default=None, choices=[16, 35],
                        help="Optional audio dim filter (best-effort from path naming)")
    parser.add_argument("--mode", type=str, default="both", choices=["abs", "signed", "both"],
                        help="Cluster on absolute errors, signed errors, or both")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="K for KMeans (adjusted automatically if too large)")
    parser.add_argument("--allow-missing-methods", action="store_true",
                        help="If set, keep utterances with missing methods (median-fill); else drop incomplete rows")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Disable feature standardization before KMeans")
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
    for p in pred_files:
        print(f"  - {p}")

    if not pred_files:
        raise SystemExit("No prediction JSONs found. Check --root / --run-suffix / filters.")

    all_df = combine_all_predictions(pred_files)

    # Apply stronger filtering after load (audio dim may only be inferable after opening JSON)
    if args.dataset is not None:
        all_df = all_df[all_df["dataset"].str.lower() == args.dataset.lower()].copy()
    if args.audio_dim is not None:
        # keep only exact matches if inferred; if unknown exists, drop unless no exact labels exist
        target = str(args.audio_dim)
        if (all_df["audio_dim"] == target).any():
            all_df = all_df[all_df["audio_dim"] == target].copy()
        else:
            print(f"[warn] No files explicitly inferred as audio_dim={target}; keeping current filtered set.")

    if all_df.empty:
        raise SystemExit("No rows remain after filtering.")

    # Group by dataset/audio_dim (works with one group too)
    grouped = all_df.groupby(["dataset", "audio_dim"], dropna=False)

    for group_key, group_df in grouped:
        analyse_group(
            group_df=group_df.copy(),
            group_key=group_key,
            out_root=args.out_dir,
            n_clusters=args.n_clusters,
            mode=args.mode,
            require_all_methods=not args.allow_missing_methods,
            standardize_for_kmeans=not args.no_standardize,
        )

    print("\n[done] Clustering analysis complete.")


if __name__ == "__main__":
    main()
