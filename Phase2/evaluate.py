#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

np.set_printoptions(precision=6, suppress=True)
THIS_DIR = Path(__file__).resolve().parent
RUNS_ROOT_DEFAULT = THIS_DIR / "runs"
AUC_FIG_DIR_DEFAULT = THIS_DIR / "AUC_figs"


class SimpleMetrics:
    @staticmethod
    def _binary_prf_for_pos_label(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: int,
        zero_division: int = 0,
    ) -> Tuple[float, float, float]:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        pos = int(pos_label)
        true_pos_mask = (y_true == pos)
        pred_pos_mask = (y_pred == pos)

        tp = float(np.sum(true_pos_mask & pred_pos_mask))
        fp = float(np.sum((~true_pos_mask) & pred_pos_mask))
        fn = float(np.sum(true_pos_mask & (~pred_pos_mask)))

        z = float(zero_division)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else z
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else z
        denom = precision + recall
        f1 = (2.0 * precision * recall / denom) if denom > 0 else z
        return float(precision), float(recall), float(f1)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int] | np.ndarray) -> np.ndarray:
        labels_arr = np.asarray(labels, dtype=np.int64)
        m = np.zeros((labels_arr.size, labels_arr.size), dtype=np.int64)
        label_to_idx = {int(v): i for i, v in enumerate(labels_arr.tolist())}
        for yt, yp in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
            i = label_to_idx.get(int(yt))
            j = label_to_idx.get(int(yp))
            if i is not None and j is not None:
                m[i, j] += 1
        return m

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        if y_true.size == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def _per_class_prf(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: np.ndarray,
        zero_division: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cm = SimpleMetrics.confusion_matrix(y_true, y_pred, labels)
        tp = np.diag(cm).astype(np.float64)
        support = np.sum(cm, axis=1).astype(np.float64)
        pred_count = np.sum(cm, axis=0).astype(np.float64)
        z = float(zero_division)
        precision = np.divide(tp, pred_count, out=np.full_like(tp, z), where=pred_count > 0)
        recall = np.divide(tp, support, out=np.full_like(tp, z), where=support > 0)
        denom = precision + recall
        f1 = np.divide(2.0 * precision * recall, denom, out=np.full_like(tp, z), where=denom > 0)
        return precision, recall, f1, support

    @staticmethod
    def precision_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: int = 1,
        average: str | None = None,
        zero_division: int = 0,
    ) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        if average is None:
            precision, _, _ = SimpleMetrics._binary_prf_for_pos_label(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                zero_division=zero_division,
            )
            return precision
        labels = np.unique(np.concatenate([y_true, y_pred])).astype(np.int64)
        precision, _, _, support = SimpleMetrics._per_class_prf(y_true, y_pred, labels, zero_division=zero_division)
        if average == "macro":
            return float(np.mean(precision)) if precision.size > 0 else 0.0
        if average == "weighted":
            wsum = float(np.sum(support))
            return float(np.sum(precision * support) / wsum) if wsum > 0 else 0.0
        raise ValueError(f"Unsupported average: {average}")

    @staticmethod
    def recall_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: int = 1,
        average: str | None = None,
        zero_division: int = 0,
    ) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        if average is None:
            _, recall, _ = SimpleMetrics._binary_prf_for_pos_label(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                zero_division=zero_division,
            )
            return recall
        labels = np.unique(np.concatenate([y_true, y_pred])).astype(np.int64)
        _, recall, _, support = SimpleMetrics._per_class_prf(y_true, y_pred, labels, zero_division=zero_division)
        if average == "macro":
            return float(np.mean(recall)) if recall.size > 0 else 0.0
        if average == "weighted":
            wsum = float(np.sum(support))
            return float(np.sum(recall * support) / wsum) if wsum > 0 else 0.0
        raise ValueError(f"Unsupported average: {average}")

    @staticmethod
    def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pos_label: int = 1,
        average: str | None = None,
        zero_division: int = 0,
    ) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        if average is None:
            _, _, f1 = SimpleMetrics._binary_prf_for_pos_label(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=pos_label,
                zero_division=zero_division,
            )
            return f1
        labels = np.unique(np.concatenate([y_true, y_pred])).astype(np.int64)
        _, _, f1, support = SimpleMetrics._per_class_prf(y_true, y_pred, labels, zero_division=zero_division)
        if average == "macro":
            return float(np.mean(f1)) if f1.size > 0 else 0.0
        if average == "weighted":
            wsum = float(np.sum(support))
            return float(np.sum(f1 * support) / wsum) if wsum > 0 else 0.0
        raise ValueError(f"Unsupported average: {average}")

    @staticmethod
    def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        labels = np.unique(np.concatenate([y_true.astype(np.int64), y_pred.astype(np.int64)])).astype(np.int64)
        _, recall, _, _ = SimpleMetrics._per_class_prf(y_true, y_pred, labels, zero_division=0)
        return float(np.mean(recall)) if recall.size > 0 else 0.0

    @staticmethod
    def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true = y_true.astype(np.int64)
        scores = scores.astype(np.float64)
        pos = (y_true == 1).astype(np.int64)
        neg = (y_true == 0).astype(np.int64)
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("roc_curve requires both positive and negative samples")

        order = np.argsort(-scores, kind="mergesort")
        y_sorted = pos[order]
        s_sorted = scores[order]

        distinct_idx = np.where(np.diff(s_sorted))[0]
        thresh_idx = np.r_[distinct_idx, y_sorted.size - 1]

        tps = np.cumsum(y_sorted)[thresh_idx].astype(np.float64)
        fps = (1 + thresh_idx - tps).astype(np.float64)

        tpr = np.r_[0.0, tps / float(n_pos)]
        fpr = np.r_[0.0, fps / float(n_neg)]
        thresholds = np.r_[np.inf, s_sorted[thresh_idx]]
        return fpr, tpr, thresholds

    @staticmethod
    def auc(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.trapz(y, x))


metrics = SimpleMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unknown_traffic_detect via union-of-subspaces")
    parser.add_argument("--work_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--features_dir", type=str, default="", help="default: <work_dir>/artifacts/features")
    parser.add_argument("--reports_dir", type=str, default="", help="default: <work_dir>/artifacts/reports")
    parser.add_argument("--feature_tag", type=str, default="internal_12", help="must match extract script")
    parser.add_argument("--subspace_dim", type=int, default=5)
    parser.add_argument("--train_known_quantile", type=float, default=0.95)
    parser.add_argument("--stage", type=str, default="Stage_3")
    parser.add_argument("--plot_auc", action="store_true")
    parser.add_argument("--auc_fig_dir", type=str, default=str(AUC_FIG_DIR_DEFAULT))
    parser.add_argument("--runs_root", type=str, default=str(RUNS_ROOT_DEFAULT))
    parser.add_argument("--font_family", type=str, default="Times New Roman")
    parser.add_argument("--save_scores", action="store_true")
    return parser.parse_args()


def ensure_dirs(work_dir: Path) -> Dict[str, Path]:
    artifacts = work_dir / "artifacts"
    paths = {
        "artifacts": artifacts,
        "features": artifacts / "features",
        "reports": artifacts / "reports",
        "logs": artifacts / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def preprocess(d: np.ndarray, labels: np.ndarray | None = None):
    if labels is None:
        return np.asarray(d)
    data = np.asarray(d)
    labels = np.asarray(labels)
    return [data[labels == l].transpose() for l in np.unique(labels)]


def calculate_left_singular_subspace(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"Expected subspace_dim > 0, got {k}")
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("Empty class matrix can not define a subspace")
    u, _, _ = np.linalg.svd(a, full_matrices=False)
    k_eff = min(k, u.shape[1])
    return np.ascontiguousarray(u[:, :k_eff])


def _compute_similarity_matrix(samples: np.ndarray, class_subspaces: List[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {samples.shape}")

    n_samples = samples.shape[0]
    n_classes = len(class_subspaces)
    sample_norm = np.linalg.norm(samples, axis=1) + eps

    sim = np.empty((n_samples, n_classes), dtype=np.float32)
    for class_idx, basis in enumerate(class_subspaces):
        proj = samples @ basis
        proj_norm = np.linalg.norm(proj, axis=1)
        sim[:, class_idx] = proj_norm / sample_norm

    return np.clip(sim, 0.0, 1.0)


def score_and_predict(d: np.ndarray, class_subspaces: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(d)
    n_classes = len(class_subspaces)
    if n_classes == 0:
        raise ValueError("No class subspaces were built from the training set")

    if d.ndim == 2:
        class_sim = _compute_similarity_matrix(np.ascontiguousarray(d), class_subspaces)
        best = np.max(class_sim, axis=1)
        pred_idx = np.argmax(class_sim, axis=1)
        score = np.arccos(np.clip(best, 0.0, 1.0))
        return score, pred_idx, class_sim

    if d.ndim == 3:
        _, _, mc_runs = d.shape
        sim_sum = None
        best = None

        for mc in range(mc_runs):
            class_sim_mc = _compute_similarity_matrix(np.ascontiguousarray(d[:, :, mc]), class_subspaces)
            if sim_sum is None:
                sim_sum = class_sim_mc.astype(np.float64, copy=True)
            else:
                sim_sum += class_sim_mc

            mc_best = np.max(class_sim_mc, axis=1)
            if best is None:
                best = mc_best
            else:
                np.maximum(best, mc_best, out=best)

        class_sim_mean = (sim_sum / float(mc_runs)).astype(np.float32, copy=False)
        pred_idx = np.argmax(class_sim_mean, axis=1)
        score = np.arccos(np.clip(best, 0.0, 1.0))
        return score, pred_idx, class_sim_mean

    raise ValueError(f"Expected d to be 2D or 3D, got shape {d.shape}")


def compute_ood_roc_metrics(score_in: np.ndarray, score_out: np.ndarray) -> Dict[str, float]:
    target_in = np.zeros_like(score_in)
    target_out = np.ones_like(score_out)
    targets = np.concatenate([target_in, target_out])
    scores = np.concatenate([score_in, score_out])
    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0
    det_error = float(np.min(0.5 * (1 - tpr) + 0.5 * fpr))
    auc = float(metrics.auc(fpr, tpr))
    return {
        "fpr95": fpr95,
        "detection_error": det_error,
        "auc": auc,
        "num_points": int(len(fpr)),
        "threshold_at_fpr95": float(thresholds[idx[0]]) if len(idx) > 0 else float("nan"),
    }


def parse_run_name(run_name: str) -> Dict[str, object] | None:
    # supported:
    # 1) tlscsv_deepresnet_{sym|asym}_{noise_pct}_cut{cut_pct}_{stage}
    # 2) tlscsv_deepresnet_{sym|asym}_{noise_pct}_PL_cut{cut_pct}_{stage}
    # 3) {sym|asym}_{noise_pct}_cut{cut_pct}_{Stage_1|Stage_2|Stage_3}
    # 4) {sym|asym}_{noise_pct}_PL_cut{cut_pct}_{Stage_2|Stage_3}
    parts = run_name.split("_")
    if len(parts) < 4:
        return None

    if len(parts) >= 6 and parts[0] == "tlscsv" and parts[1] == "deepresnet":
        noise_type = parts[2]
        noise_pct_s = parts[3]
        cut_idx = 4
        if parts[4] == "PL":
            cut_idx = 5
    else:
        noise_type = parts[0]
        noise_pct_s = parts[1]
        if len(parts) >= 4 and parts[2] == "Stage" and parts[3] == "1":
            return {
                "noise_type": noise_type,
                "noise_pct": int(noise_pct_s),
                "noise": float(int(noise_pct_s)) / 100.0,
                "cut_pct": 0,
                "cut": 0.0,
                "stage": "Stage_1",
            }
        cut_idx = 2
        if len(parts) >= 4 and parts[2] == "PL":
            cut_idx = 3

    if cut_idx >= len(parts):
        return None

    cut_token = parts[cut_idx]
    stage = "_".join(parts[cut_idx + 1 :]) if (cut_idx + 1) < len(parts) else ""
    if noise_type not in {"sym", "asym"}:
        return None
    if not noise_pct_s.isdigit():
        return None
    if not cut_token.startswith("cut") or not cut_token[3:].isdigit():
        return None
    noise_pct = int(noise_pct_s)
    cut_pct = int(cut_token[3:])
    return {
        "noise_type": noise_type,
        "noise_pct": noise_pct,
        "noise": float(noise_pct) / 100.0,
        "cut_pct": cut_pct,
        "cut": float(cut_pct) / 100.0,
        "stage": stage,
    }


def normalize_stage_label(stage: str) -> str:
    s = str(stage).strip()
    return {
        "Stage_1": "stage1",
        "Stage_2": "stage2",
        "Stage_3": "stage3",
        "stage1": "stage1",
        "stage2": "stage2",
        "stage3": "stage3",
        "nl": "stage1",
        "pl": "stage2",
        "pseudo1": "stage3",
    }.get(s, s)


def save_roc_curve_files(
    reports_dir: Path,
    dataset: str | None,
    run_name: str | None,
    stage: str,
    noise_type: str | None,
    noise: float | None,
    cut: float | None,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    auc: float,
) -> Tuple[Path, Path]:
    roc_json = reports_dir / "ood_roc_curve.json"
    roc_csv = reports_dir / "ood_roc_curve.csv"
    payload = {
        "dataset": dataset,
        "run_name": run_name,
        "stage": stage,
        "noise_type": noise_type,
        "noise": noise,
        "cut": cut,
        "auc": float(auc),
        "fpr": fpr.astype(np.float64).tolist(),
        "tpr": tpr.astype(np.float64).tolist(),
        "thresholds": thresholds.astype(np.float64).tolist(),
    }
    with roc_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    roc_arr = np.column_stack([fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)])
    np.savetxt(roc_csv, roc_arr, delimiter=",", header="fpr,tpr,threshold", comments="")
    return roc_json, roc_csv


def build_paper_style(font_family: str) -> None:
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
        }
    )


def collect_runs_for_dataset(
    runs_root: Path,
    dataset: str,
    stage: str,
    target_noise_pct: List[int],
) -> List[Dict[str, object]]:
    ds_dir = runs_root / dataset.lower()
    out: List[Dict[str, object]] = []
    if not ds_dir.exists():
        return out
    for run_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
        info = parse_run_name(run_dir.name)
        if info is None:
            continue
        if normalize_stage_label(str(info["stage"])) != normalize_stage_label(stage):
            continue
        if int(info["noise_pct"]) not in target_noise_pct:
            continue
        roc_path = run_dir / "artifacts" / "reports" / "ood_roc_curve.json"
        if not roc_path.exists():
            continue
        try:
            data = json.loads(roc_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fpr = np.asarray(data.get("fpr", []), dtype=np.float64)
        tpr = np.asarray(data.get("tpr", []), dtype=np.float64)
        if fpr.size == 0 or tpr.size == 0 or fpr.size != tpr.size:
            continue
        auc = float(data.get("auc", float("nan")))
        out.append(
            {
                "dataset": dataset,
                "run_name": run_dir.name,
                "noise_type": str(info["noise_type"]),
                "noise_pct": int(info["noise_pct"]),
                "noise": float(info["noise"]),
                "cut_pct": int(info["cut_pct"]),
                "cut": float(info["cut"]),
                "stage": stage,
                "auc": auc,
                "fpr": fpr,
                "tpr": tpr,
                "roc_json": str(roc_path),
            }
        )
    out.sort(key=lambda r: (int(r["noise_pct"]), str(r["noise_type"]), int(r["cut_pct"])))
    return out


def plot_dataset_auc_curves(
    rows: List[Dict[str, object]],
    dataset: str,
    stage: str,
    fig_dir: Path,
    font_family: str,
) -> Tuple[Path, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    build_paper_style(font_family)

    style_map = {
        ("sym", 10): {"color": "#1f77b4", "ls": "--"},
        ("sym", 30): {"color": "#2ca02c", "ls": "--"},
        ("sym", 50): {"color": "#17becf", "ls": "--"},
        ("asym", 10): {"color": "#ff7f0e", "ls": "--"},
        ("asym", 30): {"color": "#8c564b", "ls": "--"},
        ("asym", 50): {"color": "#9467bd", "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    # Pick one representative per scenario; prefer smaller CUT when multiple exist.
    selected: Dict[Tuple[str, int], Dict[str, object]] = {}
    for r in sorted(rows, key=lambda x: (str(x["noise_type"]), int(x["noise_pct"]), int(x["cut_pct"]))):
        key = (str(r["noise_type"]), int(r["noise_pct"]))
        if key not in selected:
            selected[key] = r

    legend_order = [
        ("sym", 10),
        ("sym", 30),
        ("sym", 50),
        ("asym", 10),
        ("asym", 30),
        ("asym", 50),
    ]
    for key in legend_order:
        if key not in selected:
            continue
        r = selected[key]
        st = style_map.get(key, {"color": None, "ls": "-"})
        noise_name = key[0].capitalize()
        label = f"{noise_name}-{key[1]}% (AUC={float(r['auc']) * 100.0:.2f}%)"
        ax.plot(
            np.asarray(r["fpr"], dtype=np.float64),
            np.asarray(r["tpr"], dtype=np.float64),
            linestyle=st["ls"],
            color=st["color"],
            linewidth=2.0,
            label=label,
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0, label="Random (AUC=50%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate", fontsize=18)
    ax.set_ylabel("True Positive Rate", fontsize=18)
    # ax.set_title(f"{dataset} ROC Curves ({stage})")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=False, fontsize=15)

    # Inset zoom area (upper-left ROC corner), placed at upper-right blank area.
    zoom_x0, zoom_x1 = 0.0, 0.10
    zoom_y0, zoom_y1 = 0.90, 1.0

    # Move inset to upper-right to avoid overlapping with the legend.
    inset = ax.inset_axes([0.35, 0.55, 0.3, 0.3])  # [left, bottom, width, height] in axes fraction
    inset.set_facecolor("white")
    inset.set_zorder(5)

    for key in legend_order:
        if key not in selected:
            continue
        r = selected[key]
        st = style_map.get(key, {"color": None, "ls": "-"})
        inset.plot(
            np.asarray(r["fpr"], dtype=np.float64),
            np.asarray(r["tpr"], dtype=np.float64),
            linestyle=st["ls"],
            color=st["color"],
            linewidth=1.5,
        )

    inset.set_xlim(zoom_x0, zoom_x1)
    inset.set_ylim(zoom_y0, zoom_y1)
    inset.grid(True, alpha=0.5)
    inset.tick_params(labelsize=8)

    # Main-plot highlight box.
    zoom_rect = Rectangle(
        (zoom_x0, zoom_y0),
        zoom_x1 - zoom_x0,
        zoom_y1 - zoom_y0,
        fill=False,
        edgecolor="#d95f5f",
        linewidth=1.1,
        alpha=0.9,
    )
    ax.add_patch(zoom_rect)

    # Arrow from highlighted area to inset.
    con = ConnectionPatch(
        xyA=(zoom_x1, zoom_y0 + 0.03),
        coordsA=ax.transData,
        xyB=(0.02, 0.08),
        coordsB=inset.transAxes,
        arrowstyle="->",
        lw=1.2,
        linestyle="--",
        color="#d95f5f",
        zorder=6,
    )
    fig.add_artist(con)

    fig.tight_layout()

    png_path = fig_dir / f"{dataset}_auc_curves_{stage}.png"
    pdf_path = fig_dir / f"{dataset}_auc_curves_{stage}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def aggregate_and_plot_auc_figures(args: argparse.Namespace) -> Dict[str, object]:
    runs_root = Path(args.runs_root).resolve()
    fig_dir = Path(args.auc_fig_dir).resolve()
    target_noise_pct = [10, 30, 50]
    datasets = ["D1", "D5"]

    all_index_rows: List[Dict[str, object]] = []
    generated: Dict[str, Dict[str, str]] = {}
    for ds in datasets:
        rows = collect_runs_for_dataset(runs_root=runs_root, dataset=ds, stage=args.stage, target_noise_pct=target_noise_pct)
        all_index_rows.extend(rows)
        if not rows:
            continue
        png_path, pdf_path = plot_dataset_auc_curves(rows=rows, dataset=ds, stage=args.stage, fig_dir=fig_dir, font_family=args.font_family)
        generated[ds] = {"png": str(png_path), "pdf": str(pdf_path)}

    fig_dir.mkdir(parents=True, exist_ok=True)
    index_csv = fig_dir / f"auc_curve_index_{args.stage}.csv"
    fields = ["dataset", "run_name", "noise_type", "noise_pct", "noise", "cut_pct", "cut", "stage", "auc", "roc_json"]
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_index_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    summary_json = fig_dir / f"auc_fig_summary_{args.stage}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_unix": time.time(),
                "runs_root": str(runs_root),
                "auc_fig_dir": str(fig_dir),
                "stage": str(args.stage),
                "datasets": datasets,
                "target_noise_pct": target_noise_pct,
                "num_curves_indexed": int(len(all_index_rows)),
                "generated_figures": generated,
                "index_csv": str(index_csv),
            },
            f,
            indent=2,
        )
    return {"index_csv": str(index_csv), "summary_json": str(summary_json), "generated": generated}


def compute_threshold_metrics(score_train: np.ndarray, score_in: np.ndarray, score_out: np.ndarray, known_quantile: float) -> Dict[str, float | list]:
    tau = float(np.quantile(score_train, known_quantile))
    pred_in_unknown = (score_in > tau).astype(np.int64)
    pred_out_unknown = (score_out > tau).astype(np.int64)

    y_true = np.concatenate([np.zeros_like(pred_in_unknown, dtype=np.int64), np.ones_like(pred_out_unknown, dtype=np.int64)])
    y_pred = np.concatenate([pred_in_unknown, pred_out_unknown])

    known_acc = float(np.mean(pred_in_unknown == 0))
    unknown_acc = float(np.mean(pred_out_unknown == 1))
    overall_acc = float(metrics.accuracy_score(y_true, y_pred))

    precision_unknown = float(metrics.precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall_unknown = float(metrics.recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_unknown = float(metrics.f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    precision_known = float(metrics.precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    recall_known = float(metrics.recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_known = float(metrics.f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    precision_weighted = float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_acc = overall_acc

    macro_f1 = float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(metrics.balanced_accuracy_score(y_true, y_pred))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold_tau": tau,
        "train_known_quantile": float(known_quantile),
        "known_acc": known_acc,
        "unknown_acc": unknown_acc,
        "overall_acc": overall_acc,
        "precision_unknown": precision_unknown,
        "recall_unknown": recall_unknown,
        "f1_unknown": f1_unknown,
        "precision_known": precision_known,
        "recall_known": recall_known,
        "f1_known": f1_known,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "weighted_acc": weighted_acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "confusion_matrix": cm.tolist(),
        "num_known_test": int(len(score_in)),
        "num_unknown_test": int(len(score_out)),
    }


def compute_known_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels: np.ndarray) -> Dict[str, float | list]:
    acc = float(metrics.accuracy_score(y_true, y_pred))
    weighted_acc = acc
    precision_weighted = float(metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(metrics.balanced_accuracy_score(y_true, y_pred))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_labels)
    return {
        "accuracy": acc,
        "weighted_acc": weighted_acc,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "num_samples": int(y_true.shape[0]),
        "class_labels": class_labels.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def compute_open_set_classification_metrics(
    y_known_true: np.ndarray,
    y_known_pred: np.ndarray,
    y_out_pred: np.ndarray,
    score_in: np.ndarray,
    score_out: np.ndarray,
    tau: float,
    class_labels: np.ndarray,
) -> Dict[str, float | list]:
    unknown_label = int(np.max(class_labels)) + 1
    pred_known_gate = np.where(score_in > tau, unknown_label, y_known_pred)
    pred_out_gate = np.where(score_out > tau, unknown_label, y_out_pred)

    y_true_open = np.concatenate([y_known_true, np.full_like(y_out_pred, unknown_label)])
    y_pred_open = np.concatenate([pred_known_gate, pred_out_gate])

    labels_open = np.concatenate([class_labels.astype(np.int64), np.array([unknown_label], dtype=np.int64)])
    cm_open = metrics.confusion_matrix(y_true_open, y_pred_open, labels=labels_open)

    known_accept_mask = score_in <= tau
    known_coverage = float(np.mean(known_accept_mask))
    known_acc_on_accepted = float(np.mean(y_known_pred[known_accept_mask] == y_known_true[known_accept_mask])) if np.any(known_accept_mask) else float("nan")
    open_set_accuracy = float(metrics.accuracy_score(y_true_open, y_pred_open))
    precision_weighted = float(metrics.precision_score(y_true_open, y_pred_open, average="weighted", zero_division=0))
    recall_weighted = float(metrics.recall_score(y_true_open, y_pred_open, average="weighted", zero_division=0))
    f1_weighted = float(metrics.f1_score(y_true_open, y_pred_open, average="weighted", zero_division=0))

    return {
        "threshold_tau": float(tau),
        "unknown_label": unknown_label,
        "open_set_accuracy": open_set_accuracy,
        "weighted_acc": open_set_accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "open_set_macro_f1": float(metrics.f1_score(y_true_open, y_pred_open, average="macro", zero_division=0)),
        "open_set_weighted_f1": float(metrics.f1_score(y_true_open, y_pred_open, average="weighted", zero_division=0)),
        "known_coverage": known_coverage,
        "known_acc_on_accepted": known_acc_on_accepted,
        "unknown_rejection_rate": float(np.mean(score_out > tau)),
        "labels_open": labels_open.astype(int).tolist(),
        "confusion_matrix_open": cm_open.tolist(),
    }


def build_known_weighted_report(known_metrics: Dict[str, float | list] | None) -> Dict[str, float] | None:
    if known_metrics is None:
        return None
    return {
        "weighted_acc": float(known_metrics["weighted_acc"]),
        "precision_weighted": float(known_metrics["precision_weighted"]),
        "recall_weighted": float(known_metrics["recall_weighted"]),
        "f1_weighted": float(known_metrics["f1_weighted"]),
    }


def build_ood_report(roc_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "auc": float(roc_metrics["auc"]),
    }


def build_unknown_detection_report(th_metrics: Dict[str, float | list]) -> Dict[str, float | list]:
    return {
        "threshold_tau": float(th_metrics["threshold_tau"]),
        "acc": float(th_metrics["overall_acc"]),
        "precision": float(th_metrics["precision_unknown"]),
        "recall": float(th_metrics["recall_unknown"]),
        "f1": float(th_metrics["f1_unknown"]),
        "confusion_matrix": th_metrics["confusion_matrix"],
        "num_known_test": int(th_metrics["num_known_test"]),
        "num_unknown_test": int(th_metrics["num_unknown_test"]),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_known_quantile < 1.0):
        raise ValueError(f"--train_known_quantile must be in (0,1), got {args.train_known_quantile}")

    paths = ensure_dirs(Path(args.work_dir).resolve())
    features_dir = Path(args.features_dir).resolve() if args.features_dir else paths["features"]
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    extract_meta_path = paths["logs"] / "extract_metadata.json"
    extract_meta = None
    if extract_meta_path.exists():
        with open(extract_meta_path, "r", encoding="utf-8") as f:
            extract_meta = json.load(f)

    f_train = features_dir / "featuresTrain_in.npy"
    y_train = features_dir / "labelsTrain_in.npy"
    f_test_in = features_dir / "featuresTest_in.npy"
    y_test_in = features_dir / "labelsTest_in.npy"
    f_test_out = features_dir / f"features_out_{args.feature_tag}.npy"

    for p in [f_train, y_train, f_test_in, f_test_out]:
        if not p.exists():
            raise FileNotFoundError(f"missing required feature file: {p}")

    t0 = time.time()

    train = np.squeeze(np.load(f_train, allow_pickle=False))
    labels = np.squeeze(np.load(y_train, allow_pickle=False)).astype(np.int64)
    test_in = np.squeeze(np.load(f_test_in, allow_pickle=False))
    test_out = np.squeeze(np.load(f_test_out, allow_pickle=False))

    if train.ndim != 2:
        raise ValueError(f"featuresTrain_in must be 2D, got {train.shape}")
    if test_in.shape[1] != train.shape[1]:
        raise ValueError(f"feature dim mismatch test_in={test_in.shape[1]} train={train.shape[1]}")
    if test_out.shape[1] != train.shape[1]:
        raise ValueError(f"feature dim mismatch test_out={test_out.shape[1]} train={train.shape[1]}")

    class_labels = np.unique(labels)
    grouped_train = preprocess(train, labels)
    valid_idx = [i for i, g in enumerate(grouped_train) if g.size > 0]
    grouped_train = [grouped_train[i] for i in valid_idx]
    class_labels = class_labels[valid_idx]

    class_subspaces = [calculate_left_singular_subspace(g, args.subspace_dim) for g in grouped_train]

    score_train, pred_train_idx, _ = score_and_predict(train, class_subspaces)
    score_in, pred_in_idx, _ = score_and_predict(test_in, class_subspaces)
    score_out, pred_out_idx, _ = score_and_predict(test_out, class_subspaces)

    pred_train = class_labels[pred_train_idx]
    pred_in = class_labels[pred_in_idx]
    pred_out = class_labels[pred_out_idx]

    target_in = np.zeros_like(score_in, dtype=np.int64)
    target_out = np.ones_like(score_out, dtype=np.int64)
    roc_targets = np.concatenate([target_in, target_out])
    roc_scores = np.concatenate([score_in, score_out])
    fpr, tpr, thresholds = metrics.roc_curve(roc_targets, roc_scores)

    roc_metrics = compute_ood_roc_metrics(score_in=score_in, score_out=score_out)
    th_metrics = compute_threshold_metrics(score_train, score_in, score_out, args.train_known_quantile)

    known_metrics = None
    if y_test_in.exists():
        y_true = np.squeeze(np.load(y_test_in, allow_pickle=False)).astype(np.int64)
        if y_true.shape[0] != pred_in.shape[0]:
            raise ValueError(f"labelsTest_in and featuresTest_in mismatch: {y_true.shape[0]} vs {pred_in.shape[0]}")
        known_metrics = compute_known_classification_metrics(y_true, pred_in, class_labels)

    known_weighted_report = build_known_weighted_report(known_metrics)
    ood_report = build_ood_report(roc_metrics)
    unknown_detection_report = build_unknown_detection_report(th_metrics)

    elapsed = time.time() - t0

    summary = {
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "features_dir": str(features_dir),
        "feature_tag": str(args.feature_tag),
        "subspace_dim_requested": int(args.subspace_dim),
        "num_known_classes": int(class_labels.shape[0]),
        "num_unknown_test_samples": int(test_out.shape[0]),
        "ood_metrics": ood_report,
        "known_classification_metrics": known_weighted_report,
        "threshold_metrics": unknown_detection_report,
        "extract_metadata_path": str(extract_meta_path),
        "extract_dataset": (extract_meta.get("dataset") if isinstance(extract_meta, dict) else None),
        "extract_csv_path": (extract_meta.get("csv_path") if isinstance(extract_meta, dict) else None),
        "extract_cache_npz": (extract_meta.get("cache_npz") if isinstance(extract_meta, dict) else None),
        "extract_checkpoint": (extract_meta.get("checkpoint") if isinstance(extract_meta, dict) else None),
    }
    run_name = Path(args.work_dir).resolve().name
    run_info = parse_run_name(run_name)
    dataset_name = extract_meta.get("dataset") if isinstance(extract_meta, dict) else None
    noise_type = extract_meta.get("noise_type") if isinstance(extract_meta, dict) else None
    noise = extract_meta.get("noise") if isinstance(extract_meta, dict) else None
    cut = extract_meta.get("cut") if isinstance(extract_meta, dict) else None
    if run_info is not None:
        if noise_type is None:
            noise_type = run_info["noise_type"]
        if noise is None:
            noise = run_info["noise"]
        if cut is None:
            cut = run_info["cut"]
    roc_json_path, roc_csv_path = save_roc_curve_files(
        reports_dir=reports_dir,
        dataset=(str(dataset_name) if dataset_name is not None else None),
        run_name=run_name,
        stage=str(args.stage),
        noise_type=(str(noise_type) if noise_type is not None else None),
        noise=(float(noise) if noise is not None else None),
        cut=(float(cut) if cut is not None else None),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=float(roc_metrics["auc"]),
    )
    summary["roc_curve_files"] = {
        "json": str(roc_json_path),
        "csv": str(roc_csv_path),
    }

    with open(reports_dir / "ood_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ood_report, f, indent=2)
    with open(reports_dir / "threshold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(unknown_detection_report, f, indent=2)
    if known_weighted_report is not None:
        with open(reports_dir / "known_classification_metrics.json", "w", encoding="utf-8") as f:
            json.dump(known_weighted_report, f, indent=2)
    else:
        (reports_dir / "known_classification_metrics.json").write_text("null\n", encoding="utf-8")
    with open(reports_dir / "open_set_classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(unknown_detection_report, f, indent=2)
    with open(reports_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    auc_plot_result = None
    if args.plot_auc:
        auc_plot_result = aggregate_and_plot_auc_figures(args)

    if args.save_scores:
        np.savez_compressed(
            reports_dir / "scores_debug.npz",
            score_train=score_train,
            score_in=score_in,
            score_out=score_out,
            pred_train=pred_train,
            pred_in=pred_in,
            pred_out=pred_out,
            class_labels=class_labels,
        )

    print("=== OOD Metrics ===")
    print(f"AUC: {ood_report['auc']:.7f}")
    print("=== Unknown Detection (Threshold) ===")
    print(f"Tau: {unknown_detection_report['threshold_tau']:.6f}")
    print(f"Acc: {unknown_detection_report['acc']:.6f}")
    print(f"Precision: {unknown_detection_report['precision']:.6f}")
    print(f"Recall: {unknown_detection_report['recall']:.6f}")
    print(f"F1: {unknown_detection_report['f1']:.6f}")
    print(f"Confusion Matrix: {unknown_detection_report['confusion_matrix']}")
    print("=== Known Classification (Weighted) ===")
    if known_weighted_report is None:
        print("No known-class labels available; known weighted metrics were not computed.")
    else:
        print(f"Acc: {known_weighted_report['weighted_acc']:.6f}")
        print(f"Precision: {known_weighted_report['precision_weighted']:.6f}")
        print(f"Recall: {known_weighted_report['recall_weighted']:.6f}")
        print(f"F1: {known_weighted_report['f1_weighted']:.6f}")
    print(f"Saved reports to: {reports_dir}")
    print(f"Saved ROC curve: {roc_json_path}")
    if auc_plot_result is not None:
        print(f"Updated AUC figures under: {Path(args.auc_fig_dir).resolve()}")


if __name__ == "__main__":
    main()
