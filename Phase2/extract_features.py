#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
ARGUS_DIR = THIS_DIR.parents[0]
ROOT_DIR = ARGUS_DIR.parents[0]
ETC_DIR = ARGUS_DIR / "Phase1_Training"
for p in (ETC_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preresnet import DeepResNet  # type: ignore
from utils import (  # type: ignore
    build_or_load_csv_bundle,
    default_cache_npz_for_dataset,
    normalize_dataset_id,
    resolve_device,
    resolve_dataset_and_csv_path,
    resolve_ood_class_ids,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ID/OOD features from NLNL-ETC DeepResNet checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ARGUS_DIR / "logs_csv" / "d1" / "sym_10_PL_cut10_Stage_3" / "checkpoint.pth.tar"),
        help="Trained checkpoint path",
    )
    parser.add_argument("--dataset", type=str, default="D1", choices=["D1", "D2"], help="Dataset ID: D1=label_encodered_malicious_TLS-1_processed.csv, D2=TLS1.3_like_TLS1.2_processed.csv")
    parser.add_argument("--csv_path", type=str, default="", help="Optional explicit CSV path; empty means auto-resolve by --dataset")
    parser.add_argument("--cache_npz", type=str, default="", help="Optional explicit cache npz path; empty means auto-resolve by --dataset/noise/test_size")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise rate used in training")
    parser.add_argument("--noise_type", type=str, default="asym", choices=["symm_exc", "sym", "asym"], help="Noise type used in training")
    parser.add_argument("--test_size", type=float, default=0.2, help="Known split test size")
    parser.add_argument("--train_label_source", type=str, default="noisy", choices=["noisy", "clean"], help="Train labels for subspace construction")
    parser.add_argument(
        "--train_subset_strategy",
        type=str,
        default="cut",
        choices=["all", "cut"],
        help="Training subset for OOD subspace: all samples or only high-confidence samples by CUT from training checkpoint history",
    )
    parser.add_argument("--cut", type=float, default=0.5, help="Confidence cutoff used for --train_subset_strategy cut (keep prob >= cut)")
    parser.add_argument("--batch_size_train", type=int, default=1024)
    parser.add_argument("--batch_size_eval", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mc_runs", type=int, default=1, help="MC runs for test/OOD features")
    parser.add_argument(
        "--mc_dropout_p",
        type=float,
        default=0.0,
        help="Dropout probability used only for MC inference; 0 disables MC dropout",
    )
    parser.add_argument("--d1_extra_ood_from_d2", type=int, default=0, help="For D1 only: append N random D2 samples into OOD set")
    parser.add_argument("--d1_extra_ood_csv_path", type=str, default="", help="Optional explicit D2 csv path for --d1_extra_ood_from_d2")
    parser.add_argument("--ood_cap_random", type=int, default=0, help="If >0, randomly cap OOD pool to this many samples")
    parser.add_argument("--extra_ood_from_d4_1to9", type=int, default=0, help="Append N random samples from D4 labels [1..9] into OOD pool")
    parser.add_argument("--extra_ood_d4_csv_path", type=str, default="", help="Optional explicit D4 csv path for --extra_ood_from_d4_1to9")
    parser.add_argument("--feature_tag", type=str, default="internal_12", help="Saved OOD feature suffix")
    parser.add_argument("--stage", type=str, default="Stage_3")
    parser.add_argument("--work_dir", type=str, default=str(THIS_DIR), help="Output work dir")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2026)
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


def create_loader(x: np.ndarray, y: np.ndarray, batch_size: int, workers: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def forward_with_features(model: DeepResNet, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 2:
        x = x.unsqueeze(1)

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    feat = x.view(x.size(0), -1)
    if hasattr(model, "dropout"):
        feat = model.dropout(feat)
    logits = feat if model.fc is None else model.fc(feat)
    return logits, feat


def extract_train_features(model: DeepResNet, loader: DataLoader, device: torch.device, feat_dim: int) -> np.ndarray:
    model.eval()
    n = len(loader.dataset)
    out = np.empty((n, feat_dim), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            _, feat = forward_with_features(model, xb)
            bs = xb.size(0)
            out[offset : offset + bs] = feat.detach().cpu().numpy()
            offset += bs
    return out


def extract_eval_features(
    model: DeepResNet,
    loader: DataLoader,
    device: torch.device,
    feat_dim: int,
    mc_runs: int,
    mc_dropout_enabled: bool = False,
) -> np.ndarray:
    model.eval()
    if mc_dropout_enabled:
        # Keep BN/eval behavior globally but re-enable stochastic dropout only.
        _set_dropout_train_mode(model, enabled=True)
    n = len(loader.dataset)
    out = np.empty((n, feat_dim, mc_runs), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            bs = xb.size(0)
            feat_mc = torch.empty((bs, feat_dim, mc_runs), device=xb.device)
            for mc_idx in range(mc_runs):
                _, feat = forward_with_features(model, xb)
                feat_mc[:, :, mc_idx] = feat
            out[offset : offset + bs] = feat_mc.detach().cpu().numpy()
            offset += bs
    return out


def _set_dropout_train_mode(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(enabled)


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_id(args.dataset)
    dataset_id, resolved_csv_path = resolve_dataset_and_csv_path(args.dataset, args.csv_path)
    resolved_cache_npz = (
        Path(args.cache_npz).resolve()
        if str(args.cache_npz).strip()
        else default_cache_npz_for_dataset(ARGUS_DIR / "cache", dataset_id, args.noise_type, args.noise, args.seed, args.test_size)
    )
    args.csv_path = str(resolved_csv_path)
    args.cache_npz = str(resolved_cache_npz)
    if args.mc_runs <= 0:
        raise ValueError(f"--mc_runs must be > 0, got {args.mc_runs}")
    if not (0.0 <= args.mc_dropout_p < 1.0):
        raise ValueError(f"--mc_dropout_p must be in [0, 1), got {args.mc_dropout_p}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    paths = ensure_dirs(Path(args.work_dir).resolve())

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict_raw = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict_raw, dict):
        raise TypeError("checkpoint format invalid: expected dict with state_dict")
    state_dict = normalize_state_dict_keys(state_dict_raw)

    bundle = build_or_load_csv_bundle(
        csv_path=resolved_csv_path,
        cache_npz=resolved_cache_npz,
        noise_rate=args.noise,
        noise_type=args.noise_type,
        seed=args.seed,
        test_size=args.test_size,
        d1_extra_ood_from_d2=args.d1_extra_ood_from_d2,
        d1_extra_ood_csv_path=(Path(args.d1_extra_ood_csv_path).resolve() if str(args.d1_extra_ood_csv_path).strip() else None),
        ood_cap_random=args.ood_cap_random,
        extra_ood_from_d4_1to9=args.extra_ood_from_d4_1to9,
        extra_ood_d4_csv_path=(Path(args.extra_ood_d4_csv_path).resolve() if str(args.extra_ood_d4_csv_path).strip() else None),
    )
    expected_ood_class_ids = resolve_ood_class_ids(resolved_csv_path, dataset=dataset_id)
    if not np.array_equal(bundle.ood_class_ids_original, expected_ood_class_ids):
        raise RuntimeError(
            f"OOD class split mismatch: expected {expected_ood_class_ids.tolist()}, got {bundle.ood_class_ids_original.tolist()}"
        )

    num_classes_known = int(bundle.known_class_ids_original.shape[0])
    input_dim = int(bundle.x_train.shape[1])

    conv_w_key = "conv1.weight"
    if conv_w_key not in state_dict:
        raise KeyError(f"checkpoint state_dict missing key: {conv_w_key}")
    initial_channels = int(state_dict[conv_w_key].shape[0])
    fc_w_key = "fc.weight"
    if fc_w_key not in state_dict:
        raise KeyError(f"checkpoint state_dict missing key: {fc_w_key}")
    checkpoint_num_classes = int(state_dict[fc_w_key].shape[0])
    if checkpoint_num_classes != num_classes_known:
        raise RuntimeError(
            "Checkpoint/data mismatch detected: "
            f"checkpoint_head_classes={checkpoint_num_classes}, dataset_known_classes={num_classes_known}, "
            f"dataset={dataset_id}, checkpoint={checkpoint_path}, csv_path={resolved_csv_path}, cache_npz={resolved_cache_npz}. "
            f"known_original_ids={bundle.known_class_ids_original.astype(int).tolist()}, "
            f"ood_original_ids={bundle.ood_class_ids_original.astype(int).tolist()}. "
            "Please use a checkpoint trained on the same dataset split."
        )

    net = DeepResNet(
        input_size=input_dim,
        num_classes=checkpoint_num_classes,
        initial_channels=initial_channels,
        dropout_p=args.mc_dropout_p,
    ).to(device)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    if args.mc_dropout_p > 0.0:
        # Keep BN in eval mode, only enable dropout stochasticity for MC sampling.
        _set_dropout_train_mode(net, enabled=True)

    feat_dim = int(net.fc.in_features) if net.fc is not None else int(initial_channels * 8)

    y_train_full = bundle.y_train_noisy if args.train_label_source == "noisy" else bundle.y_train_clean
    x_train_full = bundle.x_train
    train_confidence = None
    keep_mask = np.ones((x_train_full.shape[0],), dtype=bool)

    if args.train_subset_strategy == "cut":
        if not isinstance(ckpt, dict) or "train_preds_hist" not in ckpt:
            raise KeyError("checkpoint missing train_preds_hist required by --train_subset_strategy cut")
        train_preds_hist = ckpt["train_preds_hist"]
        if not isinstance(train_preds_hist, torch.Tensor):
            train_preds_hist = torch.as_tensor(train_preds_hist)
        if train_preds_hist.ndim != 3:
            raise ValueError(f"train_preds_hist must be 3D [N,H,C], got shape={tuple(train_preds_hist.shape)}")
        if int(train_preds_hist.shape[0]) != int(y_train_full.shape[0]):
            raise ValueError(
                f"train_preds_hist sample size mismatch: hist_N={int(train_preds_hist.shape[0])}, train_N={int(y_train_full.shape[0])}"
            )
        mean_prob = train_preds_hist.float().mean(dim=1)
        row_idx = torch.arange(y_train_full.shape[0], dtype=torch.long)
        label_idx = torch.from_numpy(y_train_full.astype(np.int64))
        train_confidence = mean_prob[row_idx, label_idx].detach().cpu().numpy()
        keep_mask = train_confidence >= float(args.cut)
        if not np.any(keep_mask):
            raise RuntimeError(f"No training samples kept by CUT filtering: cut={args.cut}")

    x_train = x_train_full[keep_mask]
    y_train = y_train_full[keep_mask]

    train_loader = create_loader(x_train, y_train, args.batch_size_train, args.num_workers)
    test_in_loader = create_loader(bundle.x_test, bundle.y_test, args.batch_size_eval, args.num_workers)
    ood_dummy = np.zeros((bundle.x_ood.shape[0],), dtype=np.int64)
    test_out_loader = create_loader(bundle.x_ood, ood_dummy, args.batch_size_eval, args.num_workers)

    t0 = time.time()
    features_train = extract_train_features(net, train_loader, device, feat_dim)
    features_test_in = extract_eval_features(
        net,
        test_in_loader,
        device,
        feat_dim,
        args.mc_runs,
        mc_dropout_enabled=(args.mc_dropout_p > 0.0 and args.mc_runs > 1),
    )
    features_test_out = extract_eval_features(
        net,
        test_out_loader,
        device,
        feat_dim,
        args.mc_runs,
        mc_dropout_enabled=(args.mc_dropout_p > 0.0 and args.mc_runs > 1),
    )
    elapsed = time.time() - t0

    features_dir = paths["features"]
    out_path = features_dir / f"features_out_{args.feature_tag}.npy"
    np.save(features_dir / "featuresTrain_in.npy", features_train)
    np.save(features_dir / "labelsTrain_in.npy", y_train.astype(np.int64))
    np.save(features_dir / "featuresTest_in.npy", features_test_in)
    np.save(features_dir / "labelsTest_in.npy", bundle.y_test.astype(np.int64))
    np.save(features_dir / "labelsTest_in_original.npy", bundle.y_test_original.astype(np.int64))
    np.save(out_path, features_test_out)
    np.save(features_dir / f"labelsOut_{args.feature_tag}.npy", bundle.y_ood_original.astype(np.int64))

    metadata = {
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "dataset": str(dataset_id),
        "checkpoint": str(checkpoint_path),
        "csv_path": str(Path(args.csv_path).resolve()),
        "cache_npz": str(Path(args.cache_npz).resolve()),
        "noise": float(args.noise),
        "noise_type": str(args.noise_type),
        "test_size": float(args.test_size),
        "d1_extra_ood_from_d2": int(args.d1_extra_ood_from_d2),
        "d1_extra_ood_csv_path": (
            str(Path(args.d1_extra_ood_csv_path).resolve())
            if str(args.d1_extra_ood_csv_path).strip()
            else ""
        ),
        "ood_cap_random": int(args.ood_cap_random),
        "extra_ood_from_d4_1to9": int(args.extra_ood_from_d4_1to9),
        "extra_ood_d4_csv_path": (
            str(Path(args.extra_ood_d4_csv_path).resolve())
            if str(args.extra_ood_d4_csv_path).strip()
            else ""
        ),
        "actual_extra_ood_from_d4_1to9": int(np.sum(bundle.y_ood_original == -2)),
        "train_label_source": str(args.train_label_source),
        "train_subset_strategy": str(args.train_subset_strategy),
        "cut": float(args.cut),
        "n_train_full": int(x_train_full.shape[0]),
        "feature_tag": str(args.feature_tag),
        "stage": str(args.stage),
        "mc_runs": int(args.mc_runs),
        "mc_dropout_p": float(args.mc_dropout_p),
        "feature_dim": int(feat_dim),
        "input_dim": int(input_dim),
        "num_classes_known": int(num_classes_known),
        "num_classes_checkpoint_head": int(checkpoint_num_classes),
        "known_class_ids_original": bundle.known_class_ids_original.astype(int).tolist(),
        "ood_class_ids_original": bundle.ood_class_ids_original.astype(int).tolist(),
        "known_remap_pairs": bundle.known_remap_pairs.astype(int).tolist(),
        "n_train": int(features_train.shape[0]),
        "n_train_kept": int(np.sum(keep_mask)),
        "train_keep_ratio": float(np.mean(keep_mask.astype(np.float32))),
        "n_test_in": int(features_test_in.shape[0]),
        "n_test_out": int(features_test_out.shape[0]),
        "saved_files": {
            "featuresTrain_in": str(features_dir / "featuresTrain_in.npy"),
            "labelsTrain_in": str(features_dir / "labelsTrain_in.npy"),
            "featuresTest_in": str(features_dir / "featuresTest_in.npy"),
            "labelsTest_in": str(features_dir / "labelsTest_in.npy"),
            "labelsTest_in_original": str(features_dir / "labelsTest_in_original.npy"),
            "features_out": str(out_path),
            "labels_out_original": str(features_dir / f"labelsOut_{args.feature_tag}.npy"),
        },
    }
    if train_confidence is not None:
        metadata["train_confidence_stats"] = {
            "min": float(np.min(train_confidence)),
            "max": float(np.max(train_confidence)),
            "mean": float(np.mean(train_confidence)),
            "std": float(np.std(train_confidence)),
        }
    with open(paths["logs"] / "extract_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"feature extraction done. train={features_train.shape}, test_in={features_test_in.shape}, out={features_test_out.shape}")
    print(f"saved to: {features_dir}")


if __name__ == "__main__":
    main()
