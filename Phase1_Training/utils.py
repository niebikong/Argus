#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class CsvDataBundle:
    x_train: np.ndarray
    y_train_clean: np.ndarray
    y_train_noisy: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_test_original: np.ndarray
    x_ood: np.ndarray
    y_ood_original: np.ndarray
    noisy_idx: np.ndarray
    clean_idx: np.ndarray
    classes_original: np.ndarray
    known_class_ids_original: np.ndarray
    ood_class_ids_original: np.ndarray
    known_remap_pairs: np.ndarray
    scaler_min: np.ndarray
    scaler_range: np.ndarray


class IndexedCSVDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class NpMinMaxScaler:
    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "NpMinMaxScaler":
        mn = np.min(x, axis=0)
        mx = np.max(x, axis=0)
        rg = mx - mn
        rg[rg == 0] = 1.0
        self.min_ = mn
        self.range_ = rg
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler not fitted")
        return ((x - self.min_) / self.range_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda set but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(save_dir: Path, filename: str = "train.log") -> logging.Logger:
    ensure_dir(save_dir)
    logger_name = f"nlnl_csv_{save_dir.name}_{os.getpid()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(save_dir / filename)
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def csv_to_numeric_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == "object":
            codes, _ = pd.factorize(df[col].astype(str), sort=True)
            df[col] = codes
    df = df.astype(float)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def label_encode_numpy(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes, y_encoded = np.unique(y_raw, return_inverse=True)
    return y_encoded.astype(np.int64), classes


def safe_class_split_count(n_cls: int, ratio: float) -> int:
    if n_cls <= 1:
        return 0
    n = int(round(n_cls * ratio))
    n = max(1, n)
    n = min(n, n_cls - 1)
    return n


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    rng = np.random.RandomState(seed)
    train_idx_parts: List[np.ndarray] = []
    test_idx_parts: List[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        n_test = safe_class_split_count(len(cls_idx), test_ratio)
        if n_test == 0:
            train_idx_parts.append(cls_idx)
            continue
        test_idx_parts.append(cls_idx[:n_test])
        train_idx_parts.append(cls_idx[n_test:])

    if len(test_idx_parts) == 0:
        raise RuntimeError("No test samples produced in stratified split")

    train_idx = np.concatenate(train_idx_parts)
    test_idx = np.concatenate(test_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def inject_symm_exc_noise(
    y_clean: np.ndarray,
    num_classes: int,
    noise_rate: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror NLNL/noise_generator.py with symm_exc behavior:
    - per class noisy count = int(N * noise_rate / num_classes)
    - each selected sample flips to any class except true class.
    """
    if noise_rate < 0 or noise_rate >= 1:
        raise ValueError(f"noise_rate must be in [0,1), got {noise_rate}")

    rng = np.random.RandomState(seed)
    y_noisy = y_clean.copy().astype(np.int64)
    noisy_idx_list: List[int] = []

    n_per_class = int(len(y_clean) * noise_rate / num_classes)
    for cls in range(num_classes):
        cls_idx = np.where(y_clean == cls)[0]
        if cls_idx.size == 0:
            continue
        take = min(n_per_class, cls_idx.size)
        if take > 0:
            noisy_idx_list.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

    noisy_idx = np.array(sorted(set(noisy_idx_list)), dtype=np.int64)

    for idx in noisy_idx:
        true_label = int(y_noisy[idx])
        choices = np.concatenate(
            [
                np.arange(0, true_label, dtype=np.int64),
                np.arange(true_label + 1, num_classes, dtype=np.int64),
            ]
        )
        y_noisy[idx] = int(rng.choice(choices))

    return y_noisy, noisy_idx


def multiclass_noisify(y: np.ndarray, p: np.ndarray, random_state: int = 42) -> np.ndarray:
    new_y = y.copy().astype(np.int64)
    rng = np.random.RandomState(random_state)
    for idx in np.arange(y.shape[0]):
        cls_id = int(y[idx])
        flipped = rng.multinomial(1, p[cls_id, :], 1)[0]
        new_y[idx] = int(np.where(flipped == 1)[0][0])
    return new_y


def noisify_sym(y: np.ndarray, num_classes: int, noise_rate: float, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    p = noise_rate / (num_classes - 1) * np.ones((num_classes, num_classes), dtype=np.float64)
    np.fill_diagonal(p, 1.0 - noise_rate)
    return multiclass_noisify(y, p, random_state=random_state), p


def noisify_asym(y: np.ndarray, num_classes: int, noise_rate: float, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    p = np.eye(num_classes, dtype=np.float64)
    transition = {0: 2, 4: 7, 5: 6, 9: 1, 3: 8}
    for src, dst in transition.items():
        if src < num_classes and dst < num_classes:
            p[src, src] = 1.0 - noise_rate
            p[src, dst] = noise_rate
    return multiclass_noisify(y, p, random_state=random_state), p


def noisify_binary_asym(
    y: np.ndarray,
    noise_rate: float,
    random_state: int = 42,
    src_label: int = 1,
    dst_label: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    # Binary asymmetric noise:
    # only one direction flips (default: 1 -> 0), the other class stays unchanged.
    p = np.eye(2, dtype=np.float64)
    p[src_label, src_label] = 1.0 - noise_rate
    p[src_label, dst_label] = noise_rate
    return multiclass_noisify(y, p, random_state=random_state), p


def inject_label_noise(
    y_clean: np.ndarray,
    num_classes: int,
    noise_rate: float,
    seed: int,
    noise_type: str,
    dataset_id: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if noise_type == "symm_exc":
        y_noisy, noisy_idx = inject_symm_exc_noise(y_clean, num_classes, noise_rate, seed)
        return y_noisy, noisy_idx
    if noise_type == "sym":
        y_noisy, _ = noisify_sym(y_clean, num_classes, noise_rate, random_state=seed)
        noisy_idx = np.where(y_noisy != y_clean)[0].astype(np.int64)
        return y_noisy.astype(np.int64), noisy_idx
    if noise_type == "asym":
        y_noisy, _ = noisify_asym(y_clean, num_classes, noise_rate, random_state=seed)
        noisy_idx = np.where(y_noisy != y_clean)[0].astype(np.int64)
        return y_noisy.astype(np.int64), noisy_idx
    raise ValueError(f"Unsupported noise_type: {noise_type}. Expected one of ['symm_exc', 'sym', 'asym']")


CACHE_POLICY_VERSION = 15
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
REF_CODES_DIR = PROJECT_ROOT_DIR / "Ref_codes"

DATASET_ID_TO_CSV_FILENAME = {
    "D1": "label_encodered_malicious_TLS-1_processed.csv",
    "D2": "TLS1.3_like_TLS1.2_processed.csv",
}
DATASET_ID_TO_LABEL_FILENAME: Dict[str, str] = {}
DATASET_ID_TO_OOD_CLASS_IDS = {
    "D1": np.asarray([1, 2], dtype=np.int64),
    "D2": np.asarray([36, 37, 38, 39, 40], dtype=np.int64),
}
DATASET_ID_TO_KNOWN_CLASS_IDS: Dict[str, np.ndarray] = {}
DATASET_OOD_CLASS_IDS_BY_FILENAME = {
    DATASET_ID_TO_CSV_FILENAME[k]: v for k, v in DATASET_ID_TO_OOD_CLASS_IDS.items()
}
DEFAULT_DATASET_ID = "D1"
DEFAULT_OOD_CLASS_IDS = DATASET_ID_TO_OOD_CLASS_IDS[DEFAULT_DATASET_ID]
# Backward-compatible alias used by existing scripts.
FIXED_OOD_CLASS_IDS = DEFAULT_OOD_CLASS_IDS.copy()


def normalize_dataset_id(dataset: str) -> str:
    ds = str(dataset).strip().upper()
    if ds not in DATASET_ID_TO_CSV_FILENAME:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of: {sorted(DATASET_ID_TO_CSV_FILENAME.keys())}"
        )
    return ds


def infer_dataset_id_from_csv_path(csv_path: Path) -> str | None:
    file_name = Path(csv_path).name
    for ds, csv_file in DATASET_ID_TO_CSV_FILENAME.items():
        if file_name == csv_file:
            return ds
    return None


def default_csv_path_for_dataset(dataset: str) -> Path:
    ds = normalize_dataset_id(dataset)
    return (REF_CODES_DIR / DATASET_ID_TO_CSV_FILENAME[ds]).resolve()


def default_label_csv_path_for_dataset(dataset: str) -> Path | None:
    ds = normalize_dataset_id(dataset)
    label_name = DATASET_ID_TO_LABEL_FILENAME.get(ds)
    if label_name is None:
        return None
    return (REF_CODES_DIR / label_name).resolve()


def default_cache_npz_for_dataset(
    cache_dir: Path,
    dataset: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    test_size: float,
) -> Path:
    ds = normalize_dataset_id(dataset).lower()
    noise_token = str(noise_rate).replace(".", "")
    test_token = int(round(float(test_size) * 100.0))
    return (Path(cache_dir) / f"tls_csv_{ds}_{noise_type}_n{noise_token}_seed{int(seed)}_test{test_token:02d}.npz").resolve()


def resolve_dataset_and_csv_path(dataset: str, csv_path: str) -> Tuple[str, Path]:
    ds = normalize_dataset_id(dataset)
    if str(csv_path).strip():
        resolved_csv = Path(csv_path).resolve()
        inferred_ds = infer_dataset_id_from_csv_path(resolved_csv)
        if inferred_ds is not None and inferred_ds != ds:
            raise ValueError(
                f"--dataset {ds} does not match --csv_path {resolved_csv.name} (belongs to {inferred_ds})"
            )
        return ds, resolved_csv
    return ds, default_csv_path_for_dataset(ds)


def resolve_ood_class_ids(csv_path: Path, dataset: str | None = None) -> np.ndarray:
    if dataset is not None:
        ds = normalize_dataset_id(dataset)
        return DATASET_ID_TO_OOD_CLASS_IDS[ds].astype(np.int64, copy=True)
    inferred_ds = infer_dataset_id_from_csv_path(Path(csv_path))
    if inferred_ds is not None:
        return DATASET_ID_TO_OOD_CLASS_IDS[inferred_ds].astype(np.int64, copy=True)
    return DEFAULT_OOD_CLASS_IDS.astype(np.int64, copy=True)


def resolve_known_class_ids(csv_path: Path, dataset: str | None = None) -> np.ndarray | None:
    if dataset is not None:
        ds = normalize_dataset_id(dataset)
        known = DATASET_ID_TO_KNOWN_CLASS_IDS.get(ds)
        return None if known is None else known.astype(np.int64, copy=True)
    inferred_ds = infer_dataset_id_from_csv_path(Path(csv_path))
    if inferred_ds is not None and inferred_ds in DATASET_ID_TO_KNOWN_CLASS_IDS:
        return DATASET_ID_TO_KNOWN_CLASS_IDS[inferred_ds].astype(np.int64, copy=True)
    return None


def apply_dataset_specific_sampling(
    x: np.ndarray | pd.DataFrame,
    y_raw: np.ndarray,
    dataset_id: str | None,
    seed: int,
) -> Tuple[np.ndarray | pd.DataFrame, np.ndarray, str]:
    _ = dataset_id
    _ = seed
    return x, y_raw, "none"


def load_features_and_labels_for_dataset(csv_path: Path, dataset_id: str | None) -> Tuple[np.ndarray | pd.DataFrame, np.ndarray, Path | None]:
    df = csv_to_numeric_frame(csv_path)

    x = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    return x, y_raw, None


def build_or_load_csv_bundle(
    csv_path: Path,
    cache_npz: Path,
    noise_rate: float,
    noise_type: str,
    seed: int,
    test_size: float,
    d1_extra_ood_from_d2: int = 0,
    d1_extra_ood_csv_path: Path | None = None,
    ood_cap_random: int = 0,
    extra_ood_from_d4_1to9: int = 0,
    extra_ood_d4_csv_path: Path | None = None,
) -> CsvDataBundle:
    csv_path = csv_path.resolve()
    dataset_id = infer_dataset_id_from_csv_path(csv_path)
    label_csv_path = (
        default_label_csv_path_for_dataset(dataset_id)
        if dataset_id in DATASET_ID_TO_LABEL_FILENAME
        else None
    )
    ood_class_ids = resolve_ood_class_ids(csv_path)
    known_class_ids_cfg = resolve_known_class_ids(csv_path, dataset=dataset_id)
    unknown_ids_str = ",".join(str(int(v)) for v in ood_class_ids.tolist())
    known_ids_str = (
        ",".join(str(int(v)) for v in known_class_ids_cfg.tolist())
        if known_class_ids_cfg is not None
        else "__all_except_ood__"
    )
    split_policy = "dataset_specific_ood_known_remap_then_noise"
    if dataset_id not in {"D1", "D2"}:
        raise ValueError(f"Unsupported dataset_id for Argus simplified pipeline: {dataset_id}")
    d1_extra_ood_from_d2 = int(d1_extra_ood_from_d2)
    if d1_extra_ood_from_d2 < 0:
        raise ValueError(f"d1_extra_ood_from_d2 must be >= 0, got {d1_extra_ood_from_d2}")
    d1_extra_ood_csv_resolved = (
        Path(d1_extra_ood_csv_path).resolve()
        if d1_extra_ood_csv_path is not None and str(d1_extra_ood_csv_path).strip()
        else default_csv_path_for_dataset("D2")
    )
    d1_extra_ood_csv_path_str = str(d1_extra_ood_csv_resolved) if (dataset_id == "D1" and d1_extra_ood_from_d2 > 0) else ""
    if dataset_id == "D1" and d1_extra_ood_from_d2 > 0:
        split_policy = f"{split_policy}_d1_plus_d2_extra_ood_{d1_extra_ood_from_d2}"
    ood_cap_random = int(ood_cap_random)
    if ood_cap_random < 0:
        raise ValueError(f"ood_cap_random must be >= 0, got {ood_cap_random}")
    if ood_cap_random > 0:
        split_policy = f"{split_policy}_ood_cap_random_{ood_cap_random}"
    # Keep args in signature for compatibility, but disable extra OOD path in D1-only pipeline.
    _ = extra_ood_from_d4_1to9
    _ = extra_ood_d4_csv_path
    extra_ood_from_d4_1to9 = 0
    extra_ood_d4_csv_path_str = ""
    sampling_policy = "none"
    label_csv_path_str = str(label_csv_path) if label_csv_path is not None else ""

    if cache_npz.exists():
        try:
            payload = np.load(cache_npz, allow_pickle=True)
        except Exception:
            # Corrupted/incomplete cache file; rebuild from CSV.
            cache_npz.unlink(missing_ok=True)
            payload = None
        if payload is not None:
            meta_ok = (
                ("meta_noise_rate" in payload)
                and ("meta_noise_type" in payload)
                and ("meta_seed" in payload)
                and ("meta_test_size" in payload)
                and ("meta_cache_policy_version" in payload)
                and ("meta_csv_path" in payload)
                and ("meta_ood_class_ids_str" in payload)
                and ("meta_known_class_ids_str" in payload)
                and ("meta_split_policy" in payload)
                and ("meta_sampling_policy" in payload)
                and (abs(float(payload["meta_noise_rate"]) - float(noise_rate)) < 1e-12)
                and (str(payload["meta_noise_type"].item()) == str(noise_type))
                and (int(payload["meta_seed"]) == int(seed))
                and (abs(float(payload["meta_test_size"]) - float(test_size)) < 1e-12)
                and (int(payload["meta_cache_policy_version"]) == CACHE_POLICY_VERSION)
                and (str(payload["meta_csv_path"].item()) == str(csv_path))
                and (
                    (
                        str(payload["meta_label_csv_path"].item())
                        if "meta_label_csv_path" in payload
                        else ""
                    )
                    == label_csv_path_str
                )
                and (str(payload["meta_ood_class_ids_str"].item()) == unknown_ids_str)
                and (str(payload["meta_known_class_ids_str"].item()) == known_ids_str)
                and (str(payload["meta_split_policy"].item()) == split_policy)
                and (
                    (
                        int(payload["meta_d1_extra_ood_from_d2"])
                        if "meta_d1_extra_ood_from_d2" in payload
                        else 0
                    )
                    == d1_extra_ood_from_d2
                )
                and (
                    (
                        str(payload["meta_d1_extra_ood_from_d2_csv_path"].item())
                        if "meta_d1_extra_ood_from_d2_csv_path" in payload
                        else ""
                    )
                    == d1_extra_ood_csv_path_str
                )
                and (
                    (
                        int(payload["meta_ood_cap_random"])
                        if "meta_ood_cap_random" in payload
                        else 0
                    )
                    == ood_cap_random
                )
                and (
                    (
                        int(payload["meta_extra_ood_from_d4_1to9"])
                        if "meta_extra_ood_from_d4_1to9" in payload
                        else 0
                    )
                    == extra_ood_from_d4_1to9
                )
                and (
                    (
                        str(payload["meta_extra_ood_d4_csv_path"].item())
                        if "meta_extra_ood_d4_csv_path" in payload
                        else ""
                    )
                    == extra_ood_d4_csv_path_str
                )
                and (
                    (
                        (str(payload["meta_dataset_id"].item()) if "meta_dataset_id" in payload else "")
                        == (dataset_id or "")
                    )
                )
                and (
                    str(payload["meta_sampling_policy"].item()) == "none"
                )
            )
            if meta_ok:
                return CsvDataBundle(
                    x_train=payload["x_train"].astype(np.float32),
                    y_train_clean=payload["y_train_clean"].astype(np.int64),
                    y_train_noisy=payload["y_train_noisy"].astype(np.int64),
                    x_test=payload["x_test"].astype(np.float32),
                    y_test=payload["y_test"].astype(np.int64),
                    y_test_original=payload["y_test_original"].astype(np.int64),
                    x_ood=payload["x_ood"].astype(np.float32),
                    y_ood_original=payload["y_ood_original"].astype(np.int64),
                    noisy_idx=payload["noisy_idx"].astype(np.int64),
                    clean_idx=payload["clean_idx"].astype(np.int64),
                    classes_original=payload["classes_original"].astype(np.int64),
                    known_class_ids_original=payload["known_class_ids_original"].astype(np.int64),
                    ood_class_ids_original=payload["ood_class_ids_original"].astype(np.int64),
                    known_remap_pairs=payload["known_remap_pairs"].astype(np.int64),
                    scaler_min=payload["scaler_min"].astype(np.float32),
                    scaler_range=payload["scaler_range"].astype(np.float32),
                )

    x, y_raw, label_csv_path_loaded = load_features_and_labels_for_dataset(csv_path, dataset_id)
    if label_csv_path_loaded is not None:
        label_csv_path = label_csv_path_loaded
        label_csv_path_str = str(label_csv_path)
    x, y_raw, sampling_policy = apply_dataset_specific_sampling(x, y_raw, dataset_id, seed)

    all_classes = np.sort(np.unique(y_raw)).astype(np.int64)
    missing_ood = ood_class_ids[~np.isin(ood_class_ids, all_classes)]
    if missing_ood.size > 0:
        raise ValueError(
            f"Configured unknown/OOD classes {ood_class_ids.tolist()} are not all present in {csv_path}. "
            f"Missing: {missing_ood.tolist()}, available classes: {all_classes.tolist()}"
        )

    if known_class_ids_cfg is None:
        known_class_ids = all_classes[~np.isin(all_classes, ood_class_ids)]
    else:
        missing_known = known_class_ids_cfg[~np.isin(known_class_ids_cfg, all_classes)]
        if missing_known.size > 0:
            raise ValueError(
                f"Configured known classes {known_class_ids_cfg.tolist()} are not all present in {csv_path}. "
                f"Missing: {missing_known.tolist()}, available classes: {all_classes.tolist()}"
            )
        known_class_ids = known_class_ids_cfg.copy()
    if known_class_ids.size == 0:
        raise ValueError(f"No known classes left after removing OOD classes {ood_class_ids.tolist()}")

    known_mask = np.isin(y_raw, known_class_ids)
    ood_mask = np.isin(y_raw, ood_class_ids)
    if int(np.sum(known_mask)) == 0:
        raise ValueError("No known samples available after class filtering")

    x_known = x[known_mask]
    y_known_original = y_raw[known_mask]
    x_ood_raw = x[ood_mask]
    y_ood_original = y_raw[ood_mask]

    # Cap native unknown pool first (before appending cross-dataset OOD samples).
    if ood_cap_random > 0 and x_ood_raw.shape[0] > ood_cap_random:
        rng_cap = np.random.RandomState(seed + 4049)
        pick_cap = rng_cap.choice(x_ood_raw.shape[0], size=int(ood_cap_random), replace=False)
        if isinstance(x_ood_raw, pd.DataFrame):
            x_ood_raw = x_ood_raw.iloc[pick_cap].reset_index(drop=True)
        else:
            x_ood_raw = x_ood_raw[pick_cap]
        y_ood_original = y_ood_original[pick_cap]

    if dataset_id == "D1" and d1_extra_ood_from_d2 > 0:
        x_d2_all, _, _ = load_features_and_labels_for_dataset(d1_extra_ood_csv_resolved, "D2")
        if isinstance(x_d2_all, pd.DataFrame):
            x_d2_all = x_d2_all.to_numpy(dtype=np.float32)
        if not isinstance(x_d2_all, np.ndarray):
            raise TypeError("D2 extra OOD source must resolve to ndarray features")
        if x_d2_all.ndim != 2:
            raise ValueError(f"D2 extra OOD features must be 2D, got shape {x_d2_all.shape}")
        if x_d2_all.shape[1] != x_ood_raw.shape[1]:
            raise ValueError(
                "D1/D2 feature dimension mismatch for extra OOD samples: "
                f"D1 dim={x_ood_raw.shape[1]}, D2 dim={x_d2_all.shape[1]}, d2_csv={d1_extra_ood_csv_resolved}"
            )
        take = min(int(d1_extra_ood_from_d2), int(x_d2_all.shape[0]))
        if take > 0:
            rng_extra = np.random.RandomState(seed + 1337)
            pick = rng_extra.choice(x_d2_all.shape[0], size=take, replace=False)
            x_d2_extra = x_d2_all[pick].astype(np.float32, copy=False)
            x_ood_raw = np.concatenate([x_ood_raw, x_d2_extra], axis=0)
            y_ood_original = np.concatenate([y_ood_original, np.full((take,), -1, dtype=np.int64)], axis=0)

    actual_extra_ood_from_d4_1to9 = 0

    y_known_remap = np.searchsorted(known_class_ids, y_known_original).astype(np.int64)
    known_remap_pairs = np.stack(
        [known_class_ids.astype(np.int64), np.arange(known_class_ids.size, dtype=np.int64)],
        axis=1,
    )

    x_train, x_test, y_train_clean, y_test = stratified_split(x_known, y_known_remap, test_size, seed)
    y_test_original = known_class_ids[y_test]
    scaler = NpMinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_ood = scaler.transform(x_ood_raw) if x_ood_raw.shape[0] > 0 else np.empty((0, x_train.shape[1]), dtype=np.float32)
    scaler_min = scaler.min_.astype(np.float32)
    scaler_range = scaler.range_.astype(np.float32)

    num_classes = int(known_class_ids.size)
    y_train_noisy, noisy_idx = inject_label_noise(
        y_clean=y_train_clean,
        num_classes=num_classes,
        noise_rate=noise_rate,
        seed=seed,
        noise_type=noise_type,
        dataset_id=dataset_id,
    )
    all_idx = np.arange(len(y_train_clean), dtype=np.int64)
    clean_idx = np.setdiff1d(all_idx, noisy_idx)

    ensure_dir(cache_npz.parent)
    np.savez_compressed(
        cache_npz,
        x_train=x_train,
        y_train_clean=y_train_clean,
        y_train_noisy=y_train_noisy,
        x_test=x_test,
        y_test=y_test,
        y_test_original=y_test_original,
        x_ood=x_ood,
        y_ood_original=y_ood_original,
        noisy_idx=noisy_idx,
        clean_idx=clean_idx,
        classes_original=known_class_ids.astype(np.int64),
        known_class_ids_original=known_class_ids.astype(np.int64),
        ood_class_ids_original=ood_class_ids.astype(np.int64),
        known_remap_pairs=known_remap_pairs.astype(np.int64),
        scaler_min=scaler_min,
        scaler_range=scaler_range,
        meta_noise_rate=np.asarray(noise_rate, dtype=np.float64),
        meta_noise_type=np.asarray(noise_type, dtype=object),
        meta_seed=np.asarray(seed, dtype=np.int64),
        meta_test_size=np.asarray(test_size, dtype=np.float64),
        meta_cache_policy_version=np.asarray(CACHE_POLICY_VERSION, dtype=np.int64),
        meta_dataset_id=np.asarray(dataset_id if dataset_id is not None else "", dtype=object),
        meta_csv_path=np.asarray(str(csv_path), dtype=object),
        meta_label_csv_path=np.asarray(label_csv_path_str, dtype=object),
        meta_ood_class_ids_str=np.asarray(unknown_ids_str, dtype=object),
        meta_known_class_ids_str=np.asarray(known_ids_str, dtype=object),
        meta_split_policy=np.asarray(split_policy, dtype=object),
        meta_sampling_policy=np.asarray(sampling_policy, dtype=object),
        meta_d1_extra_ood_from_d2=np.asarray(d1_extra_ood_from_d2, dtype=np.int64),
        meta_d1_extra_ood_from_d2_csv_path=np.asarray(d1_extra_ood_csv_path_str, dtype=object),
        meta_ood_cap_random=np.asarray(ood_cap_random, dtype=np.int64),
        meta_extra_ood_from_d4_1to9=np.asarray(extra_ood_from_d4_1to9, dtype=np.int64),
        meta_extra_ood_d4_csv_path=np.asarray(extra_ood_d4_csv_path_str, dtype=object),
        meta_actual_extra_ood_from_d4_1to9=np.asarray(actual_extra_ood_from_d4_1to9, dtype=np.int64),
    )

    return CsvDataBundle(
        x_train=x_train,
        y_train_clean=y_train_clean,
        y_train_noisy=y_train_noisy,
        x_test=x_test,
        y_test=y_test,
        y_test_original=y_test_original,
        x_ood=x_ood,
        y_ood_original=y_ood_original,
        noisy_idx=noisy_idx,
        clean_idx=clean_idx,
        classes_original=known_class_ids.astype(np.int64),
        known_class_ids_original=known_class_ids.astype(np.int64),
        ood_class_ids_original=ood_class_ids.astype(np.int64),
        known_remap_pairs=known_remap_pairs.astype(np.int64),
        scaler_min=scaler_min.astype(np.float32),
        scaler_range=scaler_range.astype(np.float32),
    )


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    effective_shuffle = bool(shuffle and len(dataset) > 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )


def sample_negative_labels(labels: torch.Tensor, num_classes: int, ln_neg: int = 1) -> torch.Tensor:
    labels_neg = (
        labels.unsqueeze(-1).repeat(1, ln_neg)
        + torch.randint(1, num_classes, (len(labels), ln_neg), device=labels.device)
    ) % num_classes
    return labels_neg.long()


def class_weights_from_labels(y_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = 1.0 / (counts / counts.max())
    return torch.from_numpy(weights.astype(np.float32))


def eval_test(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            bs = y.size(0)
            total_n += bs
            total_loss += float(loss.item()) * bs
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    model.train()
    if total_n == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    y_true = np.concatenate(all_true, axis=0).astype(np.int64)
    y_pred = np.concatenate(all_pred, axis=0).astype(np.int64)
    num_classes = int(max(int(y_true.max(initial=0)), int(y_pred.max(initial=0))) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1.0)

    tp = np.diag(cm)
    support = np.sum(cm, axis=1)
    pred_count = np.sum(cm, axis=0)

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    weights = np.divide(
        support,
        np.sum(support),
        out=np.zeros_like(support),
        where=np.sum(support) > 0,
    )

    acc = float(np.mean(y_true == y_pred))
    return {
        "loss": float(total_loss / total_n),
        "acc": acc,
        "precision_weighted": float(np.sum(precision * weights)),
        "recall_weighted": float(np.sum(recall * weights)),
        "f1_weighted": float(np.sum(f1 * weights)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


def model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def prob_correct_from_hist(train_preds_hist: torch.Tensor, labels_np: np.ndarray) -> np.ndarray:
    mean_prob = train_preds_hist.mean(1)
    idx = torch.arange(mean_prob.shape[0], dtype=torch.long)
    labels_t = torch.from_numpy(labels_np.astype(np.int64))
    p = mean_prob[idx, labels_t].detach().cpu().numpy()
    return p


def save_histogram(prob_correct: np.ndarray, save_path: Path) -> None:
    plt.hist(prob_correct, bins=20, range=(0.0, 1.0), edgecolor="black", color="g")
    plt.xlabel("probability")
    plt.ylabel("number of data")
    plt.grid()
    plt.savefig(str(save_path))
    plt.clf()



def save_separated_histogram(
    prob_correct: np.ndarray,
    clean_idx: np.ndarray,
    noisy_idx: np.ndarray,
    save_path: Path,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18

    clean_color = "#849FCE"
    noisy_color = "#F19841"
    edge_color="#BFBFBF"

    fig, ax = plt.subplots(figsize=(8, 6))

    if clean_idx.size > 0:
        ax.hist(
            prob_correct[clean_idx],
            bins=20,
            range=(0.0, 1.0),
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=1.0,
            color=clean_color,
            label="Clean",
        )

    if noisy_idx.size > 0:
        ax.hist(
            prob_correct[noisy_idx],
            bins=20,
            range=(0.0, 1.0),
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=1.0,
            color=noisy_color,
            label="Noisy",
        )

    ax.set_xlabel("Probability")
    ax.set_ylabel("Number of data")
    ax.grid(True, color="#B0B0B0", linewidth=0.8)

    fig.savefig(
        str(save_path),
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )

    plt.close(fig)



def compute_recall_precision_from_losses(
    train_losses: torch.Tensor,
    noise_ratio: float,
    noisy_idx: np.ndarray,
) -> Tuple[float, float]:
    losses_np = train_losses.detach().cpu().numpy()
    inds = np.argsort(losses_np)[::-1]
    rnge = int(len(losses_np) * noise_ratio)
    if rnge <= 0:
        return 0.0, 0.0
    inds_filt = inds[:rnge]
    hit = np.intersect1d(inds_filt, noisy_idx)
    recall = float(len(hit)) / float(max(1, len(noisy_idx)))
    precision = float(len(hit)) / float(max(1, rnge))
    return recall, precision


def save_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def record_ce_cut_label_accuracy_epoch(
    *,
    epoch: int,
    y_train_labels: np.ndarray,
    y_train_clean: np.ndarray,
    cut_filtered_indices: np.ndarray,
    records: List[Dict[str, float]],
    save_path: Path,
) -> Dict[str, float]:
    """
    Record CE-training label correctness for one epoch.

    CE-training samples are those NOT in ``cut_filtered_indices`` (i.e., samples
    above CUT that are kept for cross-entropy loss).
    """
    n_train = int(y_train_labels.shape[0])
    keep_mask = np.ones(n_train, dtype=bool)
    if cut_filtered_indices.size > 0:
        keep_mask[cut_filtered_indices.astype(np.int64)] = False

    ce_count = int(keep_mask.sum())
    if ce_count <= 0:
        acc = 0.0
        correct = 0
    else:
        correct = int((y_train_labels[keep_mask] == y_train_clean[keep_mask]).sum())
        acc = float(correct / ce_count)

    entry = {
        "epoch": int(epoch),
        "accuracy": float(acc),
    }
    records.append(entry)

    payload = {
        "metric": "ce_cut_label_accuracy",
        "num_train": n_train,
        "num_ce_samples": ce_count,
        "num_ce_correct": int(correct),
        "epoch_accuracy": records,
    }
    save_json(payload, save_path)
    return entry
