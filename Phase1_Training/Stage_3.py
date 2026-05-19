#!/usr/bin/env python3
# 每个 epoch 动态按 CUT 重新选训练集”，不再在开始时固定。
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]

from preresnet import DeepResNet  # type: ignore
from utils import (  # type: ignore
    IndexedCSVDataset,
    build_or_load_csv_bundle,
    class_weights_from_labels,
    default_cache_npz_for_dataset,
    create_loader,
    ensure_dir,
    eval_test,
    load_model_state,
    model_state_dict,
    normalize_dataset_id,
    prob_correct_from_hist,
    resolve_device,
    resolve_dataset_and_csv_path,
    resolve_ood_class_ids,
    record_ce_cut_label_accuracy_epoch,
    save_json,
    save_separated_histogram,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argus stage-3 training for CSV + DeepResNet")
    parser.add_argument("--dataset", type=str, default="D1", choices=["D1", "D2"], help="Dataset ID: D1=label_encodered_malicious_TLS-1_processed.csv, D2=TLS1.3_like_TLS1.2_processed.csv")
    parser.add_argument("--csv_path", type=str, default="", help="Optional explicit CSV path; empty means auto-resolve by --dataset")
    parser.add_argument("--cache_npz", type=str, default="", help="Optional explicit cache npz path; empty means auto-resolve by --dataset/noise/test_size")
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--noise_type", type=str, default="sym", choices=["symm_exc", "sym", "asym"])
    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10) # 200
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--epoch_step", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--cut", type=float, default=0.5)

    parser.add_argument("--save_dir", type=str, default=str(ROOT_DIR / "Argus" / "logs_csv"))
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--load_dir", type=str, default="")
    parser.add_argument("--load_pth", type=str, default="checkpoint.pth.tar")

    parser.add_argument("--initial_channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi_gpu", action="store_true")
    return parser.parse_args()


def build_model(input_dim: int, num_classes: int, initial_channels: int, device: torch.device, multi_gpu: bool) -> nn.Module:
    net = DeepResNet(input_size=input_dim, num_classes=num_classes, initial_channels=initial_channels)
    if multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    return net.to(device)


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_id(args.dataset)
    dataset_id, resolved_csv_path = resolve_dataset_and_csv_path(args.dataset, args.csv_path)
    resolved_cache_npz = (
        Path(args.cache_npz).resolve()
        if str(args.cache_npz).strip()
        else default_cache_npz_for_dataset(THIS_DIR / "cache", dataset_id, args.noise_type, args.noise, args.seed, args.test_size)
    )
    args.csv_path = str(resolved_csv_path)
    args.cache_npz = str(resolved_cache_npz)
    set_seed(args.seed)
    device = resolve_device(args.device)

    default_save_root = (ROOT_DIR / "Argus" / "logs_csv").resolve()
    requested_save_root = Path(args.save_dir).resolve()
    save_root = requested_save_root / dataset_id.lower() if requested_save_root == default_save_root else requested_save_root
    stage2_dir = save_root / f"{args.noise_type}_{int(args.noise*100)}_PL_cut{int(args.cut*100)}_Stage_2"
    legacy_stage2_dir = save_root / f"tlscsv_deepresnet_{args.noise_type}_{int(args.noise*100)}_PL_cut{int(args.cut*100)}"
    save_dir = Path(args.load_dir).resolve() if args.load_dir else save_root / f"{args.noise_type}_{int(args.noise*100)}_PL_cut{int(args.cut*100)}_Stage_3"
    ensure_dir(save_dir)
    logger = setup_logger(save_dir)
    logger.info(vars(args))

    pretrained = Path(args.pretrained).resolve() if args.pretrained else stage2_dir / "checkpoint.pth.tar"
    if not pretrained.exists() and not args.pretrained:
        pretrained = legacy_stage2_dir / "checkpoint.pth.tar"
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained}")

    bundle = build_or_load_csv_bundle(
        csv_path=resolved_csv_path,
        cache_npz=resolved_cache_npz,
        noise_rate=args.noise,
        noise_type=args.noise_type,
        seed=args.seed,
        test_size=args.test_size,
    )
    expected_ood_class_ids = resolve_ood_class_ids(resolved_csv_path, dataset=dataset_id)
    if not np.array_equal(bundle.ood_class_ids_original, expected_ood_class_ids):
        raise RuntimeError(
            f"OOD class split mismatch: expected {expected_ood_class_ids.tolist()}, got {bundle.ood_class_ids_original.tolist()}"
        )
    logger.info(
        "Class split policy: ood(original)=%s, known(original)=%s",
        bundle.ood_class_ids_original.tolist(),
        bundle.known_class_ids_original.tolist(),
    )
    logger.info(
        "Known remap pairs (original->remapped): %s",
        bundle.known_remap_pairs.tolist(),
    )
    logger.info("OOD sample count: %d", int(bundle.x_ood.shape[0]))

    train_y = bundle.y_train_noisy.copy().astype(np.int64)
    test_y = bundle.y_test.copy().astype(np.int64)
    num_classes = int(np.max(train_y)) + 1

    trainset = IndexedCSVDataset(bundle.x_train, train_y)
    testset = IndexedCSVDataset(bundle.x_test, test_y)
    trainloader = create_loader(trainset, args.batchSize, True, args.workers)
    testloader = create_loader(testset, args.batchSize, False, args.workers)

    net = build_model(bundle.x_train.shape[1], num_classes, args.initial_channels, device, args.multi_gpu)

    ckpt_pre = torch.load(pretrained, map_location=device)
    train_preds_hist = ckpt_pre["train_preds_hist"].cpu().float()
    noisy_idx = bundle.noisy_idx

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    load_model_state(net, ckpt_pre["state_dict"])

    num_hist = int(train_preds_hist.shape[1])
    n_train = len(trainset)
    if int(train_preds_hist.shape[0]) != n_train or int(train_preds_hist.shape[2]) != num_classes:
        raise RuntimeError(
            "pretrained train_preds_hist shape mismatch: "
            f"got {tuple(train_preds_hist.shape)}, expected ({n_train}, {num_hist}, {num_classes})"
        )
    pl_ratio = 0.0
    nl_ratio = 1.0

    epoch_resume = 0
    if args.load_dir:
        ckpt_path = Path(args.load_dir).resolve() / args.load_pth
        ckpt = torch.load(ckpt_path, map_location=device)
        load_model_state(net, ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        train_preds_hist = ckpt["train_preds_hist"].cpu()
        pl_ratio = float(ckpt.get("pl_ratio", 0.0))
        nl_ratio = float(ckpt.get("nl_ratio", 1.0))
        epoch_resume = int(ckpt.get("epoch", 0))
        logger.info("loading network SUCCESSFUL")
    else:
        logger.info("loading network FAILURE")

    best_test_acc = 0.0
    ce_cut_json_path = save_dir / "ce_cut_label_accuracy.json"
    ce_cut_label_accuracy_records = []
    if ce_cut_json_path.exists():
        with open(ce_cut_json_path, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        existing_records = existing_payload.get("epoch_accuracy", []) if isinstance(existing_payload, dict) else []
        ce_cut_label_accuracy_records = [
            {"epoch": int(item["epoch"]), "accuracy": float(item["accuracy"])}
            for item in existing_records
            if int(item.get("epoch", -1)) < epoch_resume
        ]
        if ce_cut_label_accuracy_records:
            logger.info("loaded %d existing ce-cut accuracy records", len(ce_cut_label_accuracy_records))

    for epoch in range(epoch_resume, args.max_epochs):
        do_eval = (((epoch + 1) % 20) == 0) or (epoch == args.max_epochs - 1)
        do_save = (epoch == args.max_epochs - 1)
        do_plot = do_eval

        preds = train_preds_hist.mean(1)[torch.arange(n_train), torch.from_numpy(train_y)].numpy()
        inds_filt = np.where(preds < args.cut)[0].astype(np.int64)
        inds_left = np.where(preds >= args.cut)[0].astype(np.int64)
        inds_filt_set = set(int(v) for v in inds_filt.tolist())
        nl_ratio = float(len(inds_filt)) / float(max(1, n_train))

        noisy_hit = len(np.intersect1d(inds_filt, noisy_idx))
        logger.info(
            "cut_select epoch=%d nl_ratio=%f noisy_recall=%f noisy_precision=%f ce_count=%d",
            epoch,
            nl_ratio,
            float(noisy_hit) / float(max(1, len(noisy_idx))),
            float(noisy_hit) / float(max(1, len(inds_filt))),
            int(len(inds_left)),
        )

        weight = class_weights_from_labels(train_y[inds_left], num_classes)
        weight = torch.where(torch.isinf(weight), torch.zeros_like(weight), weight).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        if epoch in args.epoch_step:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
                args.lr = g["lr"]

        train_loss = 0.0
        train_acc = 0.0

        train_preds = torch.full((n_train, num_classes), -1.0, dtype=torch.float32)

        net.train()
        for x, labels, index in trainloader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            index_cpu = index.cpu().numpy()

            optimizer.zero_grad(set_to_none=True)
            logits = net(x)

            pred = torch.argmax(logits.detach(), dim=-1)
            train_acc += float((pred == labels).sum().item())

            labels_masked = labels.clone()
            drop_local = [i for i, gidx in enumerate(index_cpu.tolist()) if gidx in inds_filt_set]
            if len(drop_local) > 0:
                labels_masked[torch.tensor(drop_local, device=device)] = -100

            valid_count = int((labels_masked >= 0).sum().item())
            if valid_count <= 0:
                train_preds[index.cpu()] = F.softmax(logits.detach(), dim=-1).cpu()
                continue

            loss = criterion(logits, labels_masked)
            loss.backward()
            optimizer.step()

            train_loss += float(x.size(0)) * float(loss.item())
            train_preds[index.cpu()] = F.softmax(logits.detach(), dim=-1).cpu()

        train_loss /= float(n_train)
        train_acc /= float(n_train)
        logger.info("[%6d/%6d] loss: %5f, acc: %5f, lr: %5f", epoch, args.max_epochs, train_loss, train_acc, args.lr)

        ce_cut_entry = record_ce_cut_label_accuracy_epoch(
            epoch=epoch,
            y_train_labels=train_y,
            y_train_clean=bundle.y_train_clean,
            cut_filtered_indices=inds_filt,
            records=ce_cut_label_accuracy_records,
            save_path=ce_cut_json_path,
        )
        logger.info("ce_cut_label_acc: epoch=%d, accuracy=%5f", ce_cut_entry["epoch"], ce_cut_entry["accuracy"])

        assert (train_preds < 0).sum().item() == 0
        train_preds_hist[:, epoch % num_hist] = train_preds

        train_soft_labels = train_preds_hist.mean(1)
        label_updated = torch.argmax(train_soft_labels, dim=-1).cpu().numpy()
        logger.info(
            "Label update status...noisy: %d/%d, clean: %d/%d",
            int((label_updated[bundle.noisy_idx] != bundle.y_train_clean[bundle.noisy_idx]).sum()),
            int(len(bundle.noisy_idx)),
            int((label_updated[bundle.clean_idx] != bundle.y_train_clean[bundle.clean_idx]).sum()),
            int(len(bundle.clean_idx)),
        )

        if do_eval:
            test_metrics = eval_test(net, testloader, criterion, device)
            logger.info(
                "\tTESTING...loss: %5f, Acc: %5f, F1_w: %5f, Recall_w: %5f, Precision_w: %5f, "
                "F1_m: %5f, Recall_m: %5f, Precision_m: %5f, best_acc: %5f",
                test_metrics["loss"],
                test_metrics["acc"],
                test_metrics["f1_weighted"],
                test_metrics["recall_weighted"],
                test_metrics["precision_weighted"],
                test_metrics["f1_macro"],
                test_metrics["recall_macro"],
                test_metrics["precision_macro"],
                best_test_acc,
            )

            is_best = test_metrics["acc"] > best_test_acc
            best_test_acc = max(test_metrics["acc"], best_test_acc)

            state = {
                "epoch": epoch,
                "state_dict": model_state_dict(net),
                "optimizer": optimizer.state_dict(),
                "train_preds_hist": train_preds_hist,
                "pl_ratio": pl_ratio,
                "nl_ratio": nl_ratio,
                "is_best": bool(is_best),
                "test_metrics": test_metrics,
                "csv_path": str(Path(args.csv_path).resolve()),
                "cache_npz": str(Path(args.cache_npz).resolve()),
                "noise": float(args.noise),
                "noise_type": str(args.noise_type),
                "test_size": float(args.test_size),
                "known_class_ids_original": bundle.known_class_ids_original.astype(int).tolist(),
                "ood_class_ids_original": bundle.ood_class_ids_original.astype(int).tolist(),
                "known_remap_pairs": bundle.known_remap_pairs.astype(int).tolist(),
            }

            if do_save:
                logger.info("saving model...")
                torch.save(state, save_dir / "checkpoint.pth.tar")

            if do_plot:
                logger.info("saving separated histogram...")
                prob_correct = prob_correct_from_hist(train_preds_hist, train_y)
                save_separated_histogram(prob_correct, bundle.clean_idx, bundle.noisy_idx, save_dir / f"histogram_sep_epoch{epoch:03d}.jpg")

    summary = {
        "stage": "Stage_3",
        "save_dir": str(save_dir),
        "best_test_acc": float(best_test_acc),
        "noise_rate_target": float(args.noise),
        "noise_rate_actual": float(np.mean(train_y != bundle.y_train_clean)),
        "num_train": int(n_train),
        "num_test": int(len(testset)),
        "known_class_ids_original": bundle.known_class_ids_original.astype(int).tolist(),
        "ood_class_ids_original": bundle.ood_class_ids_original.astype(int).tolist(),
        "known_remap_pairs": bundle.known_remap_pairs.astype(int).tolist(),
        "num_ood_total": int(bundle.x_ood.shape[0]),
    }
    save_json(summary, save_dir / "summary.json")


if __name__ == "__main__":
    main()
