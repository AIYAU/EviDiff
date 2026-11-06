"Evidence-mask experiments on IP, SA and UP."
from __future__ import annotations
import argparse
import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data.datasets import DATASETS, PatchScene, SourcePatches
from model.feature_mask import (
    ALL_EVIDENCE_METHODS,
    EVIDENCE_VARIANTS,
    SPATIAL_SUPCON_ONLY_METHOD,
    FeatureMaskNet,
)
from training.engine import close_log, configure_log, log_info
from training.joint import train_one_joint
from training.storage import existing_artifact_path

ROOT = Path(__file__).resolve().parent / "datasets"
SEEDS = [1236, 1237, 1226, 1227, 1211, 1212, 1216, 1240, 1222, 1223]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="IP")
    parser.add_argument(
        "--method",
        choices=["baseline", *ALL_EVIDENCE_METHODS],
        default="baseline",
        help="Select baseline, spatial evidence, SupCon, or their combination",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild comparison.json from saved runs without training",
    )
    parser.add_argument(
        "--classification-map",
        action="store_true",
        help="Save a classification map from the best seed/checkpoint",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Generate the classification map from completed runs without retraining",
    )
    parser.add_argument("--schedule", choices=["joint"], default="joint")
    parser.add_argument(
        "--protocol", choices=["standard"], default="standard", help="Standard episodic protocol"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help="Read matching baseline results from this experiment root; candidate results are still saved under --output",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="One timestamped console log per training invocation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--source-shots", type=int, default=1)
    parser.add_argument("--source-queries", type=int, default=5)
    parser.add_argument("--joint-episodes", type=int, default=5000)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--mapped-bands", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--mask-loss-weight", type=float, default=None, help="Spatial evidence-difference CE weight"
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=None,
        help="Masked supervised contrastive weight",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--guard-radius", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Evaluate target OA/AA/Kappa every N episodes"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.training_protocol = "standard"
    if args.method == SPATIAL_SUPCON_ONLY_METHOD:
        if args.mask_loss_weight not in (None, 0):
            parser.error("SupCon-only requires zero evidence loss weight")
        args.mask_loss_weight = 0.0
        args.contrastive_weight = (
            1.5 if args.contrastive_weight is None else args.contrastive_weight
        )
    elif args.method == "mask_evidence_spatial_contrastive":
        args.mask_loss_weight = 0.5 if args.mask_loss_weight is None else args.mask_loss_weight
        args.contrastive_weight = (
            0.1 if args.contrastive_weight is None else args.contrastive_weight
        )
    else:
        if args.contrastive_weight not in (None, 0):
            parser.error("This method does not include supervised contrastive loss")
        args.contrastive_weight = 0.0
        args.mask_loss_weight = (
            (0.5 if args.method == "mask_evidence_spatial" else 0.0)
            if args.mask_loss_weight is None
            else args.mask_loss_weight
        )
    if args.output is None:
        folder = "experiments_joint_5000"
        default_weight = (
            0
            if args.method == SPATIAL_SUPCON_ONLY_METHOD
            else 0
            if args.method == "baseline"
            else 0.5
        )
        if args.mask_loss_weight != default_weight:
            folder += f"_maskweight{args.mask_loss_weight:g}"
        args.output = Path(__file__).resolve().parent / folder
    if args.patch_size != 7:
        parser.error("Bundled source patches require patch-size 7")
    if args.shots < 2 or args.ways < 2 or args.source_shots < 1 or (args.source_queries < 1):
        parser.error("Require shots >= 2, ways >= 2 and positive source shots/queries")
    if args.joint_episodes < 1 or args.target_weight <= 0:
        parser.error("Invalid training budget or target weight")
    if args.mask_loss_weight < 0 or args.contrastive_weight < 0:
        parser.error("Loss weights must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds")
    if args.log_every < 1 or args.eval_every < 1:
        parser.error("Logging intervals must be positive")
    if args.map_only and (not args.classification_map):
        parser.error("map-only requires classification-map")
    return args


def read_runs(args, dataset, method, suffix=""):
    baseline_output = getattr(args, "baseline_output", None)
    root = baseline_output if method == "baseline" and baseline_output is not None else args.output
    folder = root / dataset / method
    rows = []
    for seed in args.seeds:
        stem = f"{args.shots}shot_seed{seed}_guard{args.guard_radius}"
        path = existing_artifact_path(folder, stem, suffix, ".json")
        if not path.exists():
            continue
        final_path = existing_artifact_path(folder, stem, "", ".json")
        expected = args.joint_episodes
        complete = False
        if final_path.exists():
            try:
                final_row = json.loads(final_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                final_row = {}
            recorded_episodes = final_row.get("joint_episodes")
            complete = (
                recorded_episodes == expected
                and final_row.get("dataset") == dataset
                and (final_row.get("method") == method)
                and (final_row.get("seed") == seed)
                and (final_row.get("shots") == args.shots)
                and all((key in final_row for key in ("OA", "AA", "kappa")))
            )
        if not complete:
            if not suffix:
                log_info(f"Skipping unfinished or stale run: {path}")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("training_protocol", "standard") != args.training_protocol:
            raise ValueError(f"Existing run has a different training protocol: {path}")
        if (
            method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only")
            and row.get("contrastive_weight") != args.contrastive_weight
        ):
            raise ValueError(f"Existing run has a different contrastive weight: {path}")
        if method in (
            "mask_evidence_spatial",
            "mask_evidence_spatial_contrastive",
            "spatial_supcon_only",
        ) and (row.get("mask_loss_weight"), row.get("mask_variant")) != (
            args.mask_loss_weight,
            EVIDENCE_VARIANTS[method],
        ):
            raise ValueError(f"Existing run has different evidence-mask settings: {path}")
        if (
            row.get("schedule"),
            row.get("joint_episodes"),
            row.get("target_weight"),
            row.get("lr_schedule", "constant"),
        ) != ("joint", args.joint_episodes, args.target_weight, args.lr_schedule):
            raise ValueError(f"Existing run has a different joint schedule: {path}")
        rows.append(row)
    return rows


def summarize(rows):
    report = {"seeds": [r["seed"] for r in rows], "n": len(rows)}
    for metric in ("OA", "AA"):
        values = [r[metric] for r in rows]
        report[metric + "_percent"] = {
            "mean": float(100 * np.mean(values)),
            "std": float(100 * np.std(values)),
        }
    kappa_values = [r["kappa"] for r in rows]
    report["kappa"] = {"mean": float(np.mean(kappa_values)), "std": float(np.std(kappa_values))}
    report["kappa_percent"] = {
        "mean": 100 * report["kappa"]["mean"],
        "std": 100 * report["kappa"]["std"],
    }
    report["per_seed"] = [
        {
            "seed": r["seed"],
            "episode": r.get("total_episode", r.get("joint_episodes")),
            "OA_percent": 100 * r["OA"],
            "AA_percent": 100 * r["AA"],
            "kappa_percent": 100 * r["kappa"],
        }
        for r in rows
    ]
    if all(("per_class" in r for r in rows)):
        per_class = np.asarray([r["per_class"] for r in rows], dtype=float)
        report["per_class_percent"] = [
            {
                "class": i,
                "mean": float(100 * np.nanmean(per_class[:, i])),
                "std": float(100 * np.nanstd(per_class[:, i])),
            }
            for i in range(per_class.shape[1])
        ]
    return report


def paired_summary(args, base, mask):
    by_seed_b = {r["seed"]: r for r in base}
    by_seed_m = {r["seed"]: r for r in mask}
    matched = [s for s in args.seeds if s in by_seed_b and s in by_seed_m]
    if not matched:
        return None
    for seed in matched:
        if by_seed_b[seed]["split_hash"] != by_seed_m[seed]["split_hash"]:
            raise ValueError(f"Support/test split differs between methods for seed {seed}")
    return {
        "seeds": matched,
        **{
            metric
            + "_pp": {
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
                "per_seed": [float(x) for x in deltas],
            }
            for metric in ("OA", "AA", "kappa")
            for deltas in [[100 * (by_seed_m[s][metric] - by_seed_b[s][metric]) for s in matched]]
        },
    }


COMMON_CLASS_COLORS = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.65, 0.35, 1.0],
        [0.75, 0.5, 0.75],
        [0.75, 1.0, 0.5],
        [0.5, 1.0, 0.65],
        [0.65, 0.65, 0.0],
        [0.75, 1.0, 0.65],
        [0.0, 0.0, 0.5],
        [0.0, 1.0, 0.75],
        [0.5, 0.75, 1.0],
    ],
    dtype=np.float32,
)
def _load_torch_checkpoint(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def save_classification_map(args, dataset: str, method: str) -> Path:
    """Render the highest-OA seed/checkpoint with the exact reported predictor."""
    rows = read_runs(args, dataset, method, "_best")
    if len(rows) != len(args.seeds):
        completed = sorted((row["seed"] for row in rows))
        raise RuntimeError(
            f"Classification map needs all {len(args.seeds)} completed {dataset} {method} seeds; found {completed}"
        )
    best = max(rows, key=lambda row: row["OA"])
    seed = int(best["seed"])
    stem = f"{args.shots}shot_seed{seed}_guard{args.guard_radius}"
    folder = args.output / dataset / method
    checkpoint_path = existing_artifact_path(folder, stem, "_best", ".pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")
    scene = PatchScene(args.data_root, dataset, args.patch_size)
    support, support_y, test, test_y = scene.split(
        args.shots, np.random.RandomState(seed), args.guard_radius
    )
    checkpoint = _load_torch_checkpoint(checkpoint_path, args.device)
    saved_support = np.asarray(checkpoint["support_centers"])
    saved_labels = np.asarray(checkpoint["support_labels"])
    if not np.array_equal(saved_support, support) or not np.array_equal(saved_labels, support_y):
        raise ValueError(f"Saved support split does not match dataset split: {checkpoint_path}")
    config = checkpoint.get("config", {})
    model = FeatureMaskNet(
        128,
        scene.bands,
        int(config.get("mapped_bands", args.mapped_bands)),
        int(config.get("dim", args.dim)),
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    support_features = []
    for start in range(0, len(support), args.batch_size):
        patches = scene.patches(support[start : start + args.batch_size]).to(args.device)
        support_features.append(model.encode(patches, "target"))
    support_features = torch.cat(support_features)
    labels = torch.as_tensor(support_y, device=args.device)
    prototypes = torch.stack(
        [support_features[labels == cls].mean(0) for cls in range(scene.classes)]
    )
    prototypes = F.normalize(prototypes, dim=-1)
    predictions = []
    for start in range(0, len(test), args.batch_size):
        patches = scene.patches(test[start : start + args.batch_size]).to(args.device)
        features = model.encode(patches, "target")
        scores = F.normalize(features, dim=-1) @ prototypes.T
        predictions.append(scores.argmax(-1).cpu().numpy())
    predictions = np.concatenate(predictions)
    reproduced_oa = float(np.mean(predictions == test_y))
    if not np.isclose(reproduced_oa, float(best["OA"]), atol=5e-08):
        raise RuntimeError(
            f"Classification-map OA {reproduced_oa:.8f} does not match saved OA {float(best['OA']):.8f}"
        )
    label_map = np.zeros(scene.shape, dtype=np.uint8)
    label_map[support[:, 0], support[:, 1]] = support_y.astype(np.uint8) + 1
    label_map[test[:, 0], test[:, 1]] = predictions.astype(np.uint8) + 1
    class_colors = COMMON_CLASS_COLORS
    if scene.classes > len(class_colors):
        raise ValueError(f"No classification-map palette for {scene.classes} classes")
    palette = np.vstack((np.zeros((1, 3), dtype=np.float32), class_colors[: scene.classes]))
    color_map = palette[label_map]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Classification maps require matplotlib>=3.7") from error
    map_dir = folder / "classification_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}_{method}_{args.shots}shot"
    image_path = map_dir / f"{prefix}.png"
    label_path = map_dir / f"{prefix}_labels.npy"
    metadata_path = map_dir / f"{prefix}.json"
    dpi = 24
    figure = plt.figure(frameon=False)
    figure.set_size_inches(scene.shape[1] * 2.0 / dpi, scene.shape[0] * 2.0 / dpi)
    axis = plt.Axes(figure, [0.0, 0.0, 1.0, 1.0])
    axis.set_axis_off()
    figure.add_axes(axis)
    axis.imshow(color_map, interpolation="nearest")
    figure.savefig(image_path, dpi=dpi)
    plt.close(figure)
    np.save(label_path, label_map)
    metadata = {
        "dataset": dataset,
        "method": method,
        "shots": args.shots,
        "selected_seed": seed,
        "selected_episode": int(best["total_episode"]),
        "OA": reproduced_oa,
        "AA": float(best["AA"]),
        "kappa": float(best["kappa"]),
        "checkpoint": str(checkpoint_path),
        "prediction": "complete_feature_prototype_classifier",
        "background_label": 0,
        "class_labels": "1-based",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log_info(
        f"Classification map saved: {image_path}; best seed: {seed}, episode: {best['total_episode']}, OA: {100 * reproduced_oa:.2f}%"
    )
    return image_path


def write_reports(args, dataset):
    candidate = args.method if args.method != "baseline" else None
    reference_method = (
        "mask_evidence_spatial"
        if candidate != SPATIAL_SUPCON_ONLY_METHOD
        and candidate in ("mask_evidence_spatial_contrastive", "spatial_supcon_only")
        else None
    )
    base = read_runs(args, dataset, "baseline")
    mask = read_runs(args, dataset, candidate) if candidate else []
    base_best = read_runs(args, dataset, "baseline", "_best")
    mask_best = read_runs(args, dataset, candidate, "_best") if candidate else []
    reference = read_runs(args, dataset, reference_method) if reference_method else []
    reference_best = read_runs(args, dataset, reference_method, "_best") if reference_method else []
    report = {}
    if base:
        report["baseline"] = summarize(base)
    if mask:
        report[candidate] = summarize(mask)
    if base_best:
        report["baseline_best"] = summarize(base_best)
    if mask_best:
        report[f"{candidate}_best"] = summarize(mask_best)
    paired = paired_summary(args, base, mask)
    if paired:
        report[f"paired_{candidate}_minus_baseline"] = paired
    paired_best = paired_summary(args, base_best, mask_best)
    if paired_best:
        report[f"paired_best_{candidate}_minus_baseline"] = paired_best
    paired_reference = paired_summary(args, reference, mask)
    paired_reference_best = paired_summary(args, reference_best, mask_best)
    if paired_reference:
        report[f"paired_{candidate}_minus_{reference_method}"] = paired_reference
    if paired_reference_best:
        report[f"paired_best_{candidate}_minus_{reference_method}"] = paired_reference_best
    reported = {
        "selection": "highest_target_test_OA_among_evaluated_episodes",
        "aggregation": "mean_of_per_seed_best_OA",
        "eval_every": args.eval_every,
        "test_selected": True,
    }
    for method in ("baseline", *([candidate] if candidate else [])):
        if f"{method}_best" in report:
            reported[method] = report[f"{method}_best"]
    if paired_best:
        reported[f"paired_{candidate}_minus_baseline"] = paired_best
    report["reported"] = reported
    folder = args.output / dataset
    folder.mkdir(parents=True, exist_ok=True)
    comparison_path = folder / (
        f"comparison_{candidate}.json"
        if candidate
        in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
        else "comparison.json"
    )
    comparison_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for method in ("baseline", *([candidate] if candidate else [])):
        for label, key in (
            ("REPORTED mean of per-seed highest-OA checkpoints", f"{method}_best"),
            ("fixed final-episode diagnostic", method),
        ):
            if key in report:
                summary = report[key]
                if key.endswith("_best"):
                    log_info("*" * 72)
                log_info(
                    f"{dataset} {method} {label}: {summary['n']} seeds, OA: {summary['OA_percent']['mean']:.2f}% +/- {summary['OA_percent']['std']:.2f}%, AA: {summary['AA_percent']['mean']:.2f}% +/- {summary['AA_percent']['std']:.2f}%, Kappa (x100): {summary['kappa_percent']['mean']:.4f} +- {summary['kappa_percent']['std']:.4f}"
                )
                if key.endswith("_best"):
                    log_info("accuracy list (seed, best episode, OA%):")
                    for row in summary["per_seed"]:
                        log_info(
                            f"seed {row['seed']}: episode {row['episode']}, OA {row['OA_percent']:.4f}%"
                        )
                    if "per_class_percent" in summary:
                        log_info("accuracy for each class (mean +- std, %):")
                        for row in summary["per_class_percent"]:
                            log_info(f"Class {row['class']}: {row['mean']:.2f} +- {row['std']:.2f}")
                    log_info("*" * 72)
    if paired_best:
        delta = paired_best["OA_pp"]
        pair = paired_best
        log_info(
            f"{dataset} paired highest-OA checkpoint {candidate} - baseline: OA {delta['mean']:+.2f} pp, AA {pair['AA_pp']['mean']:+.2f} pp, Kappa {pair['kappa_pp']['mean'] / 100:+.4f} across {len(pair['seeds'])} seeds"
        )
    if paired:
        delta = paired["OA_pp"]
        pair = paired
        log_info(
            f"{dataset} paired final-episode {candidate} - baseline: OA {delta['mean']:+.2f} pp, AA {pair['AA_pp']['mean']:+.2f} pp, Kappa {pair['kappa_pp']['mean'] / 100:+.4f} across {len(pair['seeds'])} seeds"
        )
    if paired_reference_best:
        pair = paired_reference_best
        log_info(
            f"{dataset} paired highest-OA checkpoint {candidate} - {reference_method}: OA {pair['OA_pp']['mean']:+.2f} pp, AA {pair['AA_pp']['mean']:+.2f} pp, Kappa {pair['kappa_pp']['mean'] / 100:+.4f} across {len(pair['seeds'])} seeds"
        )
    if paired_reference:
        pair = paired_reference
        log_info(
            f"{dataset} paired final-episode {candidate} - {reference_method}: OA {pair['OA_pp']['mean']:+.2f} pp, AA {pair['AA_pp']['mean']:+.2f} pp, Kappa {pair['kappa_pp']['mean'] / 100:+.4f} across {len(pair['seeds'])} seeds"
        )
    log_info(f"Comparison saved: {comparison_path}")


def main():
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if args.report_only:
        for name in names:
            write_reports(args, name)
            if args.classification_map:
                save_classification_map(args, name, args.method)
        return
    now = datetime.now(timezone(timedelta(hours=8)))
    log_dir = args.log_dir / f"{args.ways}way_{args.shots}shot_{args.dataset}"
    log_path = log_dir / f"{args.method}_{len(args.seeds)}seeds_{now:%Y-%m-%d_%H-%M-%S-%f}.log"
    configure_log(log_path)
    try:
        log_info(f"Run log: {log_path}")
        if args.map_only:
            for name in names:
                save_classification_map(args, name, args.method)
            return
        source = SourcePatches(args.data_root)
        log_info(
            f"Plan: {len(args.seeds)} seeds, {args.joint_episodes} joint source + target episodes per method and seed"
        )
        for name in names:
            for seed in args.seeds:
                train_one_joint(args, name, seed, source, args.method)
            write_reports(args, name)
            if args.classification_map:
                save_classification_map(args, name, args.method)
    except BaseException:
        log_info("Run interrupted or failed:\n" + traceback.format_exc())
        raise
    finally:
        close_log()


if __name__ == "__main__":
    main()
