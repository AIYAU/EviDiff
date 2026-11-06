"Joint source/target episodic training for the feature-mask model."
from __future__ import annotations
import hashlib
import json
import time
import numpy as np
import torch
from data.datasets import PatchScene, SourcePatches
from model.feature_mask import EVIDENCE_VARIANTS, SPATIAL_SUPCON_ONLY_METHOD, FeatureMaskNet
from training.engine import (
    episode_loss,
    log_evaluation,
    log_info,
    log_training,
    seed_everything,
    target_episode,
)
from training.storage import artifact_path, prepare_artifact_dirs


def train_one_joint(args, name: str, seed: int, source: SourcePatches, method: str):
    if args.shots < 2:
        raise ValueError("Joint training needs at least two target labels per class")
    seed_everything(seed)
    split_rng = np.random.RandomState(seed)
    source_rng = np.random.default_rng(seed + 10000)
    target_rng = np.random.default_rng(seed + 20000)
    scene = PatchScene(args.data_root, name, args.patch_size)
    support, support_y, test, test_y = scene.split(args.shots, split_rng, args.guard_radius)
    split_hash = hashlib.sha256(support.tobytes() + test.tobytes()).hexdigest()[:16]
    model = FeatureMaskNet(
        128, scene.bands, args.mapped_bands, args.dim
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.joint_episodes, eta_min=args.lr * 0.01
        )
        if args.lr_schedule == "cosine"
        else None
    )
    folder = args.output / name / method
    prepare_artifact_dirs(folder)
    stem = f"{args.shots}shot_seed{seed}_guard{args.guard_radius}"
    progress_path = artifact_path(folder, stem, "_progress", ".jsonl")
    progress_path.write_text("", encoding="utf-8")
    best = {"row": None}
    window = []
    start_time = time.monotonic()
    log_info(f"dataset: {name}, method: {method}, seed: {seed}, shots: {args.shots}")
    log_info(
        f"Training... joint source + target episodes: {args.joint_episodes}, target_weight: {args.target_weight}, lr_schedule: {args.lr_schedule}"
    )
    if method in (
        "mask_evidence_spatial",
        "mask_evidence_spatial_contrastive",
        "spatial_supcon_only",
    ):
        location = {
            SPATIAL_SUPCON_ONLY_METHOD: "spatial masked view for supervised contrastive only",
            "mask_evidence_spatial": "spatial branch before fusion",
            "mask_evidence_spatial_contrastive": "spatial branch before fusion + supervised contrastive auxiliary",
        }[method]
        evidence_name = "evidence-difference CE"
        objective = f"full CE + {args.mask_loss_weight:g} * {evidence_name}" + (
            f" + {args.contrastive_weight:g} * supervised contrastive"
            if method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only")
            else ""
        )
        if method == SPATIAL_SUPCON_ONLY_METHOD:
            objective = f"full CE + {args.contrastive_weight:g} * supervised contrastive"
        inference = "full fused features"
        log_info(
            f"Mask: {EVIDENCE_VARIANTS[method]}, location: {location}, selector: support-only top-25% evidence, objective: {objective}, inference: {inference}"
        )
    model.train()
    for step in range(args.joint_episodes):
        source_task = source.episode(args.ways, args.source_shots, args.source_queries, source_rng)
        target_task = target_episode(
            scene, support, support_y, min(args.ways, scene.classes), target_rng
        )
        optimizer.zero_grad()
        source_loss, source_stats = episode_loss(
            model,
            source_task,
            "source",
            args.ways,
            method,
            args.mask_loss_weight,
            args.contrastive_weight,
        )
        target_loss, target_stats = episode_loss(
            model,
            target_task,
            "target",
            min(args.ways, scene.classes),
            method,
            args.mask_loss_weight,
            args.contrastive_weight,
        )
        loss = source_loss + args.target_weight * target_loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        stats = {
            "loss": loss.item(),
            "full_loss": source_stats["full_loss"] + args.target_weight * target_stats["full_loss"],
            "full_acc": (source_stats["full_acc"] + target_stats["full_acc"]) / 2,
            "source_acc": source_stats["full_acc"],
            "target_acc": target_stats["full_acc"],
        }
        if (
            method
            in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
            and method != SPATIAL_SUPCON_ONLY_METHOD
        ):
            stats.update(
                evidence_loss=source_stats["evidence_loss"]
                + args.target_weight * target_stats["evidence_loss"],
                evidence_acc=(source_stats["evidence_acc"] + target_stats["evidence_acc"]) / 2,
            )
        if method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only"):
            stats["contrastive_loss"] = (
                source_stats["contrastive_loss"]
                + args.target_weight * target_stats["contrastive_loss"]
            )
        window.append(stats)
        total = step + 1
        if total % args.log_every == 0 or total == args.joint_episodes:
            log_training(
                name, method, seed, "joint", total, total, window, time.monotonic() - start_time
            )
            window.clear()
        if total == 1 or total % args.eval_every == 0 or total == args.joint_episodes:
            result = log_evaluation(
                model,
                scene,
                support,
                support_y,
                test,
                test_y,
                args,
                name,
                seed,
                method,
                "joint",
                total,
                total,
                progress_path,
                best,
                folder,
                stem,
                split_hash,
            )
            model.train()
    result.update(
        dataset=name,
        method=method,
        seed=seed,
        shots=args.shots,
        mask_variant=EVIDENCE_VARIANTS[method]
        if method
        in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
        else None,
        mask_loss_weight=args.mask_loss_weight
        if method
        in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
        else None,
        contrastive_weight=args.contrastive_weight
        if method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only")
        else None,
        schedule="joint",
        joint_episodes=args.joint_episodes,
        target_weight=args.target_weight,
        lr_schedule=args.lr_schedule,
        source_episodes=None,
        target_episodes=None,
        guard_radius=args.guard_radius,
        split_hash=split_hash,
        best_episode=best["row"]["total_episode"],
        best_OA=best["row"]["OA"],
        reported_episode=best["row"]["total_episode"],
        reported_OA=best["row"]["OA"],
        reported_AA=best["row"]["AA"],
        reported_kappa=best["row"]["kappa"],
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args) | {"dataset": name, "method": method},
            "support_centers": support,
            "support_labels": support_y,
        },
        artifact_path(folder, stem, "", ".pt"),
    )
    result_path = artifact_path(folder, stem, "", ".json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_info(
        f"Reported seed result (highest observed): episode: {best['row']['total_episode']}, OA: {100 * best['row']['OA']:.2f}%, AA: {100 * best['row']['AA']:.2f}%, Kappa: {best['row']['kappa']:.4f}"
    )
    log_info(
        f"Fixed final-episode diagnostic: OA: {100 * result['OA']:.2f}%, AA: {100 * result['AA']:.2f}%, Kappa: {result['kappa']:.4f}"
    )
    log_info(f"Saved result: {result_path}")
    return result
