"Episode training, target adaptation, evaluation, and checkpoint output."
from __future__ import annotations
import json
import random
from datetime import datetime, timedelta, timezone
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import torch
from torch.nn import functional as F
from data.datasets import PatchScene
from model.feature_mask import (
    ALL_EVIDENCE_METHODS,
    EVIDENCE_CONTRASTIVE_METHODS,
    EVIDENCE_VARIANTS,
    SPATIAL_SUPCON_ONLY_METHOD,
    FeatureMaskNet,
)
from training.storage import artifact_path

_log_stream = None


def configure_log(path):
    """Mirror all console progress for one invocation to one UTF-8 log file."""
    global _log_stream
    if _log_stream is not None:
        _log_stream.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_stream = path.open("w", encoding="utf-8", buffering=1)


def close_log():
    global _log_stream
    if _log_stream is not None:
        _log_stream.close()
        _log_stream = None


def log_info(message: str) -> None:
    """Print console progress with a timestamp and mirror it to the run log."""
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    line = f"[{timestamp}] [INFO] {message}"
    print(line, flush=True)
    if _log_stream is not None:
        print(line, file=_log_stream, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def supervised_mask_contrastive(
    full_features: torch.Tensor,
    masked_features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Align full and masked views using episode labels only."""
    features = F.normalize(torch.cat((full_features, masked_features), dim=0), dim=-1)
    repeated_labels = labels.repeat(2)
    logits = features @ features.T / temperature
    self_mask = torch.eye(len(repeated_labels), device=features.device, dtype=torch.bool)
    positives = (repeated_labels[:, None] == repeated_labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positives).sum(1).div(positives.sum(1).clamp_min(1)).mean()


def episode_loss(
    model,
    episode,
    domain,
    ways,
    method,
    mask_loss_weight=0.5,
    contrastive_weight=0.0,
):
    """Prototype CE with optional spatial evidence CE and masked supervised contrastive loss."""
    if method not in ("baseline", *ALL_EVIDENCE_METHODS):
        raise ValueError(f"Unsupported method: {method}")
    sx, sy, qx, qy = episode
    device = next(model.parameters()).device
    sx, sy, qx, qy = (sx.to(device), sy.to(device), qx.to(device), qy.to(device))
    if method == "baseline":
        z_s, z_q = (model.encode(sx, domain), model.encode(qx, domain))
    else:
        support_parts, query_parts = (
            model.encode_parts(sx, domain),
            model.encode_parts(qx, domain),
        )
        z_s = 0.5 * (support_parts[0] + support_parts[1])
        z_q = 0.5 * (query_parts[0] + query_parts[1])
    full = model.logits(z_s, sy, z_q, ways)
    loss = F.cross_entropy(full, qy)
    accuracy = (full.argmax(-1) == qy).float().mean().item()
    stats = {"full_loss": loss.item(), "full_acc": accuracy}
    if method == "baseline":
        return (loss, stats)
    if method == SPATIAL_SUPCON_ONLY_METHOD and mask_loss_weight != 0:
        raise ValueError("SupCon-only requires zero evidence weight")
    if method != SPATIAL_SUPCON_ONLY_METHOD:
        evidence = model.branch_evidence_logits(support_parts, sy, query_parts, ways, method)
        evidence_loss = F.cross_entropy(evidence, qy)
        loss = loss + mask_loss_weight * evidence_loss
        stats.update(
            evidence_loss=evidence_loss.item(),
            evidence_acc=(evidence.argmax(-1) == qy).float().mean().item(),
        )
    if method in EVIDENCE_CONTRASTIVE_METHODS:
        full_features, masked_features, labels = model.spatial_evidence_contrastive_views(
            support_parts, sy, query_parts, qy, ways
        )
        contrastive = supervised_mask_contrastive(full_features, masked_features, labels)
        loss = loss + contrastive_weight * contrastive
        stats.update(contrastive_loss=contrastive.item())
    return (loss, stats)


def target_episode(
    scene: PatchScene,
    support: np.ndarray,
    support_y: np.ndarray,
    ways: int,
    rng: np.random.Generator,
):
    """Split original labeled centers first; never train with a test center."""
    classes = rng.choice(scene.classes, ways, replace=False)
    s_coords, q_coords = ([], [])
    for c in classes:
        ids = np.flatnonzero(support_y == c)
        if len(ids) < 2:
            raise ValueError("Target adaptation needs at least two original labels per class")
        ids = rng.permutation(ids)
        s_coords.append(support[ids[:-1]])
        q_coords.append(support[ids[-1:]])
    s_coords, q_coords = (np.concatenate(s_coords), np.concatenate(q_coords))
    sy = torch.cat(
        [torch.full((len(support_y[support_y == c]) - 1,), i) for i, c in enumerate(classes)]
    ).long()
    qy = torch.arange(ways)
    return (scene.patches(s_coords), sy, scene.patches(q_coords), qy)


@torch.no_grad()
def evaluate(
    model: FeatureMaskNet,
    scene: PatchScene,
    support: np.ndarray,
    support_y: np.ndarray,
    test: np.ndarray,
    test_y: np.ndarray,
    batch_size: int,
    method: str = "baseline",
    protocol: str = "standard",
):
    model.eval()
    device = next(model.parameters()).device
    z_support = []
    for start in range(0, len(support), batch_size):
        patches = scene.patches(support[start : start + batch_size]).to(device)
        z_support.append(model.encode(patches, "target"))
    z_support = torch.cat(z_support)
    sy = torch.as_tensor(support_y, device=device)
    prototypes = torch.stack([z_support[sy == c].mean(0) for c in range(scene.classes)])
    prototypes = F.normalize(prototypes, dim=-1)
    pred = []
    for start in range(0, len(test), batch_size):
        patches = scene.patches(test[start : start + batch_size]).to(device)
        z = model.encode(patches, "target")
        scores = F.normalize(z, dim=-1) @ prototypes.T
        pred.append(scores.argmax(-1).cpu().numpy())
    pred = np.concatenate(pred)
    cm = confusion_matrix(test_y, pred, labels=np.arange(scene.classes))
    row_sum = cm.sum(axis=1)
    per_class = np.divide(
        cm.diagonal(), row_sum, out=np.full(scene.classes, np.nan), where=row_sum > 0
    )
    return {
        "OA": float(np.mean(pred == test_y)),
        "AA": float(np.nanmean(per_class)),
        "kappa": float(cohen_kappa_score(test_y, pred, labels=np.arange(scene.classes))),
        "per_class": per_class.tolist(),
        "test_count": int(len(test)),
    }


def log_training(name, method, seed, stage, step, total, stats, elapsed):
    avg = {key: sum((row[key] for row in stats)) / len(stats) for key in stats[0]}
    details = f"loss: {avg['loss']:.4f}, full_loss: {avg['full_loss']:.4f}, full_acc: {avg['full_acc']:.3f}"
    if (
        method
        in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
        and method != SPATIAL_SUPCON_ONLY_METHOD
    ):
        details += (
            f", evidence_loss: {avg['evidence_loss']:.4f}, evidence_acc: {avg['evidence_acc']:.3f}"
        )
    if method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only"):
        details += f", contrastive_loss: {avg['contrastive_loss']:.4f}"
    if "source_acc" in avg:
        details += f", source_acc: {avg['source_acc']:.3f}, target_acc: {avg['target_acc']:.3f}"
    log_info(
        f"episode: {total}, stage: {stage} ({step}), avg_last: {len(stats)}, {details}, elapsed: {elapsed:.0f}s"
    )


def log_evaluation(
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
    stage,
    step,
    total,
    progress_path,
    best,
    folder,
    stem,
    split_hash,
):
    adapted = True
    log_info(f"Testing... episode: {total}, stage: {stage}, target_adapted: {adapted}")
    metrics = evaluate(
        model,
        scene,
        support,
        support_y,
        test,
        test_y,
        args.batch_size,
        method,
        getattr(args, "protocol", "standard"),
    )
    correct = round(metrics["OA"] * metrics["test_count"])
    row = {
        "dataset": name,
        "method": method,
        "seed": seed,
        "stage": stage,
        "stage_episode": step,
        "total_episode": total,
        "adapted": adapted,
        **metrics,
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_info(
        f"\t\tAccuracy: {correct}/{metrics['test_count']} ({100 * metrics['OA']:.2f}%), AA: {100 * metrics['AA']:.2f}%, Kappa: {metrics['kappa']:.4f}"
    )
    if best["row"] is None or metrics["OA"] > best["row"]["OA"]:
        schedule = "joint"
        best_row = {
            **row,
            "shots": args.shots,
            "source_episodes": None,
            "target_episodes": None,
            "guard_radius": args.guard_radius,
            "split_hash": split_hash,
            "schedule": schedule,
            "joint_episodes": getattr(args, "joint_episodes", None),
            "target_weight": getattr(args, "target_weight", None),
            "lr_schedule": getattr(args, "lr_schedule", "constant"),
            "training_protocol": getattr(args, "training_protocol", "standard"),
        }
        best_row["mask_variant"] = (
            EVIDENCE_VARIANTS[method] if method in ALL_EVIDENCE_METHODS else None
        )
        best_row["mask_loss_weight"] = (
            args.mask_loss_weight
            if method
            in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive", "spatial_supcon_only")
            else None
        )
        best_row["contrastive_weight"] = (
            args.contrastive_weight
            if method in ("mask_evidence_spatial_contrastive", "spatial_supcon_only")
            else None
        )
        best["row"] = best_row
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": vars(args) | {"dataset": name, "method": method},
                "support_centers": support,
                "support_labels": support_y,
                "selected_episode": total,
                "selected_stage": stage,
            },
            artifact_path(folder, stem, "_best", ".pt"),
        )
        artifact_path(folder, stem, "_best", ".json").write_text(
            json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_info(
            f"Best checkpoint saved: episode={total}, stage={stage}, acc={100 * metrics['OA']:.2f}%"
        )
    log_info(
        f"best episode:[{best['row']['total_episode']}], best accuracy={100 * best['row']['OA']:.14f}%, stage={best['row']['stage']}"
    )
    return metrics
