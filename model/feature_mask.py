"Feature extractor, Support-derived evidence masks and prototype classification."
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from model.feature_extractor import FeatureExtractor
from model.mapping import Mapping

SPATIAL_SUPCON_ONLY_METHOD = "spatial_supcon_only"
SPATIAL_EVIDENCE_CONTRASTIVE_METHOD = "mask_evidence_spatial_contrastive"
ALL_EVIDENCE_METHODS = (
    "mask_evidence_spatial",
    SPATIAL_EVIDENCE_CONTRASTIVE_METHOD,
    SPATIAL_SUPCON_ONLY_METHOD,
)
EVIDENCE_CONTRASTIVE_METHODS = (SPATIAL_EVIDENCE_CONTRASTIVE_METHOD, SPATIAL_SUPCON_ONLY_METHOD)
EVIDENCE_VARIANTS = {
    "mask_evidence_spatial": "support_counterfactual_spatial_branch_evidence_v1",
    SPATIAL_EVIDENCE_CONTRASTIVE_METHOD: "support_counterfactual_spatial_branch_evidence_supcon_v1",
    SPATIAL_SUPCON_ONLY_METHOD: "support_spatial_mask_supcon_only_v1",
}


class FeatureMaskNet(nn.Module):
    def __init__(
        self,
        source_bands: int,
        target_bands: int,
        mapped_bands: int = 100,
        dim: int = 128,
        scale: float = 10.0,
    ):
        super().__init__()
        self.source_map = Mapping(source_bands, mapped_bands)
        self.target_map = Mapping(target_bands, mapped_bands)
        self.feature_extractor = FeatureExtractor(mapped_bands, dim)
        self.scale = scale

    def encode(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        mapping = self.source_map if domain == "source" else self.target_map
        return self.feature_extractor(mapping(x))

    def encode_parts(self, x: torch.Tensor, domain: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract both branches for masking before the baseline fusion."""
        mapping = self.source_map if domain == "source" else self.target_map
        return self.feature_extractor.forward_parts(mapping(x))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs
    ):
        """Accept checkpoints produced before the extractor was given a generic name."""
        legacy_prefix = prefix + "ss" + "fe."
        current_prefix = prefix + "feature_extractor."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                replacement = current_prefix + key[len(legacy_prefix):]
                state_dict.setdefault(replacement, state_dict[key])
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def logits(
        self, support: torch.Tensor, support_y: torch.Tensor, query: torch.Tensor, ways: int
    ):
        prototypes = torch.stack([support[support_y == c].mean(0) for c in range(ways)])
        return self.scale * F.normalize(query, dim=-1) @ F.normalize(prototypes, dim=-1).T

    def evidence_masks(
        self, support: torch.Tensor, support_y: torch.Tensor, ways: int, fraction: float = 0.25
    ) -> torch.Tensor:
        """Select each class's strongest support-derived feature channels."""
        with torch.no_grad():
            prototypes = torch.stack([support[support_y == c].mean(0) for c in range(ways)])
            within = torch.stack(
                [(support[support_y == c] - prototypes[c]).square().mean(0) for c in range(ways)]
            )
            distances = (prototypes[:, None] - prototypes[None, :]).square()
            other = ~torch.eye(ways, device=support.device, dtype=torch.bool)
            between = distances.masked_fill(~other[:, :, None], float("inf")).min(1).values
            reliability = between / (within + 0.25 * within.mean(0, keepdim=True) + 0.0001)
            count = max(1, min(support.shape[-1] - 1, round(fraction * support.shape[-1])))
            masks = torch.zeros_like(reliability)
            masks.scatter_(1, reliability.topk(count, dim=1).indices, 1.0)
            return masks

    def branch_evidence_logits(
        self,
        support_parts: tuple[torch.Tensor, torch.Tensor],
        support_y: torch.Tensor,
        query_parts: tuple[torch.Tensor, torch.Tensor],
        ways: int,
        method: str,
    ) -> torch.Tensor:
        """Class-specific spatial evidence difference with the spectral branch unchanged."""
        if method not in ("mask_evidence_spatial", "mask_evidence_spatial_contrastive"):
            raise ValueError(f"Unknown branch evidence method: {method}")
        spectral_s, spatial_s = support_parts
        spectral_q, spatial_q = query_parts
        spectral_p = torch.stack([spectral_s[support_y == c].mean(0) for c in range(ways)])
        spatial_p = torch.stack([spatial_s[support_y == c].mean(0) for c in range(ways)])
        spectral_mask = torch.ones_like(spectral_p)
        spatial_mask = self.evidence_masks(spatial_s, support_y, ways)

        def fused_scores(spectral_keep: torch.Tensor, spatial_keep: torch.Tensor) -> torch.Tensor:
            prototypes = 0.5 * (spectral_p * spectral_keep + spatial_p * spatial_keep)
            queries = 0.5 * (
                spectral_q[:, None, :] * spectral_keep[None, :, :]
                + spatial_q[:, None, :] * spatial_keep[None, :, :]
            )
            prototypes = F.normalize(prototypes, dim=-1)
            queries = F.normalize(queries, dim=-1)
            return (queries * prototypes[None, :, :]).sum(-1)

        evidence_score = fused_scores(spectral_mask, spatial_mask)
        spectral_non = spectral_mask
        spatial_non = 1.0 - spatial_mask
        non_evidence_score = fused_scores(spectral_non, spatial_non)
        return self.scale * (evidence_score - non_evidence_score)

    def spatial_evidence_contrastive_views(
        self,
        support_parts: tuple[torch.Tensor, torch.Tensor],
        support_y: torch.Tensor,
        query_parts: tuple[torch.Tensor, torch.Tensor],
        query_y: torch.Tensor,
        ways: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build full and support-guided spatial-evidence views for SupCon.

        Evidence masks are estimated only from labeled support features. During
        episodic training, labels select the corresponding class mask for each
        support/query feature. Inference does not call this auxiliary branch.
        """
        spectral_s, spatial_s = support_parts
        spectral_q, spatial_q = query_parts
        masks = self.evidence_masks(spatial_s, support_y, ways)
        full_s = 0.5 * (spectral_s + spatial_s)
        full_q = 0.5 * (spectral_q + spatial_q)
        masked_s = 0.5 * (spectral_s + spatial_s * masks[support_y])
        masked_q = 0.5 * (spectral_q + spatial_q * masks[query_y])
        labels = torch.cat((support_y, query_y))
        return (torch.cat((full_s, full_q)), torch.cat((masked_s, masked_q)), labels)
