"""IP spatial evidence mask with supervised contrastive auxiliary loss."""
from pathlib import Path
import runpy
import sys

ENTRY = Path(__file__).with_name("feature_mask_hsi.py")
sys.argv = [str(ENTRY), "--dataset", "IP",
            "--method", "mask_evidence_spatial_contrastive",
            "--protocol", "standard", "--schedule", "joint",
            "--shots", "5", "--ways", "5", "--joint-episodes", "5000",
            "--target-weight", "1.0",
            "--lr-schedule", "cosine", "--eval-every", "500",
            "--mask-loss-weight", "1.5",
            "--contrastive-weight", "1.5",
            "--classification-map"] + sys.argv[1:]
runpy.run_path(str(ENTRY), run_name="__main__")
