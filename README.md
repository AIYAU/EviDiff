# EviDiff

**Evidence-Difference Learning for Robust Cross-Domain Few-Shot Hyperspectral Image Classification**

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- scikit-learn >= 1.3
- Matplotlib >= 3.7

```bash
pip install -r requirements.txt
```

For GPU training, use a PyTorch build compatible with your CUDA environment.

## Datasets

- Source domain: Chikusei.
- Target domains: Indian Pines (IP), Salinas (SA), and University of Pavia (UP).

Prepare the datasets in the following structure:

```text
datasets/
├── Chikusei_imdb_128_7_7.pickle
├── IP/
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── salinas/
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── UP/
    └── paviaU/
        ├── PaviaU.mat
        └── PaviaU_gt.mat
```

## Usage

Run the following commands from the repository root.

Train and evaluate EviDiff with both loss weights set to `1.5`:

```bash
python train_IP_evidence_spatial_contrastive.py --mask-loss-weight 1.5 --contrastive-weight 1.5 --output experiments_EviDiff
python train_SA_evidence_spatial_contrastive.py --mask-loss-weight 1.5 --contrastive-weight 1.5 --output experiments_EviDiff
python train_UP_evidence_spatial_contrastive.py --mask-loss-weight 1.5 --contrastive-weight 1.5 --output experiments_EviDiff
```

Prepare the source pickle with `data` of shape `(N, 7, 7, 128)` and a one-dimensional `Labels` array. Each target MAT file must contain one user array. 

## Supplement

Each command runs 10 seeds with 5,000 joint source/target updates, 5-way tasks, 5 target labels per class, and `7 × 7` patches. Training, evaluation, summaries, and classification-map generation are integrated. Inference uses complete fused features for cosine prototype classification.



