# Unknown Traffic Labeling for Unknown Traffic

This module implements Generalized Category Discovery for unknown traffic, building upon the closed-set classification and open-set detection results.

## Overview

The GCD pipeline consists of the following steps:

1. **OOD Detection & Sample Saving** (`save_ood_samples.py`)
   - Load trained model from `train.py`
   - Perform OOD detection using Sieve score
   - Save ID samples with original labels
   - Save OOD samples with unified label (for novel category discovery)

2. **Generalized Novel Category Discovery** (`run_gcd_pipeline.py`)
   - Estimate total number of classes K (known + novel)
   - Perform semi-supervised K-Means clustering
   - Evaluate clustering performance

3. **Manual Intervention**
   - Split discovered novel samples into train/test sets
   - Prepare data for potential next phase

## Directory Structure

```
Unknown_Traffic_Labeling/
├── README.md                      # This file
├── save_ood_samples.py           # Save ID/OOD samples from OOD detection
├── run_gcd_pipeline.py           # Main GCD pipeline script
├── novel_category_discovery.py   # GNCD implementation
├── utils/
│   ├── __init__.py
│   ├── cluster_utils.py          # Clustering utilities
│   ├── metrics.py                # Evaluation metrics
│   └── faster_mix_k_means.py     # Semi-supervised K-Means
├── data/                         # Saved OOD detection data
│   └── {dataset}_{noise_ratio}_{open_ratio}_{noisy_dataset}/
│       ├── train_features.npy
│       ├── train_labels.npy
│       ├── id_features.npy
│       ├── id_labels.npy
│       ├── ood_features.npy
│       ├── ood_labels.npy
│       ├── ood_original_labels.npy
│       └── metadata.json
└── results/                      # GCD results
    └── {dataset}_{noise_ratio}_{open_ratio}_{noisy_dataset}/
        ├── gcd_results.json
        ├── config.json
        ├── clustering_predictions.png
        ├── known_vs_novel_distribution.png
        ├── metrics_summary.png
        ├── true_labels_distribution.png
        ├── novel_train_features.npy
        ├── novel_train_labels.npy
        ├── novel_test_features.npy
        ├── novel_test_labels.npy
        ├── train_features_updated.npy
        ├── train_labels_updated.npy
        ├── test_features_updated.npy
        └── test_labels_updated.npy
```

## Usage

### Step 1: Save OOD Detection Samples

First, ensure you have trained the model using `train.py`.

Then run:

```bash
cd /home/ju/Desktop/TNSE/Sieve/Unknown_Traffic_Labeling

python save_ood_samples.py \
    --dataset DDoS2019 \
    --noisy_dataset IDS2018 \
    --noise_ratio 0.1 \
    --open_ratio 0.5
```

### Step 2: Run GCD Pipeline

```bash
python run_gcd_pipeline.py \
    --dataset DDoS2019 \
    --noisy_dataset IDS2018 \
    --noise_ratio 0.1 \
    --open_ratio 0.5 \
    --max_K 50 \
    --max_kmeans_iter 200 \
    --k_means_init 10
```

## Evaluation Metrics

- **AKS (Accuracy on Known/Seen classes)**: Classification accuracy on known classes
- **ANS (Accuracy on Novel/Seen classes)**: Clustering accuracy on novel classes
- **HCA (Harmonic Clustering Accuracy)**: Harmonic mean of AKS and ANS
- **NMI (Normalized Mutual Information)**: Clustering quality metric
- **ARI (Adjusted Rand Index)**: Clustering similarity metric
