# Logs Directory

This directory stores training logs, model checkpoints, and evaluation results from `train.py`.

## Directory Structure

```
logs/
└── {dataset}/
    └── Dataset({dataset}_{noise_ratio}_{open_ratio}_{noise_mode})_NoiseDS({noisy_dataset})_Model({zeta}_{xi})/
        ├── best_acc.pth.tar       # Best model checkpoint
        ├── last.pth.tar           # Last epoch checkpoint
        ├── acc.txt                # Training accuracy log
        ├── stat.txt               # Training statistics
        ├── params.csv             # Training parameters
        ├── confusion_matrix.txt   # Test confusion matrix
        ├── epoch_label_acc.txt    # Per-epoch label accuracy
        ├── final_test_results.json # Final test metrics
        ├── Epochs_C_Measure/      # Per-epoch clustering scores
        │   └── scores_epoch_*.json
        └── OOD/                   # OOD detection results
            ├── ssd_scores_id.json
            ├── ssd_scores_ood.json
            └── ssd_ood_detection_results.json
```