# Datasets Directory

This directory contains raw dataset files and data loading utilities for the Sieve project.

## Dataset Files

| Filename | Description |
|----------|-------------|
| `label_encodered_malicious_TLS-1_processed.csv` | Malicious TLS traffic dataset|
| `TLS1.3_like_TLS1.2_processed.csv` | CipherSpectrum dataset |
| `DDoS2019.csv` | DDoS 2019 dataset |
| `processed_friday_dataset.csv` | IDS2018 dataset |

## Code Files

| Filename | Description |
|----------|-------------|
| `dataloader_tls.py` | Data loading functions and `TLSDataset` class for training/testing |
| `__init__.py` | Package initialization |

## Supported Datasets

- **Main datasets**: `malicious_tls`, `DDoS2019`
- **Noise datasets (OOD)**: `TLS1.3`, `IDS2018`
