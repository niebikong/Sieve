import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preresnet import DeepResNet
from datasets.dataloader_tls import (
    TLSDataset, GaussianNoiseTransform, FeatureSwappingTransform,
    load_TLS13, load_IDS2018
)
from utils import weighted_knn


def parse_args():
    parser = argparse.ArgumentParser('Save OOD Detection Samples for Unknown Traffic Labeling')
    parser.add_argument('--dataset', default='DDoS2019', choices=['malicious_tls', 'DDoS2019'],
                        help='main dataset name')
    parser.add_argument('--noisy_dataset', default='IDS2018', choices=['TLS1.3', 'IDS2018'],
                        help='open-set noise dataset name: TLS1.3 or IDS2018')
    parser.add_argument('--noise_mode', default='sym', type=str, help='noise mode')
    parser.add_argument('--noise_ratio', default=0.1, type=float, help='noise ratio')
    parser.add_argument('--open_ratio', default=0.5, type=float, help='open-set ratio')
    parser.add_argument('--xi', default=1.0, type=float, help='threshold for selecting samples')
    parser.add_argument('--zeta', default=0.93, type=float, help='threshold for relabelling samples')
    parser.add_argument('--k', default=100, type=int, help='neighbors for knn sample selection')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--tpr_target', default=0.95, type=float, help='target TPR for threshold')
    parser.add_argument('--gpuid', default='0', type=str, help='GPU ID')
    parser.add_argument('--run_path', type=str, help='run path containing all results')
    return parser.parse_args()


class Sieve_unknown_detect:
    """Sieve Unknown Detector based on Mahalanobis distance"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mean_feat = None
        self.std_feat = None
        self.inv_sigma_cls = [None for _ in range(num_classes)]
        self.mean_cls = [None for _ in range(num_classes)]
        
    def setup(self, train_features, train_labels):
        """Setup the detector with training features and labels"""
        print("Setting up Sieve unknown detector...")
        
        self.mean_feat = train_features.mean(0)
        self.std_feat = train_features.std(0)
        
        train_features_std = self._standardize_features(train_features)
        
        cov = lambda x: np.cov(x.T, bias=True)
        
        for cls in range(self.num_classes):
            cls_mask = train_labels == cls
            if cls_mask.sum() == 0:
                print(f"Warning: No samples found for class {cls}")
                continue
                
            cls_features = train_features_std[cls_mask]
            self.mean_cls[cls] = cls_features.mean(0)
            
            feat_cls_center = cls_features - self.mean_cls[cls]
            try:
                self.inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
            except:
                print(f"Warning: Failed to compute inverse covariance for class {cls}")
                self.inv_sigma_cls[cls] = np.eye(feat_cls_center.shape[1])
        
        print("Sieve unknown detector setup completed!")
    
    def _standardize_features(self, features):
        return (features - self.mean_feat) / (self.std_feat + 1e-10)
    
    def compute_scores(self, features):
        """Compute Sieve scores for given features"""
        features_std = self._standardize_features(features)
        score_cls = np.zeros((self.num_classes, len(features_std)))
        
        for cls in range(self.num_classes):
            if self.inv_sigma_cls[cls] is None or self.mean_cls[cls] is None:
                continue
                
            inv_sigma = self.inv_sigma_cls[cls]
            mean = self.mean_cls[cls]
            z = features_std - mean
            score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
        
        return score_cls.max(0)


def extract_features_with_multi_layer(encoder, classifier, dataloader, device, is_ood=False):
    """Extract multi-layer features with global average pooling"""
    encoder.eval()
    classifier.eval()

    all_features = []
    all_logits = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if is_ood:
                data, target = batch
                data = data.cuda()
                indices = torch.arange(batch_idx * dataloader.batch_size, 
                                       batch_idx * dataloader.batch_size + len(target))
            else:
                if isinstance(batch[0], list):
                    data = batch[0][0].cuda()
                else:
                    data = batch[0].cuda()
                target = batch[2] if len(batch) > 2 else batch[1]
                indices = batch[3] if len(batch) > 3 else torch.arange(len(target))

            if len(data.shape) == 2:
                data = data.unsqueeze(1)

            logits, feature_list = encoder.feature_list(data)

            pooled_features = []
            for layer_feat in feature_list:
                pooled_feat = F.adaptive_avg_pool1d(layer_feat, 1).squeeze(-1)
                pooled_features.append(pooled_feat)

            multi_layer_features = torch.cat(pooled_features, dim=1)

            if classifier is not None:
                final_features = encoder(data)
                classifier_logits = classifier(final_features)
            else:
                classifier_logits = logits

            multi_layer_features = F.normalize(multi_layer_features, dim=1)

            all_features.append(multi_layer_features.cpu().numpy())
            all_logits.append(classifier_logits.cpu().numpy())
            all_labels.append(target.cpu().numpy() if isinstance(target, torch.Tensor) else target)
            all_indices.append(indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices)

    return {
        "features": np.concatenate(all_features, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "indices": np.concatenate(all_indices, axis=0)
    }


def extract_raw_features(encoder, dataloader, device, is_ood=False, is_train=False):
    """Extract raw features from encoder (256-dim)
    
    For TLSDataset:
    - Train mode returns: (data, target, clean_target, index) - 4 elements
    - Test mode returns: (data, target, index) - 3 elements
    - OOD data (TensorDataset) returns: (data, target) - 2 elements
    """
    encoder.eval()
    
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting raw features")):
            if is_ood:
                # OOD data: TensorDataset returns (data, target)
                data, target = batch
                data = data.cuda()
                indices = torch.arange(batch_idx * dataloader.batch_size, 
                                       batch_idx * dataloader.batch_size + len(target))
            elif is_train:
                # Train mode: (data, target, clean_target, index)
                data = batch[0].cuda() if not isinstance(batch[0], list) else batch[0][0].cuda()
                target = batch[1]  # noisy label
                indices = batch[3] if len(batch) > 3 else torch.arange(len(target))
            else:
                # Test mode: (data, target, index)
                data = batch[0].cuda() if not isinstance(batch[0], list) else batch[0][0].cuda()
                target = batch[1]  # label
                indices = batch[2] if len(batch) > 2 else torch.arange(len(target))
            
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            
            features = encoder(data)
            features = F.normalize(features, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(target.cpu().numpy() if isinstance(target, torch.Tensor) else target)
            all_indices.append(indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices)
    
    return {
        "features": np.concatenate(all_features, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "indices": np.concatenate(all_indices, axis=0)
    }


def determine_threshold_from_train(train_scores, tpr_target=0.95):
    """Determine threshold from training scores to achieve target TPR"""
    sorted_scores = np.sort(train_scores)
    threshold_idx = int((1 - tpr_target) * len(sorted_scores))
    threshold = sorted_scores[threshold_idx]
    print(f"Threshold determined from training set: {threshold:.4f}")
    print(f"This threshold allows {tpr_target*100:.1f}% of training ID samples to be correctly classified")
    return threshold


def select_clean_samples(dataloader, encoder, classifier, args):
    """Select clean samples using the same logic as main_tls_L_batch_only.py"""
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []
    all_indices = []
    noisy_label = []

    with torch.no_grad():
        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting for sample selection'):
            if isinstance(data, list):
                data = data[0]
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            output = classifier(feature)
            prediction.append(output)
            all_indices.append(index)
            noisy_label.append(target)

        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)
        all_indices = torch.cat(all_indices, dim=0)
        noisy_label = torch.cat(noisy_label, dim=0).cuda()

        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        conf_id = torch.where(his_score > args.zeta)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        num_samples = len(feature_bank)
        chunks = max(10, num_samples // 10000)
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, chunks)
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.xi)[0]

        print(f'Sample selection: xi={args.xi}, selected {len(clean_id)}/{len(right_score)} clean samples')

    return clean_id, all_indices


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    
    # Set up paths
    if args.run_path is None:
        args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_NoiseDS({args.noisy_dataset})_Model({args.zeta}_{args.xi})'
    
    log_dir = f'/home/ju/Desktop/TNSE/Sieve/logs/{args.dataset}/{args.run_path}'
    utl_dir = Path('/home/ju/Desktop/TNSE/Sieve/Unknown_Traffic_Labeling/data')
    utl_dir.mkdir(parents=True, exist_ok=True)
    
    save_dir = utl_dir / f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noisy_dataset}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Save OOD Detection Samples for Unknown Traffic Labeling ===")
    print(f"Dataset: {args.dataset}")
    print(f"Noisy dataset: {args.noisy_dataset}")
    print(f"Log directory: {log_dir}")
    print(f"Save directory: {save_dir}")
    
    # Configure dataset-specific parameters
    if args.dataset == 'malicious_tls':
        args.num_classes = 23
        args.input_size = 117
    elif args.dataset == 'DDoS2019':
        args.num_classes = 2
        args.input_size = 82
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Initialize models
    encoder = DeepResNet(input_size=args.input_size, num_classes=None, initial_channels=32)
    classifier = torch.nn.Linear(256, args.num_classes)
    
    # Load checkpoint
    checkpoint_paths = [
        f'{log_dir}/best_acc.pth.tar',
        f'{log_dir}/last.pth.tar'
    ]
    
    checkpoint = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"Loading checkpoint from: {path}")
            checkpoint = torch.load(path, map_location='cpu')
            break
    
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in paths: {checkpoint_paths}")
    
    encoder.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()})
    classifier.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()})
    print("Model loaded successfully!")
    
    encoder.cuda()
    classifier.cuda()
    
    strong_transform = FeatureSwappingTransform(swap_ratio=0.05)  
    weak_transform = FeatureSwappingTransform(swap_ratio=0)   
    none_transform = None
    
    # Load datasets
    print("Loading datasets...")
    noise_file = f'/home/ju/Desktop/TNSE/Sieve/noise_files/{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_{args.noisy_dataset}_noise.json'
    
    train_data = TLSDataset(
        dataset=args.dataset,
        noisy_dataset=args.noisy_dataset,
        transform=weak_transform,
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        open_ratio=args.open_ratio,
        dataset_mode='train',
        noise_file=noise_file,
        selected_ood_classes=[1, 2, 3]
    )
    
    test_data = TLSDataset(
        dataset=args.dataset,
        transform=none_transform,
        dataset_mode='test'
    )
    
    # Load OOD data
    if args.noisy_dataset == 'TLS1.3':
        print("Loading TLS1.3 dataset as OOD data...")
        X_ood, y_ood, _, _ = load_TLS13()
        if args.input_size > 86:
            X_ood_padded = np.pad(X_ood, ((0, 0), (0, args.input_size - 86)), mode='constant', constant_values=0)
        else:
            X_ood_padded = X_ood[:, :args.input_size]
    elif args.noisy_dataset == 'IDS2018':
        print("Loading IDS2018 dataset as OOD data...")
        X_ood, y_ood, _, _ = load_IDS2018(selected_classes=[0, 1], target_dim=args.input_size)
        X_ood_padded = X_ood
    else:
        raise ValueError(f"Unsupported noisy_dataset: {args.noisy_dataset}. Choose from 'TLS1.3' or 'IDS2018'")
    
    print(f"OOD dataset: {len(X_ood_padded)} samples")
    
    ood_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_ood_padded),
        torch.LongTensor(y_ood)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Select clean samples from training set
    print("Selecting clean samples from training set...")
    clean_id, all_indices = select_clean_samples(train_loader, encoder, classifier, args)
    
    # Create clean training subset
    from torch.utils.data import Subset
    clean_train_data = Subset(train_data, clean_id.cpu())
    clean_train_loader = torch.utils.data.DataLoader(
        clean_train_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Extract features for OOD detection
    print(f"Extracting features from {len(clean_id)} clean training samples...")
    train_outputs = extract_features_with_multi_layer(encoder, classifier, clean_train_loader, 'cuda', is_ood=False)
    
    print("Extracting features from test set (ID)...")
    id_outputs = extract_features_with_multi_layer(encoder, classifier, test_loader, 'cuda', is_ood=False)
    
    print("Extracting features from OOD set...")
    ood_outputs = extract_features_with_multi_layer(encoder, classifier, ood_loader, 'cuda', is_ood=True)
    
    # Setup Sieve unknown detector
    detector = Sieve_unknown_detect(args.num_classes)
    detector.setup(train_outputs['features'], train_outputs['labels'])
    
    # Compute OOD scores
    print("Computing Sieve scores...")
    scores_train = detector.compute_scores(train_outputs['features'])
    scores_id = detector.compute_scores(id_outputs['features'])
    scores_ood = detector.compute_scores(ood_outputs['features'])
    
    # Determine threshold
    threshold = determine_threshold_from_train(scores_train, tpr_target=args.tpr_target)
    
    # Classify samples as ID or OOD
    id_predictions = scores_id >= threshold  # True = ID, False = OOD
    ood_predictions = scores_ood >= threshold  # True = ID (false positive), False = OOD (correct)
    
    print(f"\nID Test Set: {id_predictions.sum()}/{len(id_predictions)} classified as ID")
    print(f"OOD Test Set: {(~ood_predictions).sum()}/{len(ood_predictions)} classified as OOD")
    
    # Extract raw features for GCD
    print("\nExtracting raw features for GCD...")
    train_raw = extract_raw_features(encoder, clean_train_loader, 'cuda', is_ood=False, is_train=True)
    id_raw = extract_raw_features(encoder, test_loader, 'cuda', is_ood=False, is_train=False)
    ood_raw = extract_raw_features(encoder, ood_loader, 'cuda', is_ood=True, is_train=False)
    
    # Prepare data for GCD
    # ID samples: samples from test set classified as ID, keep original labels
    id_mask = id_predictions
    id_features = id_raw['features'][id_mask]
    id_labels = id_raw['labels'][id_mask]
    
    # OOD samples: samples from OOD set classified as OOD, assign unified label
    ood_mask = ~ood_predictions
    ood_features = ood_raw['features'][ood_mask]
    # Assign unified label (num_classes) to all OOD samples
    ood_labels = np.full(len(ood_features), args.num_classes, dtype=np.int64)
    ood_original_labels = ood_raw['labels'][ood_mask]  # Keep original labels for evaluation
    
    # Filter training data to only include true ID samples (labels < num_classes)
    # This removes any OOD samples that were mixed into the training set
    train_id_mask = train_raw['labels'] < args.num_classes
    train_features = train_raw['features'][train_id_mask]
    train_labels = train_raw['labels'][train_id_mask]
    
    print(f"\n=== GCD Data Summary ===")
    print(f"Training (labeled) samples: {len(train_features)} (filtered from {len(train_raw['features'])})")
    print(f"  - Known classes only (0-{args.num_classes-1}): {np.unique(train_labels)}")
    print(f"ID test samples (correctly classified): {len(id_features)}")
    print(f"  - Classes: {np.unique(id_labels)}")
    print(f"OOD samples (detected as novel): {len(ood_features)}")
    print(f"  - Unified label: {args.num_classes}")
    print(f"  - Original classes: {np.unique(ood_original_labels)}")
    
    # Save data
    print(f"\nSaving data to {save_dir}...")
    
    # Save training data (labeled exemplars)
    np.save(save_dir / 'train_features.npy', train_features)
    np.save(save_dir / 'train_labels.npy', train_labels)
    
    # Save ID test data
    np.save(save_dir / 'id_features.npy', id_features)
    np.save(save_dir / 'id_labels.npy', id_labels)
    
    # Save OOD data (novel samples)
    np.save(save_dir / 'ood_features.npy', ood_features)
    np.save(save_dir / 'ood_labels.npy', ood_labels)  # Unified label
    np.save(save_dir / 'ood_original_labels.npy', ood_original_labels)  # For evaluation
    
    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'noisy_dataset': args.noisy_dataset,
        'noise_ratio': args.noise_ratio,
        'open_ratio': args.open_ratio,
        'num_classes': args.num_classes,
        'ood_unified_label': args.num_classes,
        'threshold': float(threshold),
        'tpr_target': args.tpr_target,
        'train_samples': len(train_features),
        'id_samples': len(id_features),
        'ood_samples': len(ood_features),
        'feature_dim': train_features.shape[1],
        'id_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(id_labels, return_counts=True))},
        'train_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))},
        'ood_original_class_distribution': {int(k): int(v) for k, v in zip(*np.unique(ood_original_labels, return_counts=True))}
    }
    
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved successfully!")
    print(f"  - train_features.npy: {train_features.shape}")
    print(f"  - train_labels.npy: {train_labels.shape}")
    print(f"  - id_features.npy: {id_features.shape}")
    print(f"  - id_labels.npy: {id_labels.shape}")
    print(f"  - ood_features.npy: {ood_features.shape}")
    print(f"  - ood_labels.npy: {ood_labels.shape}")
    print(f"  - ood_original_labels.npy: {ood_original_labels.shape}")
    print(f"  - metadata.json")
    
    print("\n=== OOD Sample Saving Completed ===")


if __name__ == '__main__':
    main()
