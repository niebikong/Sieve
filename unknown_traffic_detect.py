import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import argparse

# Import necessary modules
from models.preresnet import DeepResNet
from datasets.dataloader_tls import TLSDataset, GaussianNoiseTransform, FeatureSwappingTransform, load_IDS2018, load_TLS13
from utils import weighted_knn

# Use the same argument parser as main_tls_L_batch_only.py
parser = argparse.ArgumentParser('Unknown Detection using Sieve')
parser.add_argument('--dataset', default='malicious_tls', choices=['malicious_tls', 'DDoS2019'],
                    help='main dataset name: malicious_tls or DDoS2019')
parser.add_argument('--noisy_dataset', default='TLS1.3', choices=['TLS1.3', 'IDS2018'],
                    help='open-set noise dataset name: TLS1.3 or IDS2018')

# dataset settings
parser.add_argument('--noise_mode', default='sym', type=str, help='artifical noise mode')  # sym, asym
parser.add_argument('--noise_ratio', default=0.5, type=float, help='artifical noise ratio')
parser.add_argument('--open_ratio', default=0.5, type=float, help='artifical noise ratio')

# model settings
parser.add_argument('--xi', default=1.0, type=float, help='threshold for selecting samples (default: 1)')
parser.add_argument('--zeta', default=0.93, type=float, help='threshold for relabelling samples (default: 0.9)')
parser.add_argument('--lambda_fc', default=1.0, type=float, help='weight of L_batch self-supervised contrastive loss (default: 1.0)')
parser.add_argument('--temperature', default=0.05, type=float, help='temperature parameter for L_batch loss (default: 0.07)')
parser.add_argument('--k', default=100, type=int, help='neighbors for knn sample selection')

# train settings
parser.add_argument('--model', default='DeepResNet', help='model architecture (default: DeepResNet)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=1024, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--run_path', type=str, help='run path containing all results')

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
        
        # Normalize features (standardization)
        self.mean_feat = train_features.mean(0)
        self.std_feat = train_features.std(0)
        
        # Standardize training features
        train_features_std = self._standardize_features(train_features)
        
        # Compute class-wise statistics
        cov = lambda x: np.cov(x.T, bias=True)
        
        for cls in range(self.num_classes):
            # Get features for current class
            cls_mask = train_labels == cls
            if cls_mask.sum() == 0:
                print(f"Warning: No samples found for class {cls}")
                continue
                
            cls_features = train_features_std[cls_mask]
            
            # Compute class mean
            self.mean_cls[cls] = cls_features.mean(0)
            
            # Compute class covariance and its inverse
            feat_cls_center = cls_features - self.mean_cls[cls]
            try:
                self.inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
            except:
                print(f"Warning: Failed to compute inverse covariance for class {cls}")
                self.inv_sigma_cls[cls] = np.eye(feat_cls_center.shape[1])
        
        print("Sieve unknown detector setup completed!")
    
    def _standardize_features(self, features):
        """Standardize features using training statistics"""
        return (features - self.mean_feat) / (self.std_feat + 1e-10)
    
    def compute_scores(self, features):
        """Compute SSD+ scores for given features"""
        # Standardize features
        features_std = self._standardize_features(features)
        
        # Compute Mahalanobis distance for each class
        score_cls = np.zeros((self.num_classes, len(features_std)))
        
        for cls in range(self.num_classes):
            if self.inv_sigma_cls[cls] is None or self.mean_cls[cls] is None:
                continue
                
            inv_sigma = self.inv_sigma_cls[cls]
            mean = self.mean_cls[cls]
            z = features_std - mean
            
            # Compute negative Mahalanobis distance
            score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
        
        # Return maximum score across all classes
        return score_cls.max(0)

def extract_features_with_multi_layer(encoder, classifier, dataloader, device, is_ood=False):
    """Extract multi-layer features similar to reference_code approach with global average pooling"""
    encoder.eval()
    classifier.eval()

    all_features = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting multi-layer features"):
            if is_ood:
                # For OOD data (TensorDataset format)
                data, target = batch
                data = data.cuda()
            else:
                # For TLSDataset format
                if isinstance(batch[0], list):  # Training data with multiple views
                    data = batch[0][0].cuda()  # Take the first view
                else:  # Test data with single view
                    data = batch[0].cuda()
                target = batch[2] if len(batch) > 2 else batch[1]

            # Ensure input is 3D: [batch_size, channels, length]
            if len(data.shape) == 2:
                data = data.unsqueeze(1)

            # Extract multi-layer features using feature_list method
            logits, feature_list = encoder.feature_list(data)

            # Apply global average pooling to each feature layer
            pooled_features = []
            for layer_feat in feature_list:
                # Global average pooling: [batch, channels, length] -> [batch, channels]
                pooled_feat = F.adaptive_avg_pool1d(layer_feat, 1).squeeze(-1)
                pooled_features.append(pooled_feat)

            # Concatenate all pooled features
            multi_layer_features = torch.cat(pooled_features, dim=1)

            # Get final logits from classifier if needed
            if classifier is not None:
                # Use the final encoder features for classifier
                final_features = encoder(data)
                classifier_logits = classifier(final_features)
            else:
                classifier_logits = logits

            # Normalize the concatenated multi-layer features
            multi_layer_features = F.normalize(multi_layer_features, dim=1)

            all_features.append(multi_layer_features.cpu().numpy())
            all_logits.append(classifier_logits.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    return {
        "features": np.concatenate(all_features, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "labels": np.concatenate(all_labels, axis=0)
    }

def determine_threshold_from_train(train_scores, tpr_target=0.95):
    """Determine threshold from training scores to achieve target TPR"""
    # Sort training scores (ID data)
    sorted_scores = np.sort(train_scores)

    # Calculate threshold that allows tpr_target proportion of training data to be classified as ID
    # Since higher scores indicate more ID-like, we want the (1-tpr_target) percentile as threshold
    threshold_idx = int((1 - tpr_target) * len(sorted_scores))
    threshold = sorted_scores[threshold_idx]

    print(f"Threshold determined from training set: {threshold:.4f}")
    print(f"This threshold allows {tpr_target*100:.1f}% of training ID samples to be correctly classified")

    return threshold

def compute_ood_metrics_with_threshold(scores_id, scores_ood, threshold):
    """Compute comprehensive OOD detection metrics using predetermined threshold"""
    # Handle potential infinite values
    scores_id = np.nan_to_num(scores_id, nan=np.nanmean(scores_id), posinf=np.nanmax(scores_id), neginf=np.nanmin(scores_id))
    scores_ood = np.nan_to_num(scores_ood, nan=np.nanmean(scores_ood), posinf=np.nanmax(scores_ood), neginf=np.nanmin(scores_ood))

    # Combine scores and create labels (1 for ID, 0 for OOD)
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])

    # Calculate AUROC (note: using scores directly because lower SSD+ scores indicate more anomalous)
    fpr, tpr, thresholds_roc = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)

    # Find TNR at TPR95 using ROC curve
    idx_tpr95 = np.abs(tpr - 0.95).argmin()
    tnr_at_tpr95 = 1 - fpr[idx_tpr95]

    # Make predictions using the predetermined threshold
    predictions = (scores >= threshold).astype(int)

    # Calculate metrics
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, zero_division=0)
    recall = metrics.recall_score(labels, predictions, zero_division=0)
    f1 = metrics.f1_score(labels, predictions, zero_division=0)

    # Calculate confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()

    # Calculate actual TPR and TNR with the predetermined threshold
    actual_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    actual_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'AUROC': auroc * 100,
        'TNR@TPR95': tnr_at_tpr95 * 100,  # From ROC curve
        'Accuracy': accuracy * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1': f1 * 100,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'threshold': threshold,
        'actual_TPR': actual_tpr * 100,
        'actual_TNR': actual_tnr * 100
    }

def save_scores(scores, dataset_type, log_dir):
    """Save scores to JSON file"""
    scores_dict = {
        'scores': scores.tolist(),
        'dataset_type': dataset_type
    }

    # Create OOD subdirectory in log directory
    ood_dir = Path(log_dir) / 'OOD'
    ood_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    save_path = ood_dir / f'ssd_scores_{dataset_type}.json'
    with open(save_path, 'w') as f:
        json.dump(scores_dict, f, indent=2)

    print(f"Scores saved to {save_path}")

def extract_features_to_npz(id_outputs, ood_outputs, args, log_dir):
    """
    Extract and save features from ID and OOD datasets to NPZ file
    
    Args:
        id_outputs: Dictionary containing ID dataset features, logits, and labels
        ood_outputs: Dictionary containing OOD dataset features, logits, and labels
        args: Arguments containing dataset configuration
        log_dir: Directory to save the NPZ file
    
    Returns:
        Path to the saved NPZ file
    
    Note:
        - ID labels: Keep original labels (0, 1, 2, ..., num_classes-1)
        - OOD labels: Unified as -1 to indicate unknown class
    """
    # Create OOD subdirectory in log directory
    ood_dir = Path(log_dir) / 'OOD'
    ood_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    # Format: features_{id_dataset}_vs_{ood_dataset}_{noise_ratio}_{open_ratio}_{noise_mode}.npz
    filename = f"features_{args.dataset}_vs_{args.noisy_dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}.npz"
    save_path = ood_dir / filename
    
    # Process labels:
    # - ID labels: Keep original labels (already correct)
    # - OOD labels: Set all to -1 (unknown class)
    id_labels = id_outputs['labels'].copy()  # Keep original ID labels
    ood_labels = np.full_like(ood_outputs['labels'], -1)  # All OOD samples labeled as -1
    
    # Get unique ID labels for verification
    unique_id_labels = np.unique(id_labels)
    unique_ood_labels = np.unique(ood_labels)
    
    # Prepare data dictionary
    data_dict = {
        # ID data (known classes with original labels)
        'id_features': id_outputs['features'],
        'id_logits': id_outputs['logits'],
        'id_labels': id_labels,
        
        # OOD data (unknown classes, all labeled as -1)
        'ood_features': ood_outputs['features'],
        'ood_logits': ood_outputs['logits'],
        'ood_labels': ood_labels,
        
        # Metadata
        'id_dataset': args.dataset,
        'ood_dataset': args.noisy_dataset,
        'noise_ratio': args.noise_ratio,
        'open_ratio': args.open_ratio,
        'noise_mode': args.noise_mode,
        'num_classes': args.num_classes,
        'input_size': args.input_size,
        'id_num_samples': len(id_outputs['features']),
        'ood_num_samples': len(ood_outputs['features']),
    }
    
    # Save to NPZ file
    np.savez_compressed(save_path, **data_dict)
    
    print(f"\n{'='*60}")
    print(f"Features saved to NPZ file:")
    print(f"  Path: {save_path}")
    print(f"  ID samples: {len(id_outputs['features'])}")
    print(f"  OOD samples: {len(ood_outputs['features'])}")
    print(f"  Feature dimension: {id_outputs['features'].shape[1]}")
    print(f"  ID dataset: {args.dataset}")
    print(f"  OOD dataset: {args.noisy_dataset}")
    print(f"  ID label range: {unique_id_labels.min()} to {unique_id_labels.max()} ({len(unique_id_labels)} classes)")
    print(f"  OOD labels: All set to -1 (unknown class)")
    print(f"{'='*60}\n")
    
    return save_path

def select_clean_samples(dataloader, encoder, classifier, args):
    """Select clean samples using the same logic as main_tls_L_batch_only.py"""
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []
    all_indices = []
    noisy_label = []

    with torch.no_grad():
        # Generate feature bank
        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting for sample selection'):
            if isinstance(data, list):  # Handle KCropsTransform output
                data = data[0]  # Take the first view
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

        # Sample relabelling
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean():.4f} max: {his_score.max():.4f} min: {his_score.min():.4f}')
        conf_id = torch.where(his_score > args.zeta)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        # Sample selection using weighted_knn
        # Increase chunks for large datasets to avoid OOM
        num_samples = len(feature_bank)
        chunks = max(10, num_samples // 10000)  # At least 10 chunks, or 1 chunk per 10k samples
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, chunks)
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.xi)[0]

        print(f'Sample selection: xi={args.xi}, selected {len(clean_id)}/{len(right_score)} clean samples')
        print(f'Clean sample ratio: {len(clean_id)/len(right_score)*100:.2f}%')

    return clean_id, all_indices

def main():
    args = parser.parse_args()

    # Set up run path similar to main_tls_L_batch_only.py
    if args.run_path is None:
        args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_NoiseDS({args.noisy_dataset})_Model({args.zeta}_{args.xi})'

    # Set up log directory
    log_dir = f'/home/ju/Desktop/TNSE/Sieve/logs/{args.dataset}/{args.run_path}'

    print("=== Sieve Unknown Detection ===")
    print(f"Dataset: {args.dataset}")
    print(f"Log directory: {log_dir}")
    print(f"Noise ratio: {args.noise_ratio}, Open ratio: {args.open_ratio}, Noise mode: {args.noise_mode}")

    # Configure dataset-specific parameters
    if args.dataset == 'malicious_tls':
        args.num_classes = 23  # Number of classes in malicious TLS dataset
        args.input_size = 86  # Feature vector size
    elif args.dataset == 'DDoS2019':
        args.num_classes = 2  # Number of classes in DDoS2019 dataset (benign: 0, attack: 1)
        args.input_size = 82  # Feature vector size
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Initialize models with correct initial channels to match checkpoint (32 channels)
    encoder = DeepResNet(input_size=args.input_size, num_classes=None, initial_channels=32)
    classifier = torch.nn.Linear(256, args.num_classes)  # 256 is the feature dimension from DeepResNet with 32 initial channels
    
    # Load checkpoint - try best_acc.pth.tar first, then last.pth.tar
    checkpoint_paths = [
        f'{log_dir}/best_acc.pth.tar',
        f'{log_dir}/last.pth.tar'
    ]

    checkpoint = None
    checkpoint_path = None

    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            break

    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in paths: {checkpoint_paths}")

    # Load model states with error handling
    try:
        encoder.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()})
        classifier.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()})
        print("Model loaded successfully!")
        if 'best_acc' in checkpoint:
            print(f"Best training accuracy: {checkpoint['best_acc']:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with strict=False...")
        encoder.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()}, strict=False)
        classifier.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()}, strict=False)
    
    encoder.cuda()
    classifier.cuda()
    
    # Setup data transforms
    # strong_transform = FeatureSwappingTransform(swap_ratio=0.05)  # 强增强使用较大的交换比例
    # weak_transform = FeatureSwappingTransform(swap_ratio=0)   # 弱增强使用较小的交换比例
    # none_transform = None
    weak_transform = GaussianNoiseTransform(mean=0., std=0.01)
    none_transform = None

    # Load datasets
    print("Loading datasets...")

    # Training data (for detector setup) - use same noise file path as main_tls_L_batch_only.py
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
        selected_ood_classes=[1, 2, 3]  # Use consistent OOD classes
    )

    # Test data (ID)
    test_data = TLSDataset(
        dataset=args.dataset,
        transform=none_transform,
        dataset_mode='test'
    )
    
    # OOD data (load based on noisy_dataset parameter)
    if args.noisy_dataset == 'TLS1.3':
        print("Loading TLS1.3 dataset as OOD data...")
        X_ood, y_ood, _, _ = load_TLS13()
        # Truncate TLS1.3 features to match main dataset dimension
        if args.input_size > 86:
            X_ood_padded = np.pad(X_ood, ((0, 0), (0, args.input_size - 86)), mode='constant', constant_values=0)
        else:
            X_ood_padded = X_ood[:, :args.input_size]
        print(f"OOD dataset: {len(X_ood_padded)} samples from {len(np.unique(y_ood))} classes (TLS1.3, padded/truncated to {args.input_size} dimensions)")
    elif args.noisy_dataset == 'IDS2018':
        print("Loading IDS2018 dataset as OOD data...")
        X_ood, y_ood, _, _ = load_IDS2018(selected_classes=[0, 1], target_dim=args.input_size)
        X_ood_padded = X_ood  # Already padded to target dimensions
        print(f"OOD dataset: {len(X_ood_padded)} samples from {len(np.unique(y_ood))} classes (IDS2018, padded to {args.input_size} dimensions)")
    else:
        raise ValueError(f"Unsupported noisy_dataset: {args.noisy_dataset}. Choose from 'TLS1.3' or 'IDS2018'")

    ood_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_ood_padded),
        torch.LongTensor(y_ood)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Select clean samples from training set using xi threshold
    print("Selecting clean samples from training set...")
    clean_id, all_indices = select_clean_samples(train_loader, encoder, classifier, args)

    # Create a subset of training data with only clean samples
    from torch.utils.data import Subset
    clean_train_data = Subset(train_data, clean_id.cpu())
    clean_train_loader = torch.utils.data.DataLoader(
        clean_train_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Extract features from clean training samples only
    print(f"Extracting features from {len(clean_id)} clean training samples...")
    train_outputs = extract_features_with_multi_layer(encoder, classifier, clean_train_loader, 'cuda', is_ood=False)
    
    print("Extracting features from test set (ID)...")
    id_outputs = extract_features_with_multi_layer(encoder, classifier, test_loader, 'cuda', is_ood=False)
    
    print("Extracting features from OOD set...")
    ood_outputs = extract_features_with_multi_layer(encoder, classifier, ood_loader, 'cuda', is_ood=True)
    
    # Save extracted features to NPZ file
    # npz_path = extract_features_to_npz(id_outputs, ood_outputs, args, log_dir)
    
    # Setup Sieve unknown detector
    detector = Sieve_unknown_detect(args.num_classes)
    detector.setup(train_outputs['features'], train_outputs['labels'])
    
    # Compute OOD scores
    print("Computing Sieve scores...")
    begin = time.time()

    scores_train = detector.compute_scores(train_outputs['features'])
    scores_id = detector.compute_scores(id_outputs['features'])
    scores_ood = detector.compute_scores(ood_outputs['features'])

    print(f"Score computation time: {time.time() - begin:.2f} seconds")

    # Print score statistics for debugging
    print(f"\nScore Statistics:")
    print(f"Train scores - Mean: {np.mean(scores_train):.4f}, Std: {np.std(scores_train):.4f}, Min: {np.min(scores_train):.4f}, Max: {np.max(scores_train):.4f}")
    print(f"ID scores - Mean: {np.mean(scores_id):.4f}, Std: {np.std(scores_id):.4f}, Min: {np.min(scores_id):.4f}, Max: {np.max(scores_id):.4f}")
    print(f"OOD scores - Mean: {np.mean(scores_ood):.4f}, Std: {np.std(scores_ood):.4f}, Min: {np.min(scores_ood):.4f}, Max: {np.max(scores_ood):.4f}")
    
    # Save scores
    save_scores(scores_id, 'id', log_dir)
    save_scores(scores_ood, 'ood', log_dir)
    
    # Determine threshold from training set
    print("Determining threshold from training set...")
    threshold = determine_threshold_from_train(scores_train, tpr_target=0.95)

    # Compute and print metrics using the predetermined threshold
    metrics_results = compute_ood_metrics_with_threshold(scores_id, scores_ood, threshold)

    # Save detailed results to JSON file
    detailed_results = {
        'dataset_config': {
            'id_dataset': args.dataset,
            'ood_dataset': args.noisy_dataset,
            'id_samples': len(scores_id),
            'ood_samples': len(scores_ood),
            'total_samples': len(scores_id) + len(scores_ood),
            'noise_ratio': args.noise_ratio,
            'open_ratio': args.open_ratio,
            'noise_mode': args.noise_mode
        },
        'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in metrics_results.items()},
        'score_statistics': {
            'id_scores': {
                'mean': float(np.mean(scores_id)),
                'std': float(np.std(scores_id)),
                'min': float(np.min(scores_id)),
                'max': float(np.max(scores_id))
            },
            'ood_scores': {
                'mean': float(np.mean(scores_ood)),
                'std': float(np.std(scores_ood)),
                'min': float(np.min(scores_ood)),
                'max': float(np.max(scores_ood))
            }
        }
    }

    # Save detailed results
    ood_dir = Path(log_dir) / 'OOD'
    results_file = ood_dir / 'ssd_ood_detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print("\n" + "="*60)
    print("Sieve Unknown Detection Results (Binary Classification)")
    print("="*60)
    print(f"Dataset Configuration:")
    print(f"  ID (In-Distribution): Malicious TLS dataset ({len(scores_id)} samples)")
    print(f"  OOD (Out-of-Distribution): {args.noisy_dataset} dataset ({len(scores_ood)} samples)")
    print(f"  Total samples: {len(scores_id) + len(scores_ood)}")
    print(f"  Noise ratio: {args.noise_ratio}, Open ratio: {args.open_ratio}, Mode: {args.noise_mode}")
    print()
    print(f"Binary Classification Metrics:")
    print(f"  AUROC: {metrics_results['AUROC']:.2f}%")
    print(f"  TNR@TPR95: {metrics_results['TNR@TPR95']:.2f}%")
    print(f"  Accuracy: {metrics_results['Accuracy']:.2f}%")
    print(f"  Precision: {metrics_results['Precision']:.2f}%")
    print(f"  Recall: {metrics_results['Recall']:.2f}%")
    print(f"  F1-Score: {metrics_results['F1']:.2f}%")
    print(f"  Threshold (from training set): {metrics_results['threshold']:.4f}")
    print(f"  Actual TPR on test set: {metrics_results['actual_TPR']:.2f}%")
    print(f"  Actual TNR on test set: {metrics_results['actual_TNR']:.2f}%")
    print()
    print(f"Confusion Matrix (ID=1, OOD=0):")
    print(f"  True Positives (TP - Correctly identified ID): {metrics_results['TP']}")
    print(f"  True Negatives (TN - Correctly identified OOD): {metrics_results['TN']}")
    print(f"  False Positives (FP - OOD misclassified as ID): {metrics_results['FP']}")
    print(f"  False Negatives (FN - ID misclassified as OOD): {metrics_results['FN']}")
    print("="*60)
    print(f"Detailed results saved to: {results_file}")

    print("\nSieve unknown detection completed successfully!")

if __name__ == '__main__':
    main()
