import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from novel_category_discovery import gncd_simple
from utils.cluster_utils import seed_torch
from utils.metrics import evaluation_gcd_all


def parse_args():
    parser = argparse.ArgumentParser('Class Number Estimation')
    parser.add_argument('--dataset', default='DDoS2019', choices=['malicious_tls', 'DDoS2019'],
                        help='main dataset name')
    parser.add_argument('--noisy_dataset', default='IDS2018', choices=['TLS1.3', 'IDS2018'],
                        help='open-set noise dataset name')
    parser.add_argument('--noise_ratio', default=0.1, type=float, help='noise ratio')
    parser.add_argument('--open_ratio', default=0.5, type=float, help='open-set ratio')
    parser.add_argument('--noise_mode', default='sym', type=str, help='noise mode')
    
    # GCD parameters
    parser.add_argument('--max_K', default=100, type=int, help='Maximum number of categories to search')
    parser.add_argument('--max_kmeans_iter', type=int, default=100, help='Max K-means iterations')
    parser.add_argument('--k_means_init', type=int, default=5, help='Number of K-means initializations')
    parser.add_argument('--pairwise_batch_size', type=int, default=2048, help='Batch size for pairwise distance')
    parser.add_argument('--use_brent_optimization', default=True, type=bool, help='Use Brent optimization for K estimation (faster)')
    parser.add_argument('--prop_train_labels', type=float, default=0.83, help='Proportion of training samples')
    parser.add_argument('--subsample_ratio', type=float, default=0.6, help='Subsample ratio for faster K estimation')
    
    parser.add_argument('--seed', default=202512, type=int, help='Random seed')
    parser.add_argument('--gpuid', default='0', type=str, help='GPU ID')
    
    return parser.parse_args()


def load_saved_data(data_dir):
    """Load saved OOD detection data"""
    print(f"Loading data from {data_dir}...")
    
    # Load features and labels
    train_features = np.load(data_dir / 'train_features.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    id_features = np.load(data_dir / 'id_features.npy')
    id_labels = np.load(data_dir / 'id_labels.npy')
    ood_features = np.load(data_dir / 'ood_features.npy')
    ood_labels = np.load(data_dir / 'ood_labels.npy')  # Unified label
    ood_original_labels = np.load(data_dir / 'ood_original_labels.npy')  # Original labels
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded data:")
    print(f"  - Train features: {train_features.shape}")
    print(f"  - ID features: {id_features.shape}")
    print(f"  - OOD features: {ood_features.shape}")
    print(f"  - Num known classes: {metadata['num_classes']}")
    
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'id_features': id_features,
        'id_labels': id_labels,
        'ood_features': ood_features,
        'ood_labels': ood_labels,
        'ood_original_labels': ood_original_labels,
        'metadata': metadata
    }


def run_gcd_phase1(data, args):
    """
    Run 1st phase of GCD:
    - Use training data as labeled exemplars (known classes)
    - Use OOD samples as unlabeled data (all treated as unknown class)
    - Goal: Estimate the number of novel classes and cluster them
    """
    print("\n" + "="*60)
    print("1st PHASE: GENERALIZED NOVEL CATEGORY DISCOVERY")
    print("="*60)
    
    # Prepare data
    # Labeled: training samples (known classes, labels 0 to num_classes-1)
    labeled_feats = data['train_features']
    labeled_targets = data['train_labels']
    
    # Unlabeled: OOD samples (treated as unknown class = num_classes)
    # In experiment setting, we don't know the true labels of OOD samples
    unlabeled_feats = data['ood_features']
    # The unified label for all OOD samples (class num_classes, i.e., class 23 for 23-class problem)
    unlabeled_unified_labels = data['ood_labels']  # All set to num_classes
    # True labels are only used for evaluation, not for clustering
    unlabeled_true_labels = data['ood_original_labels']
    
    num_known_class = data['metadata']['num_classes']
    num_true_novel_classes = len(np.unique(unlabeled_true_labels))
    
    print(f"\nData summary:")
    print(f"  - Labeled (known) samples: {len(labeled_feats)}")
    print(f"  - Unlabeled (OOD) samples: {len(unlabeled_feats)}")
    print(f"  - Known classes: {num_known_class} (labels 0-{num_known_class-1})")
    print(f"  - OOD unified label: {num_known_class} (treated as single unknown class)")
    print(f"  - True novel classes (for evaluation only): {num_true_novel_classes}")
    
    # Run GNCD
    # Note: unlabeled samples are treated as a single unknown class (label = num_known_class)
    # The goal is to discover how many clusters exist within these unknown samples
    all_predictions, results = gncd_simple(
        labeled_feats=labeled_feats,
        labeled_targets=labeled_targets,
        unlabeled_feats=unlabeled_feats,
        unlabeled_true_labels=unlabeled_true_labels,  # Only for evaluation
        num_known_class=num_known_class,
        args=args,
        phase='1st'
    )
    
    return all_predictions, results


def manual_intervention(data, predictions, args):
    """
    1st MANUAL INTERVENTION:
    - Identify novel cluster samples
    - Split into training and test sets
    - Prepare data for next phase (if needed)
    """
    print("\n" + "="*60)
    print("1st MANUAL INTERVENTION")
    print("="*60)
    
    num_known_class = data['metadata']['num_classes']
    num_labeled = len(data['train_labels'])
    
    # Get predictions for unlabeled (novel) samples
    unlabeled_pred = predictions[num_labeled:]
    
    # Identify samples predicted as novel classes (cluster ID >= num_known_class)
    mask_novel = unlabeled_pred >= num_known_class
    
    print(f"\nNovel cluster analysis:")
    print(f"  - Total unlabeled samples: {len(unlabeled_pred)}")
    print(f"  - Samples in novel clusters: {mask_novel.sum()}")
    print(f"  - Samples assigned to known classes: {(~mask_novel).sum()}")
    
    # Get novel cluster samples
    novel_feats = data['ood_features'][mask_novel]
    novel_true_labels = data['ood_original_labels'][mask_novel]
    novel_pred_labels = unlabeled_pred[mask_novel]
    
    if len(novel_feats) > 0:
        # Split novel samples into train/test
        train_feats_novel, test_feats_novel, train_labels_novel, test_labels_novel = train_test_split(
            novel_feats, novel_true_labels,
            random_state=104,
            test_size=1 - args.prop_train_labels,
            shuffle=True
        )
        
        print(f"\nNovel samples split:")
        print(f"  - Training: {len(train_feats_novel)}")
        print(f"  - Testing: {len(test_feats_novel)}")
        
        # Combine with existing training data
        train_feats_available = np.concatenate([data['train_features'], train_feats_novel], axis=0)
        train_targets_available = np.concatenate([data['train_labels'], train_labels_novel])
        
        # Combine test sets
        test_feats_available = np.concatenate([data['id_features'], test_feats_novel], axis=0)
        test_targets_available = np.concatenate([data['id_labels'], test_labels_novel])
        
        print(f"\nUpdated data:")
        print(f"  - Training samples: {len(train_feats_available)}")
        print(f"  - Testing samples: {len(test_feats_available)}")
        
        return {
            'train_features': train_feats_available,
            'train_labels': train_targets_available,
            'test_features': test_feats_available,
            'test_labels': test_targets_available,
            'novel_train_features': train_feats_novel,
            'novel_train_labels': train_labels_novel,
            'novel_test_features': test_feats_novel,
            'novel_test_labels': test_labels_novel
        }
    else:
        print("No samples assigned to novel clusters!")
        return None


def visualize_clustering_results(data, predictions, results, save_dir, args):
    """
    Visualize clustering results using t-SNE and save as images.
    
    Args:
        data: dictionary containing features and labels
        predictions: cluster predictions for all samples
        results: GCD results dictionary
        save_dir: directory to save figures
        args: arguments
    """
    print("\n" + "="*60)
    print("VISUALIZING CLUSTERING RESULTS")
    print("="*60)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_known_class = data['metadata']['num_classes']
    best_K = results['best_K']
    num_novel_clusters = best_K - num_known_class
    
    # Combine all features
    all_features = np.concatenate([data['train_features'], data['ood_features']], axis=0)
    num_labeled = len(data['train_labels'])
    
    # True labels (remap OOD labels to be after known classes)
    true_labels_known = data['train_labels']
    true_labels_novel = data['ood_original_labels'] + num_known_class
    all_true_labels = np.concatenate([true_labels_known, true_labels_novel])
    
    # Sample type: 0=known, 1=novel
    sample_types = np.concatenate([
        np.zeros(num_labeled, dtype=int),
        np.ones(len(data['ood_features']), dtype=int)
    ])
    
    print(f"Total samples: {len(all_features)}")
    print(f"  - Known class samples: {num_labeled}")
    print(f"  - Novel class samples: {len(data['ood_features'])}")
    print(f"Estimated K: {best_K} (known: {num_known_class}, novel clusters: {num_novel_clusters})")
    
    # Subsample for t-SNE if data is too large
    max_samples = 10000  # Limit for t-SNE visualization
    if len(all_features) > max_samples:
        print(f"\nSubsampling to {max_samples} samples for t-SNE visualization...")
        np.random.seed(args.seed)
        
        # Stratified sampling to preserve class distribution
        indices = []
        unique_preds = np.unique(predictions)
        samples_per_class = max(10, max_samples // len(unique_preds))
        
        for pred in unique_preds:
            class_indices = np.where(predictions == pred)[0]
            n_samples = min(len(class_indices), samples_per_class)
            selected = np.random.choice(class_indices, size=n_samples, replace=False)
            indices.extend(selected)
        
        # If still need more samples, add randomly
        if len(indices) < max_samples:
            remaining = list(set(range(len(all_features))) - set(indices))
            additional = np.random.choice(remaining, size=min(max_samples - len(indices), len(remaining)), replace=False)
            indices.extend(additional)
        
        indices = np.array(indices[:max_samples])
        
        all_features_vis = all_features[indices]
        predictions_vis = predictions[indices]
        all_true_labels_vis = all_true_labels[indices]
        sample_types_vis = sample_types[indices]
        print(f"  Subsampled to {len(indices)} samples")
    else:
        all_features_vis = all_features
        predictions_vis = predictions
        all_true_labels_vis = all_true_labels
        sample_types_vis = sample_types
    
    # t-SNE dimensionality reduction with limited threads
    print("\nPerforming t-SNE dimensionality reduction...")
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000, n_jobs=1)
    features_2d = tsne.fit_transform(all_features_vis)
    print("t-SNE completed.")
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Create consistent color maps for known and novel classes
    # Use the same colormap for both clustering_predictions and true_labels_distribution
    unique_true_labels_all = np.unique(all_true_labels_vis)
    num_total_classes = len(unique_true_labels_all)
    cmap_all = plt.cm.get_cmap('tab20', min(num_total_classes, 20))
    
    # Build color dictionary for known classes (consistent across figures)
    known_colors = {}
    for i, label in enumerate(unique_true_labels_all):
        if label < num_known_class:
            known_colors[label] = cmap_all(i % 20)
    
    # Novel cluster colors
    cmap_novel = plt.cm.get_cmap('Set3', max(num_novel_clusters, 12))
    
    # Figure 1: Clustering predictions
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Plot known class predictions (use consistent colors from known_colors)
    for k in range(num_known_class):
        mask = predictions_vis == k
        if mask.sum() > 0:
            color = known_colors.get(k, cmap_all(k % 20))
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], label=f'Known-{k}', alpha=0.6, s=15)
    
    # Plot novel cluster predictions
    for k in range(num_known_class, best_K):
        mask = predictions_vis == k
        if mask.sum() > 0:
            novel_idx = k - num_known_class
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[cmap_novel(novel_idx % 12)], marker='x', 
                       label=f'Novel-{novel_idx}', alpha=0.7, s=20)
    
    # ax1.set_title(f'GCD Clustering Results\nEstimated K={best_K} (Known: {num_known_class}, Novel Clusters: {num_novel_clusters})', 
    #               fontsize=14, fontweight='bold', fontname='Times New Roman')
    ax1.set_xlabel('TSNE-1', fontsize=22, fontname='Times New Roman')
    ax1.set_ylabel('TSNE-2', fontsize=22, fontname='Times New Roman')
    ax1.tick_params(axis='both', labelsize=18)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    # Add legend inside the plot (upper right corner)
    if best_K <= 20:
        legend = ax1.legend(loc='upper right', fontsize=16, framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    fig1.savefig(save_dir / 'clustering_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'clustering_predictions.png'}")
    plt.close(fig1)
    
    # Figure 2: Known vs Novel samples (by sample type)
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    known_mask = sample_types_vis == 0
    novel_mask = sample_types_vis == 1
    
    ax2.scatter(features_2d[known_mask, 0], features_2d[known_mask, 1], 
               c='blue', label=f'Known Classes', alpha=0.5, s=15)
    ax2.scatter(features_2d[novel_mask, 0], features_2d[novel_mask, 1], 
               c='red', label=f'Novel Classes', alpha=0.5, s=15, marker='x')
    
    # ax2.set_title(f'Known vs Novel Class Distribution\nEstimated K={best_K}', fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax2.set_xlabel('TSNE-1', fontsize=22, fontname='Times New Roman')
    ax2.set_ylabel('TSNE-2', fontsize=22, fontname='Times New Roman')
    ax2.tick_params(axis='both', labelsize=18)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname('Times New Roman')
    legend2 = ax2.legend(loc='upper right', fontsize=16, framealpha=0.9)
    for text in legend2.get_texts():
        text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    fig2.savefig(save_dir / 'known_vs_novel_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'known_vs_novel_distribution.png'}")
    plt.close(fig2)
    
    # Figure 3: True labels visualization (use same colormap as clustering_predictions)
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    for i, label in enumerate(unique_true_labels_all):
        mask = all_true_labels_vis == label
        if mask.sum() > 0:
            color = cmap_all(i % 20)
            if label < num_known_class:
                ax3.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[color], label=f'Known-{label}', alpha=0.6, s=15)
            else:
                ax3.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[color], marker='x', 
                           label=f'Novel-{label - num_known_class}', alpha=0.7, s=20)
    
    # ax3.set_title(f'True Labels Distribution\nKnown: {num_known_class}, True Novel: {results["num_true_novel_classes"]}', 
    #               fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax3.set_xlabel('TSNE-1', fontsize=22, fontname='Times New Roman')
    ax3.set_ylabel('TSNE-2', fontsize=22, fontname='Times New Roman')
    ax3.tick_params(axis='both', labelsize=18)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    # Add legend inside the plot (upper right corner)
    if num_total_classes <= 20:
        legend3 = ax3.legend(loc='upper right', fontsize=16, framealpha=0.9)
        for text in legend3.get_texts():
            text.set_fontname('Times New Roman')
    
    plt.tight_layout()
    fig3.savefig(save_dir / 'true_labels_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'true_labels_distribution.png'}")
    plt.close(fig3)
    
    # Figure 4: Summary statistics bar chart
    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy metrics
    metrics = ['Known Acc', 'Novel Acc', 'Harmonic Acc']
    values = [results['known_acc'], results['novel_acc'], results['harmonic_acc']]
    colors = ['#2ecc71', '#e74c3c', '#9b59b6']
    
    axes[0].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Accuracy', fontsize=18, fontname='Times New Roman')
    axes[0].set_title('Clustering Accuracy Metrics', fontsize=18, fontweight='bold', fontname='Times New Roman')
    axes[0].tick_params(axis='both', labelsize=18)
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_fontname('Times New Roman')
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold', fontname='Times New Roman')
    
    # NMI/ARI metrics
    metrics2 = ['Known NMI', 'Novel NMI', 'Overall NMI', 'Overall ARI']
    values2 = [results['known_nmi'], results['novel_nmi'], results['overall_nmi'], results['overall_ari']]
    colors2 = ['#3498db', '#e67e22', '#1abc9c', '#f39c12']
    
    axes[1].bar(metrics2, values2, color=colors2, edgecolor='black', linewidth=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Score', fontsize=18, fontname='Times New Roman')
    axes[1].set_title('Clustering Quality Metrics', fontsize=18, fontweight='bold', fontname='Times New Roman')
    axes[1].tick_params(axis='both', labelsize=18)
    for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        label.set_fontname('Times New Roman')
    for i, v in enumerate(values2):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold', fontname='Times New Roman')
    
    # plt.suptitle(f'GCD Results Summary - Estimated K={best_K} (Known: {num_known_class}, Novel: {num_novel_clusters})', 
    #              fontsize=20, fontweight='bold', y=1.02, fontname='Times New Roman')
    plt.tight_layout()
    fig4.savefig(save_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'metrics_summary.png'}")
    plt.close(fig4)
    
    print("\nVisualization completed!")


def save_results(results, intervention_data, save_dir, args):
    """Save all results"""
    print(f"\nSaving results to {save_dir}...")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save GCD results
    results_to_save = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results.items() if k != 'confusion_matrix'}
    
    with open(save_dir / 'gcd_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save intervention data
    if intervention_data is not None:
        np.save(save_dir / 'train_features_updated.npy', intervention_data['train_features'])
        np.save(save_dir / 'train_labels_updated.npy', intervention_data['train_labels'])
        np.save(save_dir / 'test_features_updated.npy', intervention_data['test_features'])
        np.save(save_dir / 'test_labels_updated.npy', intervention_data['test_labels'])
        
        if 'novel_train_features' in intervention_data:
            np.save(save_dir / 'novel_train_features.npy', intervention_data['novel_train_features'])
            np.save(save_dir / 'novel_train_labels.npy', intervention_data['novel_train_labels'])
            np.save(save_dir / 'novel_test_features.npy', intervention_data['novel_test_features'])
            np.save(save_dir / 'novel_test_labels.npy', intervention_data['novel_test_labels'])
    
    # Save configuration
    config = {
        'dataset': args.dataset,
        'noisy_dataset': args.noisy_dataset,
        'noise_ratio': args.noise_ratio,
        'open_ratio': args.open_ratio,
        'max_K': args.max_K,
        'max_kmeans_iter': args.max_kmeans_iter,
        'k_means_init': args.k_means_init,
        'use_brent_optimization': args.use_brent_optimization,
        'prop_train_labels': args.prop_train_labels,
        'seed': args.seed
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Results saved successfully!")


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    
    # Set random seed
    seed_torch(args.seed)
    
    print("="*60)
    print("GCD PIPELINE - Generalized Category Discovery")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Noisy dataset: {args.noisy_dataset}")
    print(f"Noise ratio: {args.noise_ratio}")
    print(f"Open ratio: {args.open_ratio}")
    print(f"Max K: {args.max_K}")
    
    # Set up paths
    data_dir = Path('/home/ju/Desktop/TNSE/Sieve/Unknown_Traffic_Labeling/data') / f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noisy_dataset}'
    results_dir = Path('/home/ju/Desktop/TNSE/Sieve/Unknown_Traffic_Labeling/results') / f'{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noisy_dataset}'
    
    # Check if data exists
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please run save_ood_samples.py first to generate the data.")
        print("\nExample:")
        print(f"  python save_ood_samples.py --dataset {args.dataset} --noisy_dataset {args.noisy_dataset} --noise_ratio {args.noise_ratio} --open_ratio {args.open_ratio}")
        return
    
    # Load saved data
    data = load_saved_data(data_dir)
    
    # Run 1st phase: GNCD
    predictions, results = run_gcd_phase1(data, args)
    
    # Manual intervention
    intervention_data = manual_intervention(data, predictions, args)
    
    # Visualize clustering results
    visualize_clustering_results(data, predictions, results, results_dir, args)
    
    # Save results
    save_results(results, intervention_data, results_dir, args)
    
    # Print final summary
    print("\n" + "="*60)
    print("GCD PIPELINE COMPLETED")
    print("="*60)
    print(f"\nResults Summary:")
    print(f"  - Best K (estimated clusters): {results['best_K']}")
    print(f"  - Known classes accuracy: {results['known_acc']:.4f}")
    print(f"  - Novel classes accuracy: {results['novel_acc']:.4f}")
    print(f"  - Harmonic accuracy: {results['harmonic_acc']:.4f}")
    print(f"  - Overall NMI: {results['overall_nmi']:.4f}")
    print(f"  - Overall ARI: {results['overall_ari']:.4f}")
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
