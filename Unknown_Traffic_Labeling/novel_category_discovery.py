import numpy as np
import torch
import math
from functools import partial
from scipy.optimize import minimize_scalar
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import davies_bouldin_score
from sklearn.metrics.cluster import silhouette_score

from utils.faster_mix_k_means import K_Means as SemiSupKMeans
from utils.metrics import evaluation_novel, evaluation_gcd_all
from utils.cluster_utils import cluster_acc


def ss_kmeans(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, K, args):
    """
    Perform semi-supervised K-Means and evaluate.
    
    Args:
        labeled_feats: labeled data features (n1*m)
        labeled_targets: labeled data labels (n1*1)
        unlabeled_feats: unlabeled data features (n2*m)
        num_known_class: number of known classes
        K: total number of classes
        args: various parameters
    
    Returns:
        ACC: Fowlkes-Mallows score on validation set
        DBS: Davies-Bouldin score on unlabeled set
    """
    # Split labeled data into anchor probe set and validation probe set (2:1)
    num_class_anchor = math.ceil(2 * num_known_class / 3)
    mask_probe = np.zeros(len(labeled_targets), dtype=int)
    mask_probe[labeled_targets >= num_class_anchor] = 1
    anchor_feats = labeled_feats[mask_probe == 0]
    val_feats = labeled_feats[mask_probe == 1]
    anchor_targets = labeled_targets[mask_probe == 0]
    val_targets = labeled_targets[mask_probe == 1]

    # Merge validation probe set and unknown unlabeled data
    u_feats = np.concatenate((val_feats, unlabeled_feats), axis=0)
    mask = np.concatenate((mask_probe, 2 * np.ones(unlabeled_feats.shape[0], dtype=int)))

    # Perform semi-supervised k-means clustering
    print(f'Performing semi-supervised k-means (k={K})')
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats_t, l_targets = (torch.from_numpy(x).float().to(device) for x in
                                      (anchor_feats, u_feats, anchor_targets))
    
    ss_k_means = SemiSupKMeans(
        k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
        n_init=args.k_means_init, random_state=None, n_jobs=None, 
        pairwise_batch_size=args.pairwise_batch_size, mode=None
    )
    ss_k_means.fit_mix(u_feats_t, l_feats, l_targets.long())

    # Get predictions
    all_pred = ss_k_means.labels_.cpu().numpy()
    val_pred = all_pred[mask == 1]
    unlabeled_pred = all_pred[mask == 2]

    # Evaluate clustering results
    ACC = fowlkes_mallows_score(val_targets, val_pred)
    DBS = davies_bouldin_score(unlabeled_feats, unlabeled_pred)
    
    return ACC, DBS


def ss_kmeans_for_search(K, labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args):
    """
    Semi-supervised K-Means for hyperparameter search.
    Returns negative score for minimization.
    """
    # Split labeled data
    num_class_anchor = math.ceil(2 * num_known_class / 3)
    mask_probe = np.zeros(len(labeled_targets), dtype=int)
    mask_probe[labeled_targets >= num_class_anchor] = 1
    anchor_feats = labeled_feats[mask_probe == 0]
    val_feats = labeled_feats[mask_probe == 1]
    anchor_targets = labeled_targets[mask_probe == 0]
    val_targets = labeled_targets[mask_probe == 1]

    # Merge validation probe set and unknown unlabeled data
    u_feats = np.concatenate((val_feats, unlabeled_feats), axis=0)
    mask = np.concatenate((mask_probe, 2 * np.ones(unlabeled_feats.shape[0], dtype=int)))

    # Perform semi-supervised k-means clustering
    K = int(K)
    print(f'Performing semi-supervised k-means (k={K})')
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats_t, l_targets = (torch.from_numpy(x).float().to(device) for x in
                                      (anchor_feats, u_feats, anchor_targets))
    
    ss_k_means = SemiSupKMeans(
        k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
        n_init=args.k_means_init, random_state=None, n_jobs=None,
        pairwise_batch_size=args.pairwise_batch_size, mode=None
    )
    ss_k_means.fit_mix(u_feats_t, l_feats, l_targets.long())

    # Get predictions
    all_pred = ss_k_means.labels_.cpu().numpy()
    val_pred = all_pred[mask == 1]
    unlabeled_pred = all_pred[mask == 2]

    # Evaluate clustering results
    ACC = fowlkes_mallows_score(val_targets, val_pred)
    DBS = davies_bouldin_score(unlabeled_feats, unlabeled_pred)
    
    print(f'K:{K}, ACC:{ACC:.4f}, DBS:{DBS:.4f}')
    return -(ACC - DBS)


def subsample_data(feats, targets, ratio=0.3, random_state=42):
    """Subsample data for faster K estimation while preserving class distribution"""
    np.random.seed(random_state)
    
    if ratio >= 1.0:
        return feats, targets
    
    unique_classes = np.unique(targets)
    selected_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        n_samples = max(1, int(len(cls_indices) * ratio))
        selected = np.random.choice(cls_indices, size=n_samples, replace=False)
        selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    return feats[selected_indices], targets[selected_indices]


def estimate_k(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args):
    """
    Estimate the total number of classes K using grid search.
    
    Args:
        labeled_feats: labeled data features
        labeled_targets: labeled data labels
        unlabeled_feats: unlabeled data features
        num_known_class: number of known classes
        args: various parameters
    
    Returns:
        best_K: estimated total number of classes
    """
    # Subsample for faster estimation
    subsample_ratio = getattr(args, 'subsample_ratio', 0.3)
    if subsample_ratio < 1.0:
        print(f"  Subsampling data with ratio {subsample_ratio} for faster K estimation...")
        labeled_feats_sub, labeled_targets_sub = subsample_data(labeled_feats, labeled_targets, subsample_ratio)
        # For unlabeled, just random sample
        n_unlabeled = max(100, int(len(unlabeled_feats) * subsample_ratio))
        unlabeled_indices = np.random.choice(len(unlabeled_feats), size=n_unlabeled, replace=False)
        unlabeled_feats_sub = unlabeled_feats[unlabeled_indices]
        print(f"  Subsampled: labeled {len(labeled_feats_sub)}, unlabeled {len(unlabeled_feats_sub)}")
    else:
        labeled_feats_sub, labeled_targets_sub = labeled_feats, labeled_targets
        unlabeled_feats_sub = unlabeled_feats
    
    records_ACC = {}
    records_DBS = {}
    
    for K in range(num_known_class + 1, args.max_K + 1):
        ACC, DBS = ss_kmeans(labeled_feats_sub, labeled_targets_sub, unlabeled_feats_sub, num_known_class, K, args)
        records_ACC[K] = ACC
        records_DBS[K] = DBS
    
    best_K_ACC = max(records_ACC, key=lambda k: records_ACC[k])
    best_K_DBS = min(records_DBS, key=lambda k: records_DBS[k])  # Lower DBS is better
    best_K = math.ceil((best_K_ACC + best_K_DBS) / 2)
    
    print(f"\nK estimation results:")
    print(f"  ACC scores: {records_ACC}")
    print(f"  DBS scores: {records_DBS}")
    print(f"  Best K by ACC: {best_K_ACC}, Best K by DBS: {best_K_DBS}")
    print(f"  Final estimated K: {best_K}")
    
    return best_K


def estimate_k_bybrent(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args):
    """
    Estimate K using Brent's method for optimization.
    
    Args:
        labeled_feats: labeled data features
        labeled_targets: labeled data labels
        unlabeled_feats: unlabeled data features
        num_known_class: number of known classes
        args: various parameters
    
    Returns:
        best_K: estimated total number of classes
    """
    test_k_means_partial = partial(
        ss_kmeans_for_search, 
        labeled_feats=labeled_feats, 
        labeled_targets=labeled_targets, 
        unlabeled_feats=unlabeled_feats, 
        num_known_class=num_known_class, 
        args=args
    )
    
    res = minimize_scalar(
        test_k_means_partial, 
        bounds=(num_known_class + 1, args.max_K), 
        method='bounded', 
        options={'disp': True}, 
        tol=1
    )
    
    best_k = int(res.x)
    print(f"The best K is {best_k}")
    return best_k


def gncd(train_feats_exemplar, train_targets_exemplar, unknown_feats, 
         predict_label_osr, online_targets_osr, num_known_class, args, phase='1st'):
    """
    Generalized Novel Category Discovery.
    
    Args:
        train_feats_exemplar: labeled exemplar features (n1*m)
        train_targets_exemplar: labeled exemplar labels (n1*1)
        unknown_feats: isolated unknown features (unlabeled) (n2*m)
        predict_label_osr: labels predicted by open-set recognition (n3*1)
        online_targets_osr: ground true labels of online data (n3*1)
        num_known_class: number of known classes
        args: various parameters
        phase: current phase name
    
    Returns:
        predict_label_ncd: updated predictions after NCD
        results: evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Generalized Novel Category Discovery - {phase}")
    print(f"{'='*60}")
    
    # Prepare known labeled data and unknown unlabeled data
    labeled_feats = train_feats_exemplar
    labeled_targets = train_targets_exemplar
    unlabeled_feats = unknown_feats

    print(f"Labeled samples: {len(labeled_feats)}")
    print(f"Unlabeled samples: {len(unlabeled_feats)}")
    print(f"Known classes: {num_known_class}")

    # Estimate total number of classes K
    print('\nEstimating the total number of classes K...')
    if args.use_brent_optimization:
        best_K = estimate_k_bybrent(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args)
    else:
        best_K = estimate_k(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args)

    # Perform semi-supervised k-means clustering based on the best K
    print(f'\nPerforming semi-supervised k-means with best K={best_K}')
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats, l_targets = (torch.from_numpy(x).float().to(device) for x in
                                    (labeled_feats, unlabeled_feats, labeled_targets))
    
    ss_k_means = SemiSupKMeans(
        k=best_K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
        n_init=args.k_means_init, random_state=None, n_jobs=None,
        pairwise_batch_size=args.pairwise_batch_size, mode=None
    )
    ss_k_means.fit_mix(u_feats, l_feats, l_targets.long())

    # Get predictions for unknown unlabeled data
    all_pred = ss_k_means.labels_.cpu().numpy()
    unlabeled_pred = all_pred[len(labeled_targets):]

    # Update predictions: assign cluster labels to unknown samples
    predict_label_ncd = predict_label_osr.copy()
    predict_label_ncd[predict_label_osr == num_known_class] = unlabeled_pred

    # Evaluate generalized novel category discovery results
    print("\n--- Evaluation Results ---")
    results = evaluation_novel(online_targets_osr, predict_label_ncd, num_known_class, phase)
    
    # Additional comprehensive evaluation
    all_results = evaluation_gcd_all(online_targets_osr, predict_label_ncd, num_known_class)
    results.update(all_results)
    
    results['best_K'] = best_K
    results['num_known_class'] = num_known_class
    results['num_labeled'] = len(labeled_feats)
    results['num_unlabeled'] = len(unlabeled_feats)

    return predict_label_ncd, results


def gncd_simple(labeled_feats, labeled_targets, unlabeled_feats, unlabeled_true_labels, 
                num_known_class, args, phase='1st'):
    """
    Simplified GNCD for direct use with saved OOD samples.
    
    In the GCD setting:
    - labeled_feats/labeled_targets: Known class samples (labels 0 to num_known_class-1)
    - unlabeled_feats: OOD samples detected by open-set recognition (treated as unknown)
    - unlabeled_true_labels: True labels of OOD samples (only for evaluation, not used in clustering)
    
    Goal: Estimate the total number of classes K and cluster all samples.
    
    Args:
        labeled_feats: labeled (ID) features with known class labels
        labeled_targets: labeled (ID) labels (0 to num_known_class-1)
        unlabeled_feats: unlabeled (OOD) features (treated as unknown class)
        unlabeled_true_labels: true labels of unlabeled data (for evaluation only)
        num_known_class: number of known classes
        args: various parameters
        phase: current phase name
    
    Returns:
        all_predictions: predictions for all samples (labeled + unlabeled)
        results: evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Generalized Novel Category Discovery - {phase}")
    print(f"{'='*60}")
    
    # Verify labeled data only contains known classes
    assert np.all(labeled_targets < num_known_class), \
        f"Labeled data should only contain known classes (0-{num_known_class-1}), but found {np.unique(labeled_targets)}"
    
    num_true_novel_classes = len(np.unique(unlabeled_true_labels))
    
    print(f"\nData Summary:")
    print(f"  Labeled (known) samples: {len(labeled_feats)}")
    print(f"  Unlabeled (OOD) samples: {len(unlabeled_feats)}")
    print(f"  Known classes: {num_known_class} (labels 0-{num_known_class-1})")
    print(f"  True novel classes (for evaluation): {num_true_novel_classes}")
    print(f"  Unique labels in labeled set: {np.unique(labeled_targets)}")

    # Estimate total number of classes K
    print('\nEstimating the total number of classes K...')
    print(f'  Search range: [{num_known_class + 1}, {args.max_K}]')
    
    if args.use_brent_optimization:
        best_K = estimate_k_bybrent(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args)
    else:
        best_K = estimate_k(labeled_feats, labeled_targets, unlabeled_feats, num_known_class, args)
    
    num_novel_clusters = best_K - num_known_class
    print(f'\nEstimated K = {best_K} (known: {num_known_class}, novel: {num_novel_clusters})')
    print(f'True novel classes: {num_true_novel_classes}')

    # Perform semi-supervised k-means clustering
    print(f'\nPerforming final semi-supervised k-means with K={best_K}...')
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    l_feats, u_feats, l_targets = (torch.from_numpy(x).float().to(device) for x in
                                    (labeled_feats, unlabeled_feats, labeled_targets))
    
    ss_k_means = SemiSupKMeans(
        k=best_K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
        n_init=args.k_means_init, random_state=None, n_jobs=None,
        pairwise_batch_size=args.pairwise_batch_size, mode=None
    )
    ss_k_means.fit_mix(u_feats, l_feats, l_targets.long())

    # Get all predictions
    all_pred = ss_k_means.labels_.cpu().numpy()
    labeled_pred = all_pred[:len(labeled_targets)]
    unlabeled_pred = all_pred[len(labeled_targets):]

    # Remap OOD true labels to be in the range [num_known_class, num_known_class + num_novel_classes)
    # e.g., OOD labels 0-14 -> 23-37 (for 23 known classes)
    # This puts all labels in the same label space for proper evaluation
    unlabeled_true_labels_remapped = unlabeled_true_labels + num_known_class
    
    # Combine true labels for evaluation (all in unified label space)
    all_true_labels = np.concatenate([labeled_targets, unlabeled_true_labels_remapped])
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # Labeled (known) samples accuracy
    labeled_acc = (labeled_pred == labeled_targets).mean()
    print(f"Known class accuracy (labeled samples): {labeled_acc:.4f}")
    
    # Clustering accuracy on unlabeled (novel) samples
    # Note: unlabeled_true_labels_remapped are in range [num_known_class, ...]
    # We use cluster_acc which uses Hungarian algorithm to find best matching
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    from sklearn.metrics import adjusted_rand_score as ari_score
    
    # Use remapped labels for novel class evaluation (labels 23-37 for 15 novel classes)
    unlabeled_acc = cluster_acc(unlabeled_true_labels_remapped.astype(int), unlabeled_pred.astype(int))
    unlabeled_nmi = nmi_score(unlabeled_true_labels_remapped, unlabeled_pred)
    unlabeled_ari = ari_score(unlabeled_true_labels_remapped, unlabeled_pred)
    print(f"Novel class clustering accuracy: {unlabeled_acc:.4f}")
    print(f"Novel class NMI: {unlabeled_nmi:.4f}")
    print(f"Novel class ARI: {unlabeled_ari:.4f}")
    
    # Known class metrics (labeled samples)
    known_nmi = nmi_score(labeled_targets, labeled_pred)
    known_ari = ari_score(labeled_targets, labeled_pred)
    print(f"Known class NMI: {known_nmi:.4f}")
    print(f"Known class ARI: {known_ari:.4f}")
    
    # Overall clustering accuracy (note: labels are in different spaces, so this is approximate)
    # For a fair comparison, we compute overall metrics on the predictions
    overall_nmi = nmi_score(all_true_labels, all_pred)
    overall_ari = ari_score(all_true_labels, all_pred)
    print(f"Overall NMI: {overall_nmi:.4f}")
    print(f"Overall ARI: {overall_ari:.4f}")
    
    # Harmonic mean of known and novel accuracy
    if labeled_acc > 0 and unlabeled_acc > 0:
        harmonic_acc = 2 / (1/labeled_acc + 1/unlabeled_acc)
    else:
        harmonic_acc = 0
    print(f"Harmonic accuracy: {harmonic_acc:.4f}")
    
    # Build results dictionary
    results = {
        # Known class metrics (based on labeled samples)
        'known_acc': float(labeled_acc),
        'known_nmi': float(known_nmi),
        'known_ari': float(known_ari),
        # Novel class metrics (based on unlabeled/OOD samples)
        'novel_acc': float(unlabeled_acc),
        'novel_nmi': float(unlabeled_nmi),
        'novel_ari': float(unlabeled_ari),
        # Overall metrics
        'overall_nmi': float(overall_nmi),
        'overall_ari': float(overall_ari),
        # Harmonic mean
        'harmonic_acc': float(harmonic_acc),
        # Additional info
        'labeled_acc': float(labeled_acc),
        'unlabeled_acc': float(unlabeled_acc),
        'best_K': int(best_K),
        'num_known_class': int(num_known_class),
        'num_novel_clusters': int(num_novel_clusters),
        'num_true_novel_classes': int(num_true_novel_classes),
        'num_labeled': len(labeled_feats),
        'num_unlabeled': len(unlabeled_feats)
    }
    
    print(f"\n--- Summary ---")
    print(f"Estimated K: {best_K} (known: {num_known_class}, novel clusters: {num_novel_clusters})")
    print(f"True novel classes: {num_true_novel_classes}")
    print(f"Known class accuracy: {results['known_acc']:.4f}")
    print(f"Novel class accuracy: {results['novel_acc']:.4f}")
    print(f"Harmonic accuracy: {results['harmonic_acc']:.4f}")
    print(f"Overall NMI: {results['overall_nmi']:.4f}")
    print(f"Overall ARI: {results['overall_ari']:.4f}")

    return all_pred, results
