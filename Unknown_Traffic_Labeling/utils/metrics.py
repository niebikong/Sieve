from .cluster_utils import cluster_acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np


def metrics_HNA(labels_true, labels_pre, unknown_label):
    """
    Calculate Harmonic Normalized Accuracy (HNA) for open-set recognition.
    
    Args:
        labels_true: ground-true open-set label n*1
        labels_pre: predicted open-set label n*1
        unknown_label: label of unknown class = num_known_class
    
    Returns:
        HNA score
    """
    labels_true = labels_true.reshape(-1)
    index_known = np.where(labels_true != unknown_label)
    index_unknown = np.where(labels_true == unknown_label)
    
    AKS = accuracy_score(labels_true[index_known], labels_pre[index_known])  # Accuracy of known classes
    AUS = accuracy_score(labels_true[index_unknown], labels_pre[index_unknown])  # Accuracy of unknown classes
    
    if AKS == 0 or AUS == 0 or np.isnan(AKS) or np.isnan(AUS):
        HNA = 0
    else:
        HNA = 2 / (1 / AKS + 1 / AUS)

    return HNA


def metrics_OSFM(cm):
    """
    Calculate Weighted-averaging open-set f-measure (OSFMw).
    
    Args:
        cm: Confusion Matrix (num_known_class+1) * (num_known_class+1)
    
    Returns:
        OSFMw score
    """
    sum_rows, sum_cols = cm.sum(axis=1), cm.sum(axis=0)

    weights = normalization(sum_rows[:-1])
    f_measure_w = 0
    for i in range(cm.shape[0] - 1):
        tp_i = cm[i][i]
        fp_i = sum_cols[i]
        fn_i = sum_rows[i]
        precision_w = tp_i / np.maximum(fp_i, 0.001)
        recall_w = tp_i / np.maximum(fn_i, 0.001)
        f_measure_w += weights[i] * (2 * precision_w * recall_w) / np.maximum((precision_w + recall_w), 0.001)

    return f_measure_w


def normalization(data):
    """Normalize data to sum to 1"""
    if len(data.shape) == 2:
        _sum = (np.ones((data.shape[1], data.shape[0])) * np.sum(data, axis=1)).T
    else:
        _sum = np.sum(data)
    return data / _sum


def evaluation_closed(y_true, y_pred, num_known_classes):
    """
    Evaluate closed-set recognition performance.
    
    Args:
        y_true: ground true label n*1
        y_pred: predicted label n*1
        num_known_classes: number of known classes
    
    Returns:
        Dictionary with accuracy, f1, and confusion matrix
    """
    print("Performing closed-set recognition evaluation")
    
    condition = y_true < num_known_classes
    acc = accuracy_score(y_true[condition], y_pred[condition])
    f1 = f1_score(y_true[condition], y_pred[condition], average="weighted")
    cm = confusion_matrix(y_true[condition], y_pred[condition])

    print(f"Accuracy on known classes: {acc:.3f}")
    print(f"F1-Score on known classes: {f1:.3f}")
    
    return {'accuracy': acc, 'f1': f1, 'confusion_matrix': cm}


def evaluation_open(y_true, y_pred, unknown_label):
    """
    Evaluate open-set recognition performance.
    
    Args:
        y_true: ground-true open-set label n*1
        y_pred: predicted open-set label n*1
        unknown_label: label of unknown class
    
    Returns:
        HNA, OSFM scores
    """
    print("Performing open-set recognition evaluation")
    cm = confusion_matrix(y_true, y_pred)
    HNA = metrics_HNA(y_true, y_pred, unknown_label)
    OSFM = metrics_OSFM(cm)

    print(f"HNA: {HNA:.3f}")
    print(f"weighted OSFM: {OSFM:.3f}")
    print("Confusion Matrix:")
    print(cm)

    return HNA, OSFM


def evaluation_novel(y_true, y_pred, num_known_classes, phase='1st'):
    """
    Evaluate generalized novel category discovery performance.
    Calculate harmonic clustering accuracy.
    
    Args:
        y_true: ground true label n*1
        y_pred: predicted label n*1
        num_known_classes: number of known (seen) classes
        phase: current phase name
    
    Returns:
        Dictionary with AKS, ANS, HCA metrics
    """
    print(f"Performing {phase} generalized novel category discovery evaluation")

    # Accuracy of seen (known) classes
    condition_seen = y_true < num_known_classes
    if condition_seen.sum() > 0:
        acc_seen = accuracy_score(y_true[condition_seen], y_pred[condition_seen])
        print(f"AKS (Accuracy on Known/Seen classes): {acc_seen:.3f}")
    else:
        acc_seen = 0
        print("No known class samples found")
    
    # Clustering accuracy of novel classes
    condition_novel = (y_true >= num_known_classes) & (y_pred >= num_known_classes)
    
    if acc_seen == 0 or np.all(condition_novel == False):
        HCA = 0
        acc_novel = 0
        print("Cannot compute HCA: no novel samples or zero known accuracy")
    else:
        try:
            _, ind, w = cluster_acc(y_true[condition_novel], y_pred[condition_novel], return_ind=True)
            ind_novel = ind[ind[:, -1] >= num_known_classes]
            acc_novel = sum([w[i, j] for i, j in ind_novel]) * 1.0 / y_true[y_true >= num_known_classes].size
            # Harmonic clustering accuracy
            HCA = 2 / (1 / acc_seen + 1 / acc_novel)
            print(f"ANS (Accuracy on Novel/Seen classes): {acc_novel:.3f}")
        except Exception as e:
            print(f"Error computing novel accuracy: {e}")
            acc_novel = 0
            HCA = 0

    print(f"Harmonic Clustering Accuracy (HCA): {HCA:.3f}")
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
    except:
        cm = None
    
    return {
        'AKS': acc_seen,
        'ANS': acc_novel,
        'HCA': HCA,
        'confusion_matrix': cm
    }


def evaluation_gcd_all(y_true, y_pred, num_known_classes):
    """
    Comprehensive evaluation for GCD task.
    
    Args:
        y_true: ground true labels
        y_pred: predicted labels
        num_known_classes: number of known classes
    
    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
    from sklearn.metrics import adjusted_rand_score as ari_score
    
    results = {}
    
    # Overall metrics
    results['overall_acc'] = cluster_acc(y_true.astype(int), y_pred.astype(int))
    results['overall_nmi'] = nmi_score(y_true, y_pred)
    results['overall_ari'] = ari_score(y_true, y_pred)
    
    # Known classes metrics
    mask_known = y_true < num_known_classes
    if mask_known.sum() > 0:
        results['known_acc'] = cluster_acc(y_true[mask_known].astype(int), y_pred[mask_known].astype(int))
        results['known_nmi'] = nmi_score(y_true[mask_known], y_pred[mask_known])
        results['known_ari'] = ari_score(y_true[mask_known], y_pred[mask_known])
    else:
        results['known_acc'] = 0
        results['known_nmi'] = 0
        results['known_ari'] = 0
    
    # Novel classes metrics
    mask_novel = y_true >= num_known_classes
    if mask_novel.sum() > 0:
        results['novel_acc'] = cluster_acc(y_true[mask_novel].astype(int), y_pred[mask_novel].astype(int))
        results['novel_nmi'] = nmi_score(y_true[mask_novel], y_pred[mask_novel])
        results['novel_ari'] = ari_score(y_true[mask_novel], y_pred[mask_novel])
    else:
        results['novel_acc'] = 0
        results['novel_nmi'] = 0
        results['novel_ari'] = 0
    
    # Harmonic mean of known and novel accuracy
    if results['known_acc'] > 0 and results['novel_acc'] > 0:
        results['harmonic_acc'] = 2 / (1/results['known_acc'] + 1/results['novel_acc'])
    else:
        results['harmonic_acc'] = 0
    
    return results
