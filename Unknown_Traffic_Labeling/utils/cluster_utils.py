from __future__ import division, print_function
import numpy as np
import sklearn.metrics
import torch
import random
import os
import argparse
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
import time


def evaluate_clustering(y_true, y_pred):
    """Evaluate clustering performance with multiple metrics"""
    start = time.time()
    print('Computing metrics...')
    
    if len(set(y_pred)) < 1000:
        acc = cluster_acc(y_true.astype(int), y_pred.astype(int))
    else:
        acc = None

    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    print(f'Finished computing metrics {time.time() - start:.2f}s')

    return acc, nmi, ari, pur


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy using Hungarian algorithm.
    
    Args:
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        return_ind: whether to return the assignment indices
    
    Returns:
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    """Compute purity score for clustering"""
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def mixed_eval(targets, preds, mask):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask.
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    
    Args:
        targets: All ground truth labels
        preds: All predictions
        mask: Mask defining two subsets
    
    Returns:
        Metrics for labelled and unlabelled subsets
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int), preds.astype(int)), \
                                                         nmi_score(targets, preds), \
                                                         ari_score(targets, preds)

        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'
              .format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

        return (unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()

    else:
        labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask],
                                                               preds.astype(int)[mask]), \
                                                   nmi_score(targets[mask], preds[mask]), \
                                                   ari_score(targets[mask], preds[mask])

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                     preds.astype(int)[~mask]), \
                                                         nmi_score(targets[~mask], preds[~mask]), \
                                                         ari_score(targets[~mask], preds[~mask])

        return (labelled_acc, labelled_nmi, labelled_ari), (
            unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    """Convert string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
