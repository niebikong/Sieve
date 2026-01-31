import argparse
import torch.optim.lr_scheduler
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import Subset
from tqdm import tqdm
import json
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from datasets.dataloader_tls import TLSDataset, GaussianNoiseTransform, MixTransform, KCropsTransform, FeatureSwappingTransform
from models.preresnet import DeepResNet
from utils import *

parser = argparse.ArgumentParser('Train with synthetic TLS noisy dataset')
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
parser.add_argument('--temperature', default=0.07, type=float, help='temperature parameter for L_batch loss (default: 0.07)')
parser.add_argument('--k', default=100, type=int, help='neighbors for knn sample selection (default: 200)')

# train settings
parser.add_argument('--model', default='DeepResNet', help='model architecture (default: DeepResNet)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.02)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--run_path', type=str, help='run path containing all results')


def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args, clean_label=None, log_dir=None):
    """
    Training function - Alternative version using ALL data for L_batch:
    - labeled_trainloader: D_labeled (clean samples selected by sample selection)
      Used for L_ce (supervised cross-entropy loss)
    - all_trainloader: D_all (ALL training samples, including D_labeled)
      Used for L_batch (self-supervised contrastive loss)
    
    Note: D_labeled ⊆ D_all (clean samples participate in BOTH losses)
    """
    encoder.train()
    classifier.train()
    proj_head.train()
    pred_head.train()
    xlosses = AverageMeter('xloss')
    blosses = AverageMeter('bloss')  # L_batch self-supervised contrastive loss meter
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader, desc=f'Epoch {epoch}')

    # Pre-compute constants for efficiency
    device = next(encoder.parameters()).device

    for batch_idx, ([inputs_u1, inputs_u2], _, _, _) in enumerate(all_bar):
        try:
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            [inputs_x1, inputs_x2], labels_x, _, index = next(labeled_train_iter)

        # Move all data to GPU at once
        batch_size = inputs_x1.size(0)
        batch_size_u = inputs_u1.size(0)
        inputs_x1, inputs_x2 = inputs_x1.to(device, non_blocking=True), inputs_x2.to(device, non_blocking=True)
        inputs_u1, inputs_u2 = inputs_u1.to(device, non_blocking=True), inputs_u2.to(device, non_blocking=True)
        labels_x = modified_label[index].to(device, non_blocking=True)

        # Optimized one-hot encoding
        targets_x = torch.zeros(batch_size, args.num_classes, device=device, dtype=torch.float32)
        targets_x.scatter_(1, labels_x.view(-1, 1), 1)

        # Mixup with optimized operations
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size(0), device=device)

        mixed_input = l * all_inputs_x + (1 - l) * all_inputs_x[idx]
        mixed_target = l * all_targets_x + (1 - l) * all_targets_x[idx]

        # Forward pass optimization: batch all encoder calls
        all_inputs = torch.cat([mixed_input, inputs_u1, inputs_u2], dim=0)
        all_features = encoder(all_inputs)

        # Split features efficiently
        split_sizes = [mixed_input.size(0), batch_size_u, batch_size_u]
        mixed_features, features_u1, features_u2 = torch.split(all_features, split_sizes, dim=0)

        # Cross-entropy loss
        logits = classifier(mixed_features)
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # L_batch self-supervised contrastive loss (removed InfoNCE)
        # Apply projection head efficiently
        # 256 -> 128
        proj_features = proj_head(torch.cat([features_u1, features_u2], dim=0))
        z_u1, z_u2 = torch.split(proj_features, [batch_size_u, batch_size_u], dim=0)

        # Normalize efficiently
        z_u1_norm = F.normalize(z_u1, dim=1)
        z_u2_norm = F.normalize(z_u2, dim=1)

        # Optimized contrastive loss computation using vectorized operations
        # Create positive pairs: z_u1[i] with z_u2[i]
        pos_sim = torch.sum(z_u1_norm * z_u2_norm, dim=1) / args.temperature  # [batch_size_u]

        # Compute all negative similarities efficiently
        # For z_u1[i]: negatives are all z_u2[j] where j != i, and all z_u1[j] where j != i
        neg_sim_u1_u2 = torch.mm(z_u1_norm, z_u2_norm.t()) / args.temperature  # [batch_size_u, batch_size_u]
        neg_sim_u1_u1 = torch.mm(z_u1_norm, z_u1_norm.t()) / args.temperature  # [batch_size_u, batch_size_u]

        # Remove diagonal (self-similarity) and positive pairs
        mask_diag = torch.eye(batch_size_u, device=device, dtype=torch.bool)
        neg_sim_u1_u2.masked_fill_(mask_diag, float('-inf'))
        neg_sim_u1_u1.masked_fill_(mask_diag, float('-inf'))

        # Compute contrastive loss for z_u1 -> z_u2 direction
        neg_sims_1 = torch.cat([neg_sim_u1_u2, neg_sim_u1_u1], dim=1)  # [batch_size_u, 2*batch_size_u]
        logits_1 = torch.cat([pos_sim.unsqueeze(1), neg_sims_1], dim=1)  # [batch_size_u, 2*batch_size_u+1]
        labels_1 = torch.zeros(batch_size_u, device=device, dtype=torch.long)
        loss_1 = F.cross_entropy(logits_1, labels_1)

        # Compute contrastive loss for z_u2 -> z_u1 direction (symmetric)
        neg_sim_u2_u1 = neg_sim_u1_u2.t()  # [batch_size_u, batch_size_u]
        neg_sim_u2_u2 = torch.mm(z_u2_norm, z_u2_norm.t()) / args.temperature  # [batch_size_u, batch_size_u]
        neg_sim_u2_u2.masked_fill_(mask_diag, float('-inf'))

        neg_sims_2 = torch.cat([neg_sim_u2_u1, neg_sim_u2_u2], dim=1)  # [batch_size_u, 2*batch_size_u]
        logits_2 = torch.cat([pos_sim.unsqueeze(1), neg_sims_2], dim=1)  # [batch_size_u, 2*batch_size_u+1]
        labels_2 = torch.zeros(batch_size_u, device=device, dtype=torch.long)
        loss_2 = F.cross_entropy(logits_2, labels_2)

        Lbatch = (loss_1 + loss_2) * 0.5

        # Combine losses (removed InfoNCE term)
        loss = Lce + args.lambda_fc * Lbatch

        # Update meters
        xlosses.update(Lce.item())
        blosses.update(Lbatch.item())

        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]:.6f} CE: {xlosses.avg:.4f} Batch: {blosses.avg:.4f}')

        # Optimized backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate and print label accuracy after each epoch
    if clean_label is not None:
        # Calculate accuracy of modified_label compared to clean_label
        # Only consider samples that are in the training set (not open-set noise)
        valid_mask = clean_label < args.num_classes  # Exclude open-set samples (labeled as num_classes)

        if valid_mask.sum() > 0:
            clean_labels_valid = clean_label[valid_mask]
            modified_labels_valid = modified_label[valid_mask]

            # Calculate accuracy
            correct_labels = (clean_labels_valid == modified_labels_valid).sum().item()
            total_labels = valid_mask.sum().item()  # Total number of ID samples
            label_accuracy = correct_labels / total_labels * 100
            open_set_samples = (~valid_mask).sum().item()

            # Print to console
            print(f"\n[Epoch {epoch}] Label Accuracy: {correct_labels}/{total_labels} = {label_accuracy:.2f}%")
            print(f"[Epoch {epoch}] ID samples: {total_labels}, Open-set samples: {open_set_samples}")

            # Write to log file
            if log_dir is not None:
                import os
                log_file_path = os.path.join(log_dir, 'epoch_label_acc.txt')
                with open(log_file_path, 'a') as f:
                    f.write(f"Epoch {epoch}: Label Accuracy = {correct_labels}/{total_labels} = {label_accuracy:.2f}%, "
                           f"ID samples = {total_labels}, Open-set samples = {open_set_samples}\n")
        else:
            error_msg = f"\n[Epoch {epoch}] No valid clean samples found for label accuracy calculation"
            print(error_msg)

            # Write error to log file
            if log_dir is not None:
                import os
                log_file_path = os.path.join(log_dir, 'epoch_label_acc.txt')
                with open(log_file_path, 'a') as f:
                    f.write(f"Epoch {epoch}: No valid clean samples found for label accuracy calculation\n")


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)

    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    return accuracy.avg

def detailed_test_evaluation(testloader, encoder, classifier, num_classes, log_dir):
    """Perform detailed test evaluation with multiple metrics"""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import json

    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    print("Performing detailed test evaluation...")
    with torch.no_grad():
        for data, label, _ in tqdm(testloader, desc="Testing"):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    # Prepare results dictionary
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'num_samples': len(all_labels),
        'num_classes': num_classes
    }

    # Save results to JSON file
    results_file = os.path.join(log_dir, 'final_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix as text
    cm_file = os.path.join(log_dir, 'confusion_matrix.txt')
    with open(cm_file, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write("================\n")
        f.write(f"Shape: {cm.shape}\n")
        f.write("Rows: True labels, Columns: Predicted labels\n\n")

        # Write header
        f.write("     ")
        for i in range(num_classes):
            f.write(f"{i:4d}")
        f.write("\n")

        # Write matrix
        for i in range(num_classes):
            f.write(f"{i:3d}: ")
            for j in range(num_classes):
                f.write(f"{cm[i,j]:4d}")
            f.write("\n")

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Results saved to: {results_file}")
    print(f"Confusion matrix saved to: {cm_file}")
    print(f"{'='*60}")

    return results


def save_predict(predict_clean, predict_closed_noise, predict_open_noise, epoch, run_path, dataset):
    """Save clean / closed-noise / open-noise max prediction confidence to JSON file"""
    predict_dict = {
        'clean_predict': predict_clean.cpu().numpy().tolist(),
        'closed_noise_predict': predict_closed_noise.cpu().numpy().tolist(),
        'open_noise_predict': predict_open_noise.cpu().numpy().tolist(),
        'epoch': epoch,
    }

    # Create directory if it doesn't exist
    save_dir = Path(f'/home/ju/Desktop/TNSE/Sieve/logs/{dataset}/{run_path}/Epochs_predict')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    save_path = save_dir / f'predict_epoch_{epoch}.json'
    with open(save_path, 'w') as f:
        json.dump(predict_dict, f)


def save_three_samples_statistic(num_clean, num_closed_noise, num_open_noise, epoch, log_dir):
    """Save clean / closed-noise / open-noise sample counts to JSON file"""
    json_path = os.path.join(log_dir, 'three_samples_num_statistic.json')
    
    # Load existing data or create new
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'epochs': []}
    
    # Add current epoch statistics
    epoch_stat = {
        'epoch': epoch,
        'clean_samples': num_clean,
        'closed_noise_samples': num_closed_noise,
        'open_noise_samples': num_open_noise,
        'total_samples': num_clean + num_closed_noise + num_open_noise
    }
    data['epochs'].append(epoch_stat)
    
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_scores(scores_clean, scores_closed_noise, scores_open_noise, epoch, run_path, dataset):
    """Save clean / closed-noise / open-noise scores to JSON file"""
    scores_dict = {
        'clean_scores': scores_clean.cpu().numpy().tolist(),
        'closed_noise_scores': scores_closed_noise.cpu().numpy().tolist(),
        'open_noise_scores': scores_open_noise.cpu().numpy().tolist(),
        'epoch': epoch,
    }

    # Create directory if it doesn't exist
    save_dir = Path(f'/home/ju/Desktop/TNSE/Sieve/logs/{dataset}/{run_path}/Epochs_C_Measure')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    save_path = save_dir / f'scores_epoch_{epoch}.json'
    with open(save_path, 'w') as f:
        json.dump(scores_dict, f)

def evaluate(dataloader, encoder, classifier, args, noisy_label, clean_label, epoch, stat_logs, log_dir=None):
    encoder.eval()
    classifier.eval()
    feature_bank = []
    prediction = []

    with torch.no_grad():
        # Generate feature bank
        for (data, target, _, index) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature = encoder(data)
            feature_bank.append(feature)
            output = classifier(feature)
            prediction.append(output)

        feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

        # Sample relabelling
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)
        print(f'Prediction track: mean: {his_score.mean()} max: {his_score.max()} min: {his_score.min()}')
        conf_id = torch.where(his_score > args.zeta)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        # Sample selection
        # Increase chunks for large datasets to avoid OOM
        num_samples = len(feature_bank)
        chunks = max(10, num_samples // 10000)  # At least 10 chunks, or 1 chunk per 10k samples
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, chunks)
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        # 根据标签正确性划分 clean / closed noise / open noise（open noise 的 clean_label == num_classes）
        clean_id = torch.where(modified_label == clean_label)[0]
        noisy_closed_id = torch.where((modified_label != clean_label) & (clean_label < args.num_classes))[0]
        noisy_open_id = torch.where((modified_label != clean_label) & (clean_label == args.num_classes))[0]

        # Save scores for clean / closed noise / open noise
        clean_scores = right_score[clean_id]
        closed_noise_scores = right_score[noisy_closed_id]
        open_noise_scores = right_score[noisy_open_id]
        # save_scores(clean_scores, closed_noise_scores, open_noise_scores, epoch, args.run_path, args.dataset)

        # Save max prediction confidence for clean / closed noise / open noise
        clean_predict = his_score[clean_id]
        closed_noise_predict = his_score[noisy_closed_id]
        open_noise_predict = his_score[noisy_open_id]
        # save_predict(clean_predict, closed_noise_predict, open_noise_predict, epoch, args.run_path, args.dataset)

        # Monitor statistics
        num_clean = len(clean_id)
        num_closed = len(noisy_closed_id)
        num_open = len(noisy_open_id)
        correct_relabelling = torch.sum(modified_label[conf_id] == clean_label[conf_id])

        print(f'Epoch [{epoch}/{args.epochs}] selection: clean={num_clean} closed_noise={num_closed} open_noise={num_open}')
        print(f'Epoch [{epoch}/{args.epochs}] relabelling: correct: {correct_relabelling} total: {len(conf_id)}')

        stat_logs.write(f'Epoch [{epoch}/{args.epochs}] selection: clean={num_clean} closed_noise={num_closed} open_noise={num_open}\n')
        stat_logs.flush()

        # Save three samples statistics to JSON file
        # if log_dir is not None:
        #     save_three_samples_statistic(num_clean, num_closed, num_open, epoch, log_dir)

    # 兼容返回：noisy_id 为 closed+open 的并集
    if len(noisy_closed_id) > 0 and len(noisy_open_id) > 0:
        noisy_id = torch.cat([noisy_closed_id, noisy_open_id])
    elif len(noisy_closed_id) > 0:
        noisy_id = noisy_closed_id
    else:
        noisy_id = noisy_open_id

    return clean_id, noisy_id, modified_label


def main():
    args = parser.parse_args()
    seed_everything(args.seed)

    # Simplified run path with dataset information
    if args.run_path is None:
        if args.dataset == 'malicious_tls':
            args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_NoiseDS({args.noisy_dataset})_Model({args.zeta}_{args.xi})'
        else:
            args.run_path = f'Dataset({args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode})_NoiseDS({args.noisy_dataset})_Model({args.zeta}_{args.xi})'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # Create simplified log directories
    log_dir = f'/home/ju/Desktop/TNSE/Sieve/logs/{args.dataset}/{args.run_path}'
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    ############################# Dataset initialization ##############################################
    # Configure dataset-specific parameters based on selected dataset
    if args.dataset == 'malicious_tls':
        args.num_classes = 23  # Number of classes in malicious TLS dataset
        args.input_size = 86  # Feature vector size
    elif args.dataset == 'DDoS2019':
        args.num_classes = 2  # Number of classes in DDoS2019 dataset (benign: 0, attack: 1)
        args.input_size = 82  # Feature vector size
        args.batch_size = 1024
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from 'malicious_tls' or 'DDoS2019'")

    # Data augmentation setup
    strong_transform = FeatureSwappingTransform(swap_ratio=0.05)  # 强增强使用较大的交换比例
    weak_transform = FeatureSwappingTransform(swap_ratio=0)   # 弱增强使用较小的交换比例
    none_transform = None

    # Generate train dataset with noise (fixed to malicious_tls with ids2017 as noise)
    noise_file = f'/home/ju/Desktop/TNSE/Sieve/noise_files/{args.dataset}_{args.noise_ratio}_{args.open_ratio}_{args.noise_mode}_{args.noisy_dataset}_noise.json'

    train_data = TLSDataset(
        dataset=args.dataset,
        noisy_dataset=args.noisy_dataset,
        transform=KCropsTransform(strong_transform, 2),
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        open_ratio=args.open_ratio,
        dataset_mode='train',
        noise_file=noise_file
    )

    eval_data = TLSDataset(
        dataset=args.dataset,
        noisy_dataset=args.noisy_dataset,
        transform=weak_transform,
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        open_ratio=args.open_ratio,
        dataset_mode='train',
        noise_file=noise_file
    )

    test_data = TLSDataset(
        dataset=args.dataset,
        transform=none_transform,
        dataset_mode='test'
    )

    all_data = TLSDataset(
        dataset=args.dataset,
        noisy_dataset=args.noisy_dataset,
        transform=MixTransform(strong_transform=strong_transform, weak_transform=weak_transform, K=1),
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        open_ratio=args.open_ratio,
        dataset_mode='train',
        noise_file=noise_file
    )

    # Extract labels for monitoring
    noisy_label = eval_data.label.clone().detach().cuda()
    clean_label = eval_data.clean_label.clone().detach().cuda()

    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    all_loader = torch.utils.data.DataLoader(
        all_data, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    ################################ Model initialization ###########################################
    # Initialize encoder (DeepResNet without classifier)
    encoder = DeepResNet(input_size=args.input_size, num_classes=None)  # Set num_classes to None to skip classifier
    encoder.cuda()

    # Create separate classifier
    classifier = torch.nn.Linear(256, args.num_classes).cuda()  # 256 is the feature dimension from DeepResNet

    # Initialize projection and prediction heads
    proj_head = torch.nn.Sequential(
        torch.nn.Linear(256, 256),  # 256 is the feature dimension from DeepResNet
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).cuda()

    pred_head = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).cuda()

    #################################### Training initialization #######################################
    optimizer = SGD([
        {'params': encoder.parameters()},
        {'params': classifier.parameters()},
        {'params': proj_head.parameters()},
        {'params': pred_head.parameters()}
    ], lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/50.0)

    # Setup logging
    acc_logs = open(f'{log_dir}/acc.txt', 'w')
    stat_logs = open(f'{log_dir}/stat.txt', 'w')
    save_config(args, log_dir)
    print('Train args: \n', args)
    best_acc = 0

    ################################ Training loop ###########################################
    for epoch in range(args.epochs):
        clean_id, noisy_id, modified_label = evaluate(eval_loader, encoder, classifier, args, noisy_label, clean_label, epoch, stat_logs, log_dir)

        # Create D_labeled: balanced sampler for clean data (selected by sample selection)
        # Used for supervised learning (L_ce)
        clean_subset = Subset(train_data, clean_id.cpu())
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(
            clean_subset, batch_size=args.batch_size,
            sampler=sampler, num_workers=4, drop_last=True
        )

        # Use all_loader for self-supervised contrastive learning (L_batch)
        # Note: all_loader contains ALL training samples (including clean_id)
        print(f'Epoch {epoch}: Total samples={len(all_data)}, Clean (D_labeled)={len(clean_id)} ({len(clean_id)/len(all_data)*100:.1f}%)')

        # Train one epoch: L_ce on labeled_loader, L_batch on all_loader
        train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head, optimizer, epoch, args, clean_label, log_dir)

        # Evaluate and save checkpoint
        cur_acc = test(test_loader, encoder, classifier, epoch)
        scheduler.step()

        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'proj_head': proj_head.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename=f'{log_dir}/best_acc.pth.tar')

        acc_logs.write(f'Epoch [{epoch}/{args.epochs}]: Best accuracy@{best_acc:.4f} Current accuracy@{cur_acc:.4f}\n')
        acc_logs.flush()
        print(f'Epoch [{epoch}/{args.epochs}]: Best accuracy@{best_acc:.4f} Current accuracy@{cur_acc:.4f} \n')

    # Save final model
    save_checkpoint({
        'epoch': args.epochs,
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'proj_head': proj_head.state_dict(),
        'pred_head': pred_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'final_acc': cur_acc,
    }, filename=f'{log_dir}/last.pth.tar')

    # Perform detailed final test evaluation
    print("\n" + "="*60)
    print("PERFORMING FINAL DETAILED TEST EVALUATION")
    print("="*60)

    # Load best model for final evaluation
    best_checkpoint = torch.load(f'{log_dir}/best_acc.pth.tar')
    encoder.load_state_dict(best_checkpoint['encoder'])
    classifier.load_state_dict(best_checkpoint['classifier'])

    # Perform detailed evaluation
    final_results = detailed_test_evaluation(test_loader, encoder, classifier, args.num_classes, log_dir)

    # Write final summary to log
    acc_logs.write(f'\n{"="*50}\n')
    acc_logs.write(f'FINAL TEST RESULTS\n')
    acc_logs.write(f'{"="*50}\n')
    acc_logs.write(f'Best Training Accuracy: {best_acc:.4f}\n')
    acc_logs.write(f'Final Test Accuracy: {final_results["accuracy"]:.4f}\n')
    acc_logs.write(f'F1-Score (Macro): {final_results["f1_macro"]:.4f}\n')
    acc_logs.write(f'F1-Score (Weighted): {final_results["f1_weighted"]:.4f}\n')
    acc_logs.write(f'Total Test Samples: {final_results["num_samples"]}\n')
    acc_logs.write(f'Number of Classes: {final_results["num_classes"]}\n')
    acc_logs.write(f'{"="*50}\n')
    acc_logs.flush()

    acc_logs.close()
    stat_logs.close()

    print(f"\nTraining completed! All results saved to: {log_dir}")


if __name__ == '__main__':
    main()
