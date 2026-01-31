import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import random
import os
import pickle
from pathlib import Path
import time


class GaussianNoiseTransform:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class FeatureSwappingTransform:
    def __init__(self, swap_ratio=0.1):
        """特征交换转换
        
        Args:
            swap_ratio (float): 要交换的特征对的比例，strong transform用较大的值，weak transform用较小的值
        """
        self.swap_ratio = swap_ratio
        
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = x.clone()
        n_features = len(x)
        n_swaps = int(n_features * self.swap_ratio)
        
        if n_swaps > 0:
            swap_indices = torch.randperm(n_features-1)[:n_swaps]
            for idx in swap_indices:
                x[idx], x[idx+1] = x[idx+1].clone(), x[idx].clone()
        
        return x

    def __repr__(self):
        return f"FeatureSwappingTransform(swap_ratio={self.swap_ratio})"


class MixTransform:
    """Optimized mixed transform for efficient data augmentation"""
    def __init__(self, strong_transform, weak_transform, K=2):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K = K

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        if self.K == 2:
            return [self.weak_transform(x.clone()), self.strong_transform(x.clone())]
        else:
            return [self.strong_transform(x.clone()) for _ in range(self.K)]


class KCropsTransform:
    """Optimized K-crops transform with memory efficiency"""
    def __init__(self, transform, K=2):
        self.transform = transform
        self.K = K

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        if self.K == 1:
            return [self.transform(x)]
        elif self.K == 2:
            return [self.transform(x.clone()), self.transform(x.clone())]
        else:
            return [self.transform(x.clone()) for _ in range(self.K)]


class TLSDataset(Dataset):
    def __init__(self, dataset='malicious_tls', noisy_dataset='TLS1.3', transform=None, noise_mode='sym',
                 dataset_mode='train', noise_ratio=0.5, open_ratio=0.5, noise_file=None,
                 selected_ood_classes=[1, 2, 3]):
        self.r = noise_ratio
        self.on = open_ratio
        self.transform = transform
        self.mode = dataset_mode
        self.dataset = dataset
        self.noisy_dataset = noisy_dataset
        self.selected_ood_classes = selected_ood_classes

        # Configure dataset-specific parameters
        if dataset == 'malicious_tls':
            self.num_classes = 23
            self.feature_dim = 86
        elif dataset == 'DDoS2019':
            self.num_classes = 2
            self.feature_dim = 82
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Choose from 'malicious_tls' or 'DDoS2019'")

        # 非对称噪声转换矩阵
        if dataset == 'malicious_tls':
            self.transition = {2: 0, 4: 7, 5: 6, 9: 1, 3: 8}
        elif dataset == 'DDoS2019':
            self.transition = {1: 0}
        else:
            self.transition = {}

        print(f"Initializing TLSDataset - Dataset: {dataset}, Mode: {dataset_mode}, Noise: {noise_mode}, "
              f"Noise ratio: {noise_ratio}, Open ratio: {open_ratio}")
        print(f"Selected OOD classes: {selected_ood_classes}")

        # Load the main dataset
        if dataset == 'malicious_tls':
            X_train_main, y_train_main, X_test_main, y_test_main = load_malicious_TLS()
        elif dataset == 'DDoS2019':
            X_train_main, y_train_main, X_test_main, y_test_main = load_DDoS2019()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Load the noise dataset based on noisy_dataset parameter
        X_train_ids, y_train_ids, X_test_ids, y_test_ids = None, None, None, None
        
        if dataset == 'malicious_tls':
            if noisy_dataset == 'TLS1.3':
                X_train_ids, y_train_ids, X_test_ids, y_test_ids = load_TLS13(selected_classes=selected_ood_classes)
            elif noisy_dataset == 'IDS2018':
                X_train_ids, y_train_ids, X_test_ids, y_test_ids = load_IDS2018(selected_classes=[0], target_dim=self.feature_dim)
            else:
                raise ValueError(f"Unsupported noisy_dataset for malicious_tls: {noisy_dataset}. Choose from 'TLS1.3' or 'IDS2018'")
        elif dataset == 'DDoS2019':
            if noisy_dataset == 'IDS2018':
                X_train_ids, y_train_ids, X_test_ids, y_test_ids = load_IDS2018(selected_classes=[0], target_dim=self.feature_dim)
            elif noisy_dataset == 'TLS1.3':
                X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_TLS13(selected_classes=selected_ood_classes)
                # Truncate TLS1.3 features from 86 to 82 dimensions
                X_train_ids = X_train_raw[:, :self.feature_dim]
                X_test_ids = X_test_raw[:, :self.feature_dim]
                y_train_ids, y_test_ids = y_train_raw, y_test_raw
            else:
                raise ValueError(f"Unsupported noisy_dataset for DDoS2019: {noisy_dataset}. Choose from 'TLS1.3' or 'IDS2018'")

        # Verify main dataset dimensions
        assert X_train_main.shape[1] == self.feature_dim, f"Main data dimension mismatch. Expected {self.feature_dim}, got {X_train_main.shape[1]}"

        if self.mode == 'test':
            self.data = torch.FloatTensor(X_test_main)
            self.label = torch.LongTensor(y_test_main)
            self.clean_label = self.label.clone()
            print(f"Test dataset loaded: {len(self.data)} samples")

        elif self.mode == 'train':
            self.data = torch.FloatTensor(X_train_main)
            self.clean_label = torch.LongTensor(y_train_main)
            self.label = self.clean_label.clone()

            if X_train_ids is not None:
                noise_data = torch.FloatTensor(X_train_ids)
                print(f"Training dataset: {len(self.data)} main samples, {len(noise_data)} OOD samples")
            else:
                noise_data = None
                print(f"Training dataset: {len(self.data)} samples (no noise data available)")

            if noise_data is not None:
                if os.path.exists(noise_file):
                    print(f"Loading noise configuration from: {noise_file}")
                    with open(noise_file, "r") as f:
                        noise = json.load(f)
                    noise_labels = noise['noise_labels']
                    self.open_noise = noise['open_noise']
                    self.closed_noise = noise['closed_noise']

                    print("Applying cached noise configuration...")
                    for cleanIdx, noisyIdx in self.open_noise:
                        if noisyIdx < len(noise_data):
                            self.data[cleanIdx] = noise_data[noisyIdx]
                            self.clean_label[cleanIdx] = self.num_classes

                    self.label = torch.LongTensor(noise_labels)
                    print(f"Applied {len(self.open_noise)} open-set noise samples and {len(self.closed_noise)} closed-set noise samples")

                else:
                    print("Generating new noise configuration...")
                    random.seed(42)
                    np.random.seed(42)

                    noise_labels = []
                    idx = list(range(len(self.data)))
                    random.shuffle(idx)

                    num_total_noise = int(self.r * len(self.data))
                    num_open_noise = int(self.on * num_total_noise)
                    num_closed_noise = num_total_noise - num_open_noise

                    print(f'Noise statistics:')
                    print(f'  Total samples: {len(self.data)}')
                    print(f'  Clean samples: {len(self.data) - num_total_noise}')
                    print(f'  Closed-set noise: {num_closed_noise}')
                    print(f'  Open-set noise: {num_open_noise}')
                    print(f'  Selected OOD classes: {self.selected_ood_classes}')

                    if num_open_noise > len(noise_data):
                        print(f"Warning: Requested {num_open_noise} open-set samples but only {len(noise_data)} available")
                        num_open_noise = len(noise_data)
                        num_closed_noise = num_total_noise - num_open_noise

                    target_noise_idx = list(range(len(noise_data)))
                    random.shuffle(target_noise_idx)

                    self.open_noise = list(zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))
                    self.closed_noise = idx[num_open_noise:num_total_noise]

                    noise_labels = [int(label) for label in self.clean_label]

                    for i in self.closed_noise:
                        if noise_mode == 'sym':
                            noiselabel = random.randint(0, self.num_classes - 1)
                        else:
                            current_label = int(self.clean_label[i])
                            noiselabel = self.transition.get(current_label, current_label)
                        noise_labels[i] = noiselabel

                    print("Applying open-set noise...")
                    for cleanIdx, noisyIdx in self.open_noise:
                        self.data[cleanIdx] = noise_data[noisyIdx]
                        self.clean_label[cleanIdx] = self.num_classes

                    noise = {
                        'noise_labels': noise_labels,
                        'open_noise': self.open_noise,
                        'closed_noise': self.closed_noise,
                        'selected_ood_classes': self.selected_ood_classes,
                        'noise_mode': noise_mode,
                        'noise_ratio': self.r,
                        'open_ratio': self.on
                    }

                    os.makedirs(os.path.dirname(noise_file), exist_ok=True)
                    print(f"Saving noise configuration to: {noise_file}")
                    with open(noise_file, "w") as f:
                        json.dump(noise, f, indent=2)

                    self.label = torch.LongTensor(noise_labels)
            else:
                print("No noise data available, no noise injection applied")
                self.open_noise = []
                self.closed_noise = []

        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.mode}!')

    def __getitem__(self, index):
        data = self.data[index].clone()

        if self.transform is not None:
            if hasattr(self.transform, '__call__'):
                data = self.transform(data)

        if self.mode == 'train':
            target = self.label[index]
            clean_target = self.clean_label[index]
            return data, target, clean_target, index
        else:
            target = self.label[index]
            return data, target, index

    def __len__(self):
        return len(self.data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)

    def update_labels(self, new_label):
        self.label = new_label.cpu()


def load_malicious_TLS():
    """Load and preprocess malicious TLS dataset with caching"""
    cache_file = Path('/home/ju/Desktop/TNSE/Sieve/cache/malicious_tls.pkl')
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        print(f"Loading malicious TLS data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Processing malicious TLS data...")
    start_time = time.time()

    output_path = r"/home/ju/Desktop/TNSE/Sieve/datasets/label_encodered_malicious_TLS-1_processed.csv"
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
            data[col] = label_encoder.fit_transform(data[col])

    data = data.astype(float)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    result = (X_train, y_train, X_test, y_test)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"Malicious TLS data processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    return result


def load_TLS13(selected_classes=None):
    """Load and preprocess TLS1.3 dataset with caching
    
    Args:
        selected_classes (list|None): Optional list of class indices to keep as open-set noise.
            If None, all classes are used.
    
    Dataset info:
    - Path: /home/ju/Desktop/TNSE/Sieve/datasets/TLS1.3_like_TLS1.2_processed.csv
    - Classes: 41
    - Feature dimension: 86 (same as malicious_tls)
    """
    normalized_classes = None
    if selected_classes is not None:
        normalized_classes = sorted(set(map(int, selected_classes)))

    cache_key = "TLS1.3_all" if normalized_classes is None else f"TLS1.3_classes_{'_'.join(map(str, normalized_classes))}"
    cache_file = Path(f"/home/ju/Desktop/TNSE/Sieve/cache/{cache_key}.pkl")
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        print(f"Loading TLS1.3 data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Processing TLS1.3 data{' for classes ' + str(normalized_classes) if normalized_classes else ''}...")
    start_time = time.time()

    output_path = r"/home/ju/Desktop/TNSE/Sieve/datasets/TLS1.3_like_TLS1.2_processed.csv"
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
            data[col] = label_encoder.fit_transform(data[col])

    data = data.astype(float)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    result = (X_train, y_train, X_test, y_test)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"TLS1.3 data processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Feature dimension: {X_train.shape[1]}")

    return result


def load_DDoS2019():
    """Load and preprocess DDoS2019 dataset with caching
    
    Dataset info:
    - Path: /home/ju/Desktop/TNSE/Sieve/datasets/DDoS2019.csv
    - Classes: 2 (benign: 0, attack: 1)
    - Feature dimension: 82
    """
    cache_file = Path('/home/ju/Desktop/TNSE/Sieve/cache/DDoS2019.pkl')
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        print(f"Loading DDoS2019 data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Processing DDoS2019 data...")
    start_time = time.time()

    output_path = r"/home/ju/Desktop/TNSE/Sieve/datasets/DDoS2019.csv"
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
            data[col] = label_encoder.fit_transform(data[col])

    data = data.astype(float)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.astype(int)

    max_samples = 100000
    if len(X) > max_samples:
        print(f"Sampling {max_samples} samples from {len(X)} total samples...")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    result = (X_train, y_train, X_test, y_test)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"DDoS2019 data processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Feature dimension: {X_train.shape[1]}")

    return result


def load_IDS2018(selected_classes=[0, 1], target_dim=82):
    """Load and preprocess IDS2018 dataset with caching
    
    Args:
        selected_classes (list): List of class indices to use
        target_dim (int): Target feature dimension to pad/truncate to
    
    Dataset info:
    - Path: /home/ju/Desktop/NetMamba/PNP/SMP/dataset/processed_friday_dataset.csv
    - Classes: [0, 1] (2 classes)
    - Feature dimension: 72
    """
    cache_key = f"ids2018_classes_{'_'.join(map(str, sorted(selected_classes)))}_dim_{target_dim}"
    cache_dir = Path('/home/ju/Desktop/TNSE/Sieve/cache')
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"

    if cache_file.exists():
        print(f"Loading IDS2018 data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Processing IDS2018 data for classes {selected_classes}...")
    start_time = time.time()

    output_path = r'/home/ju/Desktop/NetMamba/PNP/SMP/dataset/processed_friday_dataset.csv'

    try:
        df = pd.read_csv(output_path)
        print(f"Loaded IDS2018 dataset with shape: {df.shape}")

        X_all = df.iloc[:, :-1].values
        y_all = df.iloc[:, -1].values

        mask = np.isin(y_all, selected_classes)
        X_filtered = X_all[mask]
        y_filtered = y_all[mask]

        max_samples = min(100000, len(X_filtered))
        if len(X_filtered) > max_samples:
            indices = np.random.choice(len(X_filtered), max_samples, replace=False)
            X_sampled = X_filtered[indices]
            y_sampled = y_filtered[indices]
        else:
            X_sampled = X_filtered
            y_sampled = y_filtered

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_sampled)

        current_dim = X_scaled.shape[1]
        if current_dim < target_dim:
            X_padded = np.zeros((X_scaled.shape[0], target_dim))
            X_padded[:, :current_dim] = X_scaled
            X_scaled = X_padded
        elif current_dim > target_dim:
            X_scaled = X_scaled[:, :target_dim]

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sampled, test_size=0.3, random_state=42)

        result = (X_train, y_train, X_test, y_test)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        print(f"IDS2018 data processing completed in {time.time() - start_time:.2f} seconds")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        return result

    except Exception as e:
        print(f"Error loading IDS2018 dataset: {e}")
        dummy_X = np.random.rand(1000, target_dim)
        dummy_y = np.zeros(1000)
        X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=0.3, random_state=42)
        return (X_train, y_train, X_test, y_test)
