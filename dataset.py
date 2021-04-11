import os
import random

from torch.utils.data import Dataset, DataLoader, BatchSampler
import numpy as np
import torch

class EmbeddingTrainDataset(Dataset):
    def __init__(self, opts):
        path = opts['train_path']
        self.dataset = []
        self.count = 0
        self.labels = []
        spk_idx = 0
        for speaker in os.listdir(path):
            speaker_path = os.path.join(path, speaker)
            embeddings = os.listdir(speaker_path)
            if len(embeddings) > 30:
                for embedding in embeddings:
                    self.dataset.append(os.path.join(speaker_path, embedding))
                    self.labels.append(spk_idx)
                spk_idx += 1

    def __len__(self):
        return len(self.dataset) * 5 # form more trials
    
    def __getitem__(self, idx):
        embedding_path = self.dataset[idx]
        sid = self.labels[idx]
        embedding = np.load(embedding_path)
        return embedding, sid

class EmbeddingEnrollDataset(Dataset):
    def __init__(self, opts):
        path = opts['enroll_path']
        self.dataset = []
        self.count = 0
        self.labels = []
        spk_idx = 0
        for speaker in os.listdir(path):
            speaker_path = os.path.join(path, speaker)
            self.dataset.append(speaker_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        speaker_path = self.dataset[idx]
        embeddings = os.listdir(speaker_path)
        all_embeddings = []
        for embedding in embeddings:
            embedding_path = os.path.join(speaker_path, embedding)
            embedding = np.load(embedding_path)
            all_embeddings.append(embedding.reshape(-1))
        embedding = torch.from_numpy(np.array(all_embeddings))
        return embedding.unsqueeze(0), os.path.basename(speaker_path)

    def __call__(self):
        idx = 0
        while idx < len(self.dataset):
            embedding, spk = self.__getitem__(idx)
            yield embedding, spk
            idx += 1

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, all_speech, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = all_speech
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
