import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def load_sleep_data(data_path='preprocessed'):
    data = []
    labels = []
    for file in os.listdir(data_path):
        if not file.endswith('.npz'):
            continue
        path = os.path.join(data_path, file)
        loaded = np.load(path)
        data.append(loaded['x'])
        labels.append(loaded['y'])
    return data, labels


class BatchedSleepDataset(Dataset):
    def __init__(self, data, labels, minibatch_size=20):
        self.data = data
        self.labels = labels
        self.minibatch_size = minibatch_size
        self.reshuffle()

    def reshuffle(self):
        self.shifts = []
        self.cur_lens = []
        self.total_len = 0
        for subj_data in self.data:
            max_skip = 5 * self.minibatch_size + subj_data.shape[0] % self.minibatch_size
            cur_skip = random.randint(0, max_skip)
            self.shifts.append(cur_skip)
            cur_len = (subj_data.shape[0] - cur_skip) // self.minibatch_size
            self.cur_lens.append(cur_len)
            self.total_len += cur_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.data) - 1
        for idx, subj_data in enumerate(self.data):
            cur_len = self.cur_lens[idx]
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        start_idx = self.shifts[subj_idx] + index * self.minibatch_size
        item_data = self.data[subj_idx][start_idx:start_idx + self.minibatch_size]
        item_labels = self.labels[subj_idx][start_idx:start_idx + self.minibatch_size]
        return torch.tensor(item_data, dtype=torch.float).unsqueeze(1), \
               torch.tensor(item_labels, dtype=torch.long).unsqueeze(1)


class SleepDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float).unsqueeze(1), \
               torch.tensor(self.labels[index], dtype=torch.long).unsqueeze(1)


def load_split_sleep_dataset(data_path='preprocessed', minibatch_size=20):
    data, labels = load_sleep_data(data_path)
    train_size = round(len(data) - 8)
    data_train, labels_train = data[:train_size], labels[:train_size]
    data_test, labels_test = data[train_size:], labels[train_size:]
    return BatchedSleepDataset(data_train, labels_train, minibatch_size), SleepDataset(data_test, labels_test)
