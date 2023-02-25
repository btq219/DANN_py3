import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


def split_dataset(dataset, train_size=0.8):
    """
    :param dataset:
    :param train_size: 0 to 1
    :param random_seed:
    :return:
    """
    # random.seed(random_seed)
    train_len = int(train_size * len(dataset))
    test_len = len(dataset) - train_len

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    return train_dataset, test_dataset

class UwbDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return torch.tensor(sample, dtype=torch.float), torch.tensor(label)



def get_dataset(data_files, train_size=0.8):
    data = []
    labels = []

    for i, file in enumerate(data_files) :
        # Load data and labels from .mat file
        label = int(file[-5])
        mat_data = loadmat(file)['Scenario' + str(label)]
        n, _ = mat_data.shape
        mat_labels = np.ones(n) * (label-1)

        data.append(mat_data)
        labels.append(mat_labels)

    # Concatenate data and labels from each file into a single array
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = np.expand_dims(data, axis=1)


    # Create a dataset from your data and labels
    dataset = UwbDataset(data, labels)
    train_dataset, test_dataset = split_dataset(dataset, train_size=train_size)


    return train_dataset, test_dataset