import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader



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



def get_dataset(data_files):
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

    return dataset