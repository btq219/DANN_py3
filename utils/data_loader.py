import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


# class GetLoader(data.Dataset):
#     def __init__(self, data_root, data_list, transform=None):
#         self.root = data_root
#         self.transform = transform
#
#         f = open(data_list, 'r')
#         data_list = f.readlines()
#         f.close()
#
#         self.n_data = len(data_list)
#
#         self.img_paths = []
#         self.img_labels = []
#
#         for data in data_list:
#             self.img_paths.append(data[:-3])
#             self.img_labels.append(data[-2])
#
#     def __getitem__(self, item):
#         img_paths, labels = self.img_paths[item], self.img_labels[item]
#         imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
#
#         if self.transform is not None:
#             imgs = self.transform(imgs)
#             labels = int(labels)
#
#         return imgs, labels
#
#     def __len__(self):
#         return self.n_data


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

        mat_data = loadmat(file)['Scenario'+ str(i+1)]
        n, _ = mat_data.shape
        mat_labels = np.ones(n) * i

        data.append(mat_data)
        labels.append(mat_labels)

    # Concatenate data and labels from each file into a single array
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = np.expand_dims(data, axis=1)


    # Create a dataset from your data and labels
    dataset = UwbDataset(data, labels)

    return dataset