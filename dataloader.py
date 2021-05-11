import models.pointnet as pn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py


class ModelNet40(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.pcs, self.labels = self.load_train_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pcs[idx], self.labels[idx][0]

    def load_train_data(self):
        with h5py.File(self.data_dir + "ply_data_train0.h5") as f:
            pc = f["data"][...]
            labels = f["label"][...]

        for i in range(1, 5):
            with h5py.File(self.data_dir + "ply_data_train" + str(i) + ".h5") as f:
                pc_tmp = f["data"][...]
                labels_tmp = f["label"][...]
            pc = np.concatenate([pc, pc_tmp])
            labels = np.concatenate([labels, labels_tmp])

        return pc, labels

    @property
    def labels_map(self):
        with open(self.data_dir + "shape_names.txt", "r") as f:
            labels = f.readlines()

        return {i: labels[i].rstrip("\n") for i in range(len(labels))}
