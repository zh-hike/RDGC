from torch.utils.data import Dataset
from ..util import data_preprocess, get_missmatrix
import numpy as np
from abc import abstractmethod

class CommonDataset(Dataset):
    def __init__(self, root_path, normalize, miss_rate, num_view, num_sample):
        self.root_path = root_path
        self.normalize = normalize
        self.X = []
        self.miss_matrix = get_missmatrix(miss_rate, num_view, num_sample)
        self.y = None

    def prepare(self):
        self._load_data()
        for i in range(len(self.X)):
            assert isinstance(self.X[i], np.ndarray)
            self.X[i] = self.X[i].astype('float32')
            self.X[i] = self.X[i] * self.miss_matrix[i]
        assert isinstance(self.y, np.ndarray)

        self.X = data_preprocess(self.X, self.normalize)
        self.y = self.y - self.y.min()

    @abstractmethod
    def _load_data(self):
        pass

    def __getitem__(self, idx):
        return [x[idx] for x in self.X], self.y[idx], [miss[idx] for miss in self.miss_matrix]

    def __len__(self):
        return len(self.y)