from torch.utils.data import Dataset
import numpy as np

class MNIST(Dataset):
    def __init__(self, 
                    miss_rate=0.1,
                    normalize="MinMaxScaler"):
        self.X = []    # list  [nxd1, nxd2, ,,,,, nxdv]
        self.y = None    # np.ndarray
        self.miss_matrix = None   # np.ndarray,   vxn

        self.check()

    def check(self):
        assert isinstance(self.X, list)
        assert isinstance(self.y, np.ndarray)
        self.y = self.y.astype('int64')
        assert isinstance(self.miss_matrix, (list, np.ndarray))
        assert len(self.X) == len(self.miss_matrix)
        for i in range(len(self.X)):
            assert isinstance(self.X[i], np.ndarray)
            assert self.X[i].shape[0] == self.y.shape[0]
            self.X[i] = self.X[i].astype('float32')
            assert isinstance(self.miss_matrix[i], np.ndarray)


       

    def __getitem__(self, idx):
        return [x[idx] for x in self.X], self.y[idx], [miss[idx] for miss in self.miss_matrix]
    

    def __len__(self):
        return self.X[0].shape[0]