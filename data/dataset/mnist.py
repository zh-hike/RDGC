from .common import CommonDataset
import os
import scipy.io as io

class MNIST(CommonDataset):
    def __init__(self, root_path, normalize, miss_rate, num_view, num_sample):
        super(MNIST, self).__init__(root_path, normalize, miss_rate, num_view, num_sample)
        self.prepare()
    
    def _load_data(self):
        path = os.path.join(self.root_path, 'mnist.mat')
        data = io.loadmat(path)
        self.X = [x.T.astype('float32') for x in data['X'][0]]
        self.y = data['truth'].squeeze().astype('int64')