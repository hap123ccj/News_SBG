import torch
from torch.utils.data import Dataset
import os

class MyDataset(Dataset):
    def __init__(self, inputpath):
        assert os.path.exists(inputpath+'label.pl')
        assert os.path.exists(inputpath+'feature.pl')
        self.y = torch.load(inputpath+'label.pl')
        self.x = torch.load(inputpath+'feature.pl')
        # if not self.x.shape[0]==self.y.shape[0]:
        #     print(self.y.shape)
        #     self.y = self.y[:self.x.shape[0]]
        #     print(self.y.shape)
        print(self.x.shape[0])
        assert self.x.shape[0]==self.y.shape[0]
    #定义初始化变量
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    #定义每次取出的对应数值
    def __len__(self):
        return self.x.shape[0]
    #定义tensor的总长度
