# coding : utf-8
# Author : yuxiang Zeng

import platform
import time

from torch.utils.data import Dataset, DataLoader

from datasets.data_generator import get_train_valid_test_dataset
from utility.utils import *
import numpy as np
import torch as t
import pickle as pk
import dgl as d
from tqdm import *


# 加载处理好的数据集
def load_data(args):
    string = args.path + f"SuccessRate_{args.MaxBlockBack}_{args.MaxRTT}.csv"
    data = pd.read_csv(string).to_numpy()
    final_data = []
    for i in range(len(data)):
        temp = data[i, 0].split('\t')
        ans = []
        for j in range(len(temp)):
            if temp[j] == '':
                continue
            ans.append(float(temp[j]))
        final_data.append(ans)
    final_data = np.array(final_data)
    return final_data


# 合并异常值数据矩阵
def merge_Tensor_outlier(Tensor, outlier):
    merge = Tensor
    for i in range(len(Tensor)):
        for j in range(len(Tensor[0])):
            if Tensor[i][j] >= 0 and outlier[i][j] == 1:
                merge[i][j] = 0

    return merge


# 数据集定义
class QoSDataset(Dataset):
    def __getitem__(self, index):
        output = self.idx[index]
        userIdx, itemIdx, value = t.as_tensor(output[0]).long(), t.as_tensor(output[1]).long(), output[2]
        return userIdx, itemIdx, value

    def __len__(self):
        return len(self.idx)

    def __init__(self, data, args):
        self.path = args.path
        self.args = args
        self.data = data
        self.data[self.data == -1] = 0
        self.idx = self.get_index(self.data)
        self.max_value = data.max()
        self.train_Tensor, self.valid_Tensor, self.test_Tensor, self.max_value = get_train_valid_test_dataset(self.data, args)

    @staticmethod
    def get_index(data):
        userIdx, itemIdx = data.nonzero()
        value = []
        for i in range(len(userIdx)):
            value.append(data[userIdx[i], itemIdx[i]])
        index = np.transpose([userIdx, itemIdx, np.array(value)])
        return t.tensor(index)

    def get_tensor(self):
        return self.train_Tensor, self.valid_Tensor, self.test_Tensor
