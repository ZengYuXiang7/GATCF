# coding:utf-8
# Author: yuxiang Zeng
from torch.utils.data import Dataset, DataLoader
from torch.nn import *
from tqdm import *
from time import time
import numpy as np
import pickle as pk
import pandas as pd
import torch as t


# 精度计算
def ErrMetrics(realVec, estiVec):
    # 将 y_true 和 y_pred 转换为 numpy 数组或 PyTorch 张量
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)
    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    absError = np.abs(estiVec - realVec)
    MAE = np.mean(absError)
    RMSE = np.linalg.norm(absError) / np.sqrt(np.array(absError.shape[0]))
    NMAE = np.sum(np.abs(realVec - estiVec)) / np.sum(realVec)
    relativeError = absError / realVec
    MRE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))
    NPRE = np.array(np.percentile(relativeError, 90))  #
    return MAE, RMSE, NMAE, MRE, NPRE


if __name__ == '__main__':
    pred = t.Tensor([[1, 2, 3, 4]])
    true = t.Tensor([[2, 1, 4, 5]])
    print(ErrMetrics(pred, true))

