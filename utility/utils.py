# Author : yuxiang Zeng
# 损失函数，反馈日志
import numpy as np
import pandas as pd
import torch
import random
import csv
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_settings(args, config):
    if args.debug:
        args.epochs = 1
        args.record = 0
        args.lr = 1e-3
        args.decay = 1e-3

    if args.density == 0.10:
        # args.att_lr = 8e-3
        pass
    return args

class Logger:
    def __init__(self, args):
        self.args = args

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())), string)

    def __call__(self, string):
        if self.args.verbose:
            self.log(string)

    def print(self, string):
        self.args.verbose = 1
        self.__call__(string)
        self.args.verbose = 0


def to_cuda(*tensors):
    """
    将不定数量的张量转移到CUDA设备
    :param tensors: 不定数量的张量
    :return: 转移到CUDA设备后的张量
    """
    return [tensor.to('cuda') for tensor in tensors]

def get_grad(loss, parameters):
    gards = torch.autograd.grad(loss, parameters, retain_graph=True)
    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gards]))
    return grad_norm

def optimizer_zero_grad(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def optimizer_step(*optimizers, scaler=None):
    for optimizer in optimizers:
        if scaler is not None:
            scaler.step(optimizer)
        else:
            optimizer.step()

def lr_scheduler_step(*lr_scheduler):
    for scheduler in lr_scheduler:
        scheduler.step()

def result_append(results, new_MAE, new_RMSE, new_NMAE, new_NRMSE, new_NPRE, new_TIME):
    results['MAE'].append(new_MAE)
    results['RMSE'].append(new_RMSE)
    results['NMAE'].append(new_NMAE)
    results['NRMSE'].append(new_NRMSE)
    results['NPRE'].append(new_NPRE)
    results['TIME'].append(new_TIME)
    return results

def makedir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    return False

def computer_info():
    import platform

    def showinfo(tip, info):
        print("{} : {}".format(tip, info))

    showinfo("操作系统及版本信息", platform.platform())
    showinfo('获取系统版本号', platform.version())
    showinfo('获取系统名称', platform.system())
    showinfo('系统位数', platform.architecture())
    showinfo('计算机类型', platform.machine())
    showinfo('计算机名称', platform.node())
    showinfo('处理器类型', platform.processor())
    showinfo('计算机相关信息', platform.uname())

