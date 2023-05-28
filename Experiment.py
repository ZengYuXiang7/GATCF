# coding : utf-8
# Author : yuxiang Zeng

from lib.data_loader import get_dataloaders
from lib.early_stop import EarlyStopping
from lib.metrics import ErrMetrics
from lib.train_tools import get_loss_function, get_optimizer
from models.model import GATMF

from datasets.dataset import *
from utility.utils import *
import numpy as np
from time import time
import collections
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
global log

# 训练函数
###################################################################################################################################
def train(model, train_loader, valid_loader, log, args):
    training_time = []
    model = model.cuda() if args.devices == 'gpu' else model
    loss_function = get_loss_function(args).cuda() if args.devices == 'gpu' else get_loss_function(args)
    optimizer_embeds = get_optimizer(model.get_embeds_parameters(), lr=args.lr, decay=args.decay, args=args)
    optimizer_tf = get_optimizer(model.get_attention_parameters(), lr=args.att_lr, decay=args.att_decay, args=args)
    scheduler_tf = t.optim.lr_scheduler.StepLR(optimizer_tf, step_size=args.lr_step, gamma=0.50)
    optimizer_mlp = get_optimizer(model.get_mlp_parameters(), lr=args.lr, decay=args.decay, args=args)
    early_stop = EarlyStopping(args, patience=args.early_stop, delta=0)
    best_model, best_epoch = None, None
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    for epoch in range(args.epochs):
        t.set_grad_enabled(True)
        t1 = time()
        max_grad = 1e-9
        model.train()
        for train_Batch in (train_loader):
            userIdx, itemIdx, value = train_Batch
            if args.devices == 'gpu':
                userIdx, itemIdx, value = to_cuda(userIdx, itemIdx, value)
            pred = model.forward(userIdx, itemIdx, True)
            loss = loss_function(pred.to(t.float32), value.to(t.float32))
            optimizer_zero_grad(optimizer_embeds, optimizer_tf)
            optimizer_zero_grad(optimizer_mlp)
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)
            optimizer_step(optimizer_embeds, optimizer_tf)
            optimizer_step(optimizer_mlp)
        model.eval()
        t2 = time()
        training_time.append(t2 - t1)
        t.set_grad_enabled(False)
        lr_scheduler_step(optimizer_embeds, scheduler_tf)

        validMAE, validRMSE, validNMAE, validMRE, validNPRE = model_testing(model, valid_loader, args)
        if args.verbose and (epoch + 1) % args.verbose == 0:
            log(f'Epoch {(epoch + 1):3d} : [loss = {loss:6.6f}] -- MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f} Training_time = {sum(training_time[:epoch]):.1f} s')
        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = validMAE, validRMSE, validNMAE, validMRE, validNPRE
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model.state_dict())
        if args.early_stop:
            early_stop(validMAE, model)
            if early_stop.early_stop:
                break
    sum_time = sum(training_time[: best_epoch])
    log(f'Best epoch {best_epoch :2d} : MAE : {validBestMAE:5.4f}  RMSE : {validBestRMSE:5.4f}  NMAE : {validBestNMAE:5.4f}  MRE : {validBestMRE:5.4f}  NPRE : {validBestNPRE:5.4f}\n')
    return best_model, sum_time
###################################################################################################################################


###################################################################################################################################
def model_testing(model, test_loader, args):
    writeIdx = 0
    preds = t.zeros((len(test_loader.dataset),)).to('cuda')
    reals = t.zeros((len(test_loader.dataset),)).to('cuda')
    t.set_grad_enabled(False)
    model.prepare_test_model()
    for test_Batch in test_loader:
        userIdx, itemIdx, value = test_Batch
        if args.devices == 'gpu':
            userIdx, itemIdx, value = to_cuda(userIdx, itemIdx, value)
        pred = model.forward(userIdx, itemIdx, False)
        preds[writeIdx:writeIdx + len(pred)] = pred
        reals[writeIdx:writeIdx + len(pred)] = value
        writeIdx += len(pred)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)
    t.set_grad_enabled(True)
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE
###################################################################################################################################


###################################################################################################################################
# 对该模型进行训练
def train_model(model, datasets, log, args):
    train_loader, valid_loader, test_loader = get_dataloaders(datasets, args)
    best_model, training_time = train(model, train_loader, test_loader, log, args)
    model.load_state_dict(best_model)
    MAE, RMSE, NMAE, MRE, NPRE = model_testing(model, test_loader, args)
    return MAE, RMSE, NMAE, MRE, NPRE, training_time
###################################################################################################################################


###################################################################################################################################
def run(log, args):
    set_settings(args, None)
    df = np.array(load_data(args))
    datasets = QoSDataset(np.array(df), args)
    model = GATMF(args)
    model.max_value = datasets.max_value
    set_seed(int(time()))
    MAE, RMSE, NMAE, NRMSE, NPRE, TIME = train_model(model, datasets, log, args)
    return MAE, RMSE, NMAE, NRMSE, NPRE, TIME
###################################################################################################################################


###################################################################################################################################
def main(log, args):
    log(str(args))

    # print(torch.__version__)
    # 设定时间种子
    seed = int(time()) % 2023 if args.random_state is None else args.random_state
    set_seed(seed)

    Metrics = []

    results = collections.defaultdict(list)
    for roundId in range(args.rounds):
        MAE, RMSE, NMAE, NRMSE, NPRE, TIME = run(log, args)
        results = result_append(results, MAE, RMSE, NMAE, NRMSE, NPRE, TIME)
        metrics = f'Round {roundId + 1} : MAE = {MAE :.4f}, RMSE = {RMSE :.4f}, NMAE = {NMAE :.4f}, MRE = {NRMSE :.4f}, NPRE = {NPRE :.4f} TIME = {TIME:.2f}s'
        Metrics.append(metrics)
    RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE, TIME = np.mean(results['MAE']), np.mean(results['RMSE']), np.mean(results['NMAE']), np.mean(results['NRMSE']), np.mean(results['NPRE']), np.mean(results['TIME'])

    if args.rounds != 0:
        for roundId in range(args.rounds):
            log(str(Metrics[roundId]))

    log(f'\nResults : MAE = {RunMAE :.4f}, RMSE = {RunRMSE :.4f}, NMAE = {RunNMAE :.4f}, MRE = {RunMRE :.4f}, NPRE = {RunNPRE :.4f} TIME = {TIME:.2f}s\n')

    return RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE, TIME
###################################################################################################################################


###################################################################################################################################
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # 路径设置
    parser.add_argument('--path', nargs='?', default='./datasets/data/')
    parser.add_argument('--dataset', type=str, default='rt')
    parser.add_argument('--MaxBlockBack', type=int, default=100)
    parser.add_argument('--MaxRTT', type=int, default=5000)
    parser.add_argument('--density', type=float, default=0.05)
    parser.add_argument('--interaction', type=str, default='GATMF')
    # 实验常用
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--att_lr', type=float, default=4e-3)
    parser.add_argument('--att_decay', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--devices', type=str, default='gpu')  # cpu
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--valid', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--quick_train', type=int, default=0)
    # 基础设置
    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--max_gradient', type=float, default=0.30)
    parser.add_argument('--lr_step', type=int, default=50)

    # 模型超参数
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.00)
    parser.add_argument('--alpha', type=float, default=0.20)

    args = parser.parse_args()

    log = Logger(args)
    log('Experiment start!')
    main(log, args)
    log('Experiment success!\n')
###################################################################################################################################
