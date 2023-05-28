# coding:utf-8
# Author: yuxiang Zeng
import torch as t
import torch.optim

def get_loss_function(args):
    loss_function = None
    if args.loss_func == 'L1Loss':
        loss_function = t.nn.L1Loss()
    elif args.loss_func == 'MSELoss':
        loss_function = t.nn.MSELoss()
    elif args.loss_func == 'CauchyLoss':
        loss_function = CauchyLoss()
    elif args.loss_func == 'SmoothL1Loss':
        loss_function = t.nn.SmoothL1Loss()
    return loss_function


def get_optimizer(parameters, lr, decay, args):
    optimizer_name = args.optim
    learning_rate = lr
    weight_decay = decay

    if optimizer_name == 'SGD':
        optimizer = t.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Momentum':
        optimizer = t.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = t.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = t.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        optimizer = t.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = t.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        optimizer = t.optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adamax':
        optimizer = t.optim.Adamax(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer name")

    return optimizer


class CauchyLoss(t.nn.Module):
    def __init__(self, alpha=1.0):
        super(CauchyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        delta = self.alpha
        diff = y_true - y_pred
        loss = delta ** 2 * torch.log(1 + (diff / delta) ** 2)
        return torch.mean(loss)
