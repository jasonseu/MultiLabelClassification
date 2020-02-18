import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_save, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_save = False
        else:
            self.best_score = score
            self.counter = 0
        return is_save, is_terminate


class TencentLoss(object):
    def __init__(self, class_num, pos_weight=12.0):
        super(TencentLoss, self).__init__()
        self.pos_weight = torch.FloatTensor(class_num).fill_(pos_weight).cuda()
        self.pre_status = torch.IntTensor(class_num).fill_(-1).cuda()
        self.t = None

    def __call__(self, input, target):
        r = self._get_adaptive_weight(target)
        output = F.binary_cross_entropy_with_logits(input, target, weight=r, pos_weight=self.pos_weight)
        return output

    def _get_adaptive_weight(self, target):
        class_status = torch.sum(target, dim=0)
        cur_status = class_status > torch.tensor(0.0).cuda()
        cur_status = cur_status.type_as(self.pre_status)
        if torch.all(torch.eq(self.pre_status, cur_status)):
            self.t += 1
        else:
            self.t = 1
            self.pre_status = cur_status

        pos_r = max(0.01, math.log10(10/(0.01+self.t)))
        neg_r = max(0.01, math.log10(10/(8+self.t)))
        pos_r = target.clone().fill_(pos_r)
        neg_r = target.clone().fill_(neg_r)

        r = torch.where(target == 1, pos_r, neg_r)
        return r


def get_summary_writer(args):
    log_dir = os.path.join('logs', args.dataset, args.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def save_checkpoint(save_dict, args, is_best=False):
    model = 'best-model' if is_best else 'model'
    prefix = f'{args.dataset}-{args.model}-{args.loss}-{args.optimizer}-{args.lr}'
    save_path = f'tmp/{prefix}-{model}.pth.tar'
    torch.save(save_dict, save_path)

def load_checkpoint(args, is_best=False):
    model = 'best-model' if is_best else 'model'
    prefix = f'{args.dataset}-{args.model}-{args.loss}-{args.optimizer}-{args.lr}'
    save_path = f'tmp/{prefix}-{model}.pth.tar'
    print(f'loading model from checkpoint file {save_path}')
    checkpoint = torch.load(save_path)
    return checkpoint