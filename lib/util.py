# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math

import torch
import torch.nn.functional as F


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

class FocalLoss(object):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, input, target):
        input_prob = torch.sigmoid(input)
        hard_easy_weight = (1 - input_prob) * target + input_prob * (1 - target)
        posi_nega_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = (posi_nega_weight * torch.pow(hard_easy_weight, self.gamma)).detach()
        focal_loss = F.binary_cross_entropy_with_logits(input, target, weight=focal_weight)
        return focal_loss