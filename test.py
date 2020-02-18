import os
import json
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam, SGD, lr_scheduler

from datasets import data_factory
from models import model_factory
from utils.data_loader import get_loader
from utils.train_utils import *
from utils.metrics import *

torch.backends.cudnn.benchmark = True

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args

        test_dataset = data_factory[args.dataset](self.args, 'test')
        self.test_loader = get_loader(test_dataset, args, 'test')
        test_dataset = data_factory[args.dataset](self.args, 'val')
        self.test_loader = get_loader(test_dataset, args, 'val')
        self.num_classes = test_dataset.num_classes

        self.model = model_factory[args.model](self.args, self.num_classes)
        self.model.cuda()

        if self.args.loss == 'BCElogitloss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == 'tencentloss':
            self.criterion = TencentLoss(self.num_classes)

        self.voc12_mAP = VOC12mAP(self.num_classes)
        self.average_loss = AverageLoss(self.args.batch_size)
        self.average_topk_meter = TopkAverageMeter(self.num_classes, topk=self.args.topk)
        self.average_threshold_meter = ThresholdAverageMeter(self.num_classes, threshold=self.args.threshold)

    def run(self):
        checkpoint = load_checkpoint(self.args, True)
        self.model.load_state_dict(checkpoint)
        print(f'loading best checkpoint success')
        
        self.model.eval()
        self.voc12_mAP.reset()
        self.average_loss.reset()
        self.average_topk_meter.reset()
        self.average_threshold_meter.reset()
        desc = "EVALUATION - loss: {:.4f}"
        pbar = tqdm(total=len(self.test_loader), leave=False, desc=desc.format(0))
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)
                loss = loss.cpu().numpy()

                y = y.cpu().numpy()
                confidence = torch.sigmoid(pred_y)
                confidence = confidence.cpu().numpy()
                self.voc12_mAP.update(confidence, y)
                self.average_loss.update(loss)
                self.average_topk_meter.update(confidence, y)
                self.average_threshold_meter.update(confidence, y)

                pbar.desc = desc.format(loss)
                pbar.update(1)
        pbar.close()

        ap_list, mAP = self.voc12_mAP.compute()
        mLoss = self.average_loss.compute()
        self.average_topk_meter.compute()
        self.average_threshold_meter.compute()

        res = {
            'mAP': mAP,
            'ap_list': ap_list,
            'topk_cp': self.average_topk_meter.cp,
            'topk_cr': self.average_topk_meter.cr,
            'topk_cf1': self.average_topk_meter.cf1,
            'topk_op': self.average_topk_meter.op,
            'topk_or': self.average_topk_meter.or_,
            'topk_of1': self.average_topk_meter.of1,
            'threshold_cp': self.average_threshold_meter.cp,
            'threshold_cr': self.average_threshold_meter.cr,
            'threshold_cf1': self.average_threshold_meter.cf1,
            'threshold_op': self.average_threshold_meter.op,
            'threshold_or': self.average_threshold_meter.or_,
            'threshold_of1': self.average_threshold_meter.of1,
        }
        with open('result.json', 'w') as fw:
            json.dump(res, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--model', type=str, default='ssgrl')
    parser.add_argument('--loss', type=str, default='BCElogitloss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--pretrain_model', type=str, default='tmp/resnet101-5d3b4d8f.pth')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--scale_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=576)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()
    name_mapping = {
        'vg500': 'visual_genome',
        'coco': 'coco',
        'oi': 'open_images',
        'in': 'imagenet',
        'tc': 'tencent'
    }
    args.dataset_fullname = name_mapping.get(args.dataset)
    print(args)
    
    tester = Tester(args)
    tester.run()
