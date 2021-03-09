# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import json
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models import model_factory
from lib.util import *
from lib.metrics import *
from lib.dataset import MLDataset

torch.backends.cudnn.benchmark = True

class Evaluator(object):
    def __init__(self, args):
        super(Evaluator, self).__init__()
        self.args = args

        test_transform = transforms.Compose([
            transforms.Resize((args.scale_size, args.scale_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        ml_dataset = MLDataset(args.val_path, args.label_path, test_transform)
        self.data = ml_dataset.data
        self.labels = ml_dataset.labels
        self.data_loader = DataLoader(
            dataset=ml_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        self.model = model_factory[args.model](args, args.num_classes)
        self.model.cuda()

        if args.loss == 'BCElogitloss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'tencentloss':
            self.criterion = TencentLoss(args.num_classes)
        elif args.loss == 'focalloss':
            self.criterion = FocalLoss()

        self.args = args
        self.voc12_mAP = VOC12mAP(args.num_classes)
        self.average_topk_meter = TopkAverageMeter(args.num_classes, topk=args.topk)
        self.average_threshold_meter = ThresholdAverageMeter(args.num_classes, threshold=args.threshold)

    def run(self):
        model_dict = torch.load(self.args.ckpt_best_path)
        self.model.load_state_dict(model_dict)
        print(f'loading best checkpoint success')
        
        fw = open(os.path.join(self.args.output_dir, 'prediction.txt'), 'w')
        self.model.eval()
        self.voc12_mAP.reset()
        self.average_topk_meter.reset()
        self.average_threshold_meter.reset()
        desc = "EVALUATION - loss: {:.4f}"
        pbar = tqdm(total=len(self.data_loader), leave=False, desc=desc.format(0))
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)
                loss = loss.cpu().numpy()

                y = y.cpu().numpy()
                confidence = torch.sigmoid(pred_y)
                confidence = confidence.cpu().numpy()
                self.voc12_mAP.update(confidence, y)
                self.average_topk_meter.update(confidence, y)
                self.average_threshold_meter.update(confidence, y)

                topk_inds = np.argsort(-confidence)[:, :self.args.topk]
                i *= self.args.batch_size
                for j in range(x.size(0)):
                    img_name = os.path.basename(self.data[i+j][0])
                    pred_labels = [self.labels[ind] for ind in topk_inds[j]]
                    fw.write('{}\t{}\n'.format(img_name, ' '.join(pred_labels)))
                pbar.desc = desc.format(loss)
                pbar.update(1)
        pbar.close()
        fw.close()

        ap_list, mAP = self.voc12_mAP.compute()
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
        with open(os.path.join(self.args.output_dir, 'result.json'), 'w') as fw:
            json.dump(res, fw)

        print('model {} data {} mAP: {}'.format(self.args.model, self.args.data, mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/voc2012_resnet101.yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    args = Namespace(**cfg)
    print(args)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    evaluator = Evaluator(args)
    evaluator.run()
