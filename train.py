# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import argparse
from argparse import Namespace

import torch
from torch import nn
from torch.optim import Adam, SGD, lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import model_factory
from lib.util import *
from lib.metrics import *
from lib.dataset import MLDataset

torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        train_transform = transforms.Compose([
            transforms.Resize((args.scale_size, args.scale_size)),
            transforms.RandomChoice([
                transforms.RandomCrop(640),
                transforms.RandomCrop(576),
                transforms.RandomCrop(512), 
                transforms.RandomCrop(384),
                transforms.RandomCrop(320)
            ]),
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        train_dataset = MLDataset(args.train_path, args.label_path, train_transform)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_transform = transforms.Compose([
            transforms.Resize((args.scale_size, args.scale_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        val_dataset = MLDataset(args.val_path, args.label_path, val_transform)
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        self.model = model_factory[args.model](args, args.num_classes)
        self.model.cuda()

        trainable_parameters = filter(lambda param: param.requires_grad, self.model.parameters())
        if args.optimizer == 'Adam':
            self.optimizer = Adam(trainable_parameters, lr=args.lr)
        elif args.optimizer == 'SGD':
            self.optimizer = SGD(trainable_parameters, lr=args.lr)

        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=2, verbose=True)
        if args.loss == 'BCElogitloss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'tencentloss':
            self.criterion = TencentLoss(args.num_classes)
        elif args.loss == 'focalloss':
            self.criterion = FocalLoss()
        self.early_stopping = EarlyStopping(patience=5)

        self.voc12_mAP = VOC12mAP(args.num_classes)
        self.average_loss = AverageLoss()
        self.average_topk_meter = TopkAverageMeter(args.num_classes, topk=args.topk)
        self.average_threshold_meter = ThresholdAverageMeter(args.num_classes, threshold=args.threshold)

        self.args = args
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=args.log_dir)

    def run(self):
        s_epoch = 0
        if self.args.resume:
            checkpoint = torch.load(self.args.ckpt_latest_path)
            s_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.early_stopping.best_score = checkpoint['best_score']
            print('loading checkpoint success (epoch {})'.format(s_epoch))
        
        for epoch in range(s_epoch, self.args.max_epoch):
            self.train(epoch)
            save_dict = {
                'epoch': epoch + 1,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'best_score': self.early_stopping.best_score
            }
            torch.save(save_dict, self.args.ckpt_latest_path)

            mAP = self.validation(epoch)
            self.lr_scheduler.step(mAP)
            is_save, is_terminate = self.early_stopping(mAP)
            if is_terminate:
                break
            if is_save:
                torch.save(self.model.state_dict(), self.args.ckpt_best_path)

    def train(self, epoch):
        self.model.train()
        if self.args.model == 'ssgrl':
            self.model.resnet_101.eval()
            self.model.resnet_101.layer4.train()
        for _, batch in enumerate(self.train_loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            pred_y = self.model(x)
            loss = self.criterion(pred_y, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.global_step % 400 == 0:
                self.writer.add_scalar('Loss/train', loss, self.global_step)

            print('TRAIN [epoch {}] loss: {:4f}'.format(epoch, loss))
            self.global_step += 1

    def validation(self, epoch):
        self.model.eval()
        self.voc12_mAP.reset()
        self.average_loss.reset()
        self.average_topk_meter.reset()
        self.average_threshold_meter.reset()
        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)

                y = y.cpu().numpy()
                pred_y = pred_y.cpu().numpy()
                loss = loss.cpu().numpy()
                self.voc12_mAP.update(pred_y, y)
                self.average_loss.update(loss, x.size(0))
                self.average_topk_meter.update(pred_y, y)
                self.average_threshold_meter.update(pred_y, y)

        _, mAP = self.voc12_mAP.compute()
        mLoss = self.average_loss.compute()
        self.average_topk_meter.compute()
        self.average_threshold_meter.compute()
        self.writer.add_scalar('Loss/val', mLoss, self.global_step)
        self.writer.add_scalar('mAP/val', mAP, self.global_step)

        print("Validation [epoch {}] mAP: {:.4f} loss: {:.4f}".format(epoch, mAP, mLoss))
        return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/coco_resnet101.yaml')
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    cfg['resume'] = args.resume
    args = Namespace(**cfg)
    print(args)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    trainer = Trainer(args)
    trainer.run()
