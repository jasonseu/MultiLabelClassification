import os
import argparse
from tqdm import tqdm
import pickle

import torch
from torch import nn
from torch.optim import Adam, SGD, lr_scheduler

from datasets import data_factory
from models import model_factory
from utils.data_loader import get_loader
from utils.train_utils import *
from utils.metrics import *

torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        train_dataset = data_factory[args.dataset](self.args, 'train')
        self.train_loader = get_loader(train_dataset, args, 'train')
        self.num_classes = train_dataset.num_classes

        val_dataset = data_factory[args.dataset](self.args, 'val')
        self.val_loader = get_loader(val_dataset, args, 'val')

        self.model = model_factory[args.model](self.args, self.num_classes)
        self.model.cuda()

        trainable_parameters = filter(lambda param: param.requires_grad, self.model.parameters())
        if self.args.optimizer == 'Adam':
            self.optimizer = Adam(trainable_parameters, lr=self.args.lr)
        elif self.args.optimizer == 'SGD':
            self.optimizer = SGD(trainable_parameters, lr=self.args.lr)

        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, verbose=True)
        if self.args.loss == 'BCElogitloss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == 'tencentloss':
            self.criterion = TencentLoss(self.num_classes)
        self.early_stopping = EarlyStopping(patience=5)

        self.voc12_mAP = VOC12mAP(self.num_classes)
        self.average_loss = AverageLoss(self.args.batch_size)
        self.average_topk_meter = TopkAverageMeter(self.num_classes, topk=self.args.topk)
        self.average_threshold_meter = ThresholdAverageMeter(self.num_classes, threshold=self.args.threshold)

        self.global_step = 0
        self.writer = get_summary_writer(self.args)

    def run(self):
        start_epoch = 0
        if self.args.resume:
            checkpoint = load_checkpoint(self.args)
            start_epoch = checkpoint['epoch']
            self.global_step = (start_epoch - 1) * len(self.train_loader)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.early_stopping.best_score = checkpoint['best_score']
            print(f'loading checkpoint success (epoch {start_epoch})')
        
        for epoch in range(start_epoch, self.args.max_epoch):
            self.train(epoch)
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_score': self.early_stopping.best_score
            }
            save_checkpoint(save_dict, self.args)

            val_loss = self.validation(epoch)
            self.lr_scheduler.step(val_loss)
            is_save, is_terminate = self.early_stopping(-val_loss)
            if is_terminate:
                break
            if is_save:
                save_checkpoint(self.model.state_dict(), args, True)

    def train(self, epoch):
        self.model.train()
        if self.args.model == 'ssgrl':
            self.model.resnet_101.eval()
            self.model.resnet_101.layer4.train()
        desc = "TRAINING - loss: {:.4f}"
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=desc.format(0))
        for step, batch in enumerate(self.train_loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            pred_y = self.model(x)
            loss = self.criterion(pred_y, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            if self.global_step % 400 == 0:
                self.writer.add_scalar('Loss/train', loss, self.global_step)

            pbar.desc = desc.format(loss)
            pbar.update(1)
        pbar.close()

    def validation(self, epoch):
        self.model.eval()
        self.voc12_mAP.reset()
        self.average_loss.reset()
        self.average_topk_meter.reset()
        self.average_threshold_meter.reset()
        desc = "VALIDATION - loss: {:.4f}"
        pbar = tqdm(total=len(self.val_loader), leave=False, desc=desc.format(0))
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                pred_y = self.model(x)
                loss = self.criterion(pred_y, y)

                y = y.cpu().numpy()
                pred_y = pred_y.cpu().numpy()
                loss = loss.cpu().numpy()
                self.voc12_mAP.update(pred_y, y)
                self.average_loss.update(loss)
                self.average_topk_meter.update(pred_y, y)
                self.average_threshold_meter.update(pred_y, y)

                pbar.desc = desc.format(loss)
                pbar.update(1)

        ap_list, mAP = self.voc12_mAP.compute()
        mLoss = self.average_loss.compute()
        self.average_topk_meter.compute()
        self.average_threshold_meter.compute()
        self.writer.add_scalar('Loss/val', mLoss, self.global_step)
        self.writer.add_scalar('mAP/val', mAP, self.global_step)

        tqdm.write(f"Validation Results - Epoch: {epoch} mAP: {mAP:.4f} loss: {mLoss:.4f}")
        pbar.close()
        return mLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--model', type=str, default='ssgrl')
    parser.add_argument('--loss', type=str, default='BCElogitloss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--pretrain_model', type=str, default='tmp/resnet101-5d3b4d8f.pth')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=8)
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
    
    trainer = Trainer(args)
    trainer.run()
