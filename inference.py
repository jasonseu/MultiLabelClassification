import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

from datasets import data_factory
from models import model_factory
from utils.data_loader import get_transform
from utils.train_utils import load_checkpoint


def infer(args):
    label_path = f'data/{args.dataset_fullname}/{args.dataset}_label.txt'
    labels = [line.strip() for line in open(label_path)]
    num_classes = len(labels)

    model = model_factory[args.model](args, num_classes)
    model.cuda()

    checkpoint = load_checkpoint(args, True)
    model.load_state_dict(checkpoint)

    transform = get_transform(args, 'test')
    res = []
    for image_name in os.listdir(args.image_dir):
        image_path = os.path.join(args.image_dir, image_name)
        image_data = Image.open(image_path).convert('RGB')
        image_data = transform(image_data)
        image_data = np.expand_dims(image_data, 0)
        image_data = torch.from_numpy(image_data).cuda()
        confidence = torch.sigmoid(model(image_data))
        confidence = confidence.cpu().detach().numpy().squeeze()
        topk_inds = np.argsort(-confidence)[:args.topk]
        threshold_inds = np.argwhere(confidence > args.threshold).squeeze(-1)
        print('==='*30)
        print(image_name)
        print([labels[i] for i in topk_inds])
        print([labels[i] for i in threshold_inds])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--model', type=str, default='ssgrl')
    parser.add_argument('--loss', type=str, default='BCElogitloss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--pretrain_model', type=str, default='tmp/resnet101-5d3b4d8f.pth')

    parser.add_argument('--scale_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=576)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--image_dir', type=str, default='input')

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
    
    infer(args)