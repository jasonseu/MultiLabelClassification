# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-1-19
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch import nn
from torchvision import models

class ResNet101(nn.Module):

    def __init__(self, args, num_labels):
        super(ResNet101, self).__init__()
        self.network = models.resnet101(pretrained=False, num_classes=num_labels)

        print('loading pretrained model from imagenet')
        model_dict = self.network.state_dict()
        resnet_pretrained = torch.load(args.initmodel)
        pretrain_dict = {k:v for k, v in resnet_pretrained.items() if not k.startswith('fc')}
        model_dict.update(pretrain_dict)
        self.network.load_state_dict(model_dict)
        
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.layer4.parameters():
            param.requires_grad = True
        self.network.fc.requires_grad = True

    def forward(self, x):
        x = self.network(x)
        return x
