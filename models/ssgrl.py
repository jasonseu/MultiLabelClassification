import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.ssgrl_backbone import resnet101
from models.ssgrl_utils import Semantic, GGNN, Element_Wise_Layer

class SSGRL(nn.Module):
    def __init__(self, args, num_classes=80):
        super(SSGRL, self).__init__()
        self.args = args

        self.num_classes = num_classes
        self.image_feature_dim = 2048
        self.output_dim = 2048
        self.word_feature_dim = 300
        self.word_file = self.args.embedding_path
        self.graph_file = self.args.graph_path
        self.time_step = 3
        
        self._word_features = self._load_features()
        self._in_matrix, self._out_matrix = self.load_matrix()

        self.word_semantic = Semantic(
            num_classes=self.num_classes,
            image_feature_dim=self.image_feature_dim,
            word_feature_dim=self.word_feature_dim
        )

        self.graph_net = GGNN(
            input_dim=self.image_feature_dim,
            time_step=self.time_step,
            in_matrix=self._in_matrix,
            out_matrix=self._out_matrix
        )

        self.fc_output = nn.Linear(2 * self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.output_dim)

        self.resnet_101 = resnet101()
        self._load_pretrain_model()
        for param in self.resnet_101.parameters():
            param.requires_grad = False
        for param in self.resnet_101.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        batch_size = x.size()[0]
        img_feature_map = self.resnet_101(x)
        graph_net_input = self.word_semantic(batch_size, img_feature_map, self._word_features)
        graph_net_feature = self.graph_net(graph_net_input)

        output = torch.cat((graph_net_feature.view(batch_size*self.num_classes,-1), graph_net_input.view(-1, self.image_feature_dim)), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, self.num_classes, self.output_dim)
        result = self.classifiers(output)
        return result 

    def _load_pretrain_model(self):
        model_dict = self.resnet_101.state_dict()
        print('loading pretrained model from imagenet')
        resnet_pretrained = torch.load(self.args.initmodel)
        pretrain_dict = {k:v for k, v in resnet_pretrained.items() if not k.startswith('fc')}
        model_dict.update(pretrain_dict)
        self.resnet_101.load_state_dict(model_dict)

    def _load_features(self):
        return torch.from_numpy(np.load(self.word_file).astype(np.float32)).cuda()

    def load_matrix(self):
        mat = np.load(self.graph_file)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix = torch.from_numpy(_in_matrix).cuda()
        _out_matrix = torch.from_numpy(_out_matrix).cuda()
        return _in_matrix, _out_matrix
