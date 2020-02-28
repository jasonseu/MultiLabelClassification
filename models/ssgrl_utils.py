import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(Semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, batch_size, img_feature_map, word_features):
        # temp, t = self.ex(batch_size, img_feature_map, word_features)
        # img_feature_map: 2 x 2048 x 3 x 3
        convsize = img_feature_map.size(3)

        f_wh_feature = img_feature_map.permute((0, 2, 3, 1)).contiguous().view(batch_size * convsize * convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature) # 18 x 1024
        f_wd_feature = self.fc_2(word_features).view(self.num_classes, 1, self.intermediary_dim) # 80 x 1024 - > 80 x 1 x 1024
        lb_feature = self.fc_3(torch.tanh(f_wd_feature * f_wh_feature).view(-1, self.intermediary_dim)) # 80 x 18 x 1024 -> 1440 x 1024

        coefficient = self.fc_a(lb_feature) # 1440 x 1
        coefficient = coefficient.view(self.num_classes, batch_size, -1).transpose(0, 1) # 80 x 2 x 9 -> 2 x 80 x 9
        # b = torch.all(torch.eq(coefficient, t))

        coefficient = F.softmax(coefficient, dim=2) # 2 x 80 x 9
        img_feature_map = img_feature_map.permute(0, 2, 3, 1).view(batch_size, convsize * convsize, -1) # 2 x 3 x 3 x 2048 -> 2 x 9 x 2048

        graph_net_input = torch.bmm(coefficient, img_feature_map)   # 2 x 80 x 2048
        return graph_net_input

    # def ex(self, batch_size, img_feature_map, word_features):
    #     convsize = img_feature_map.size()[3]

    #     img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2),2,3)
    #     f_wh_feature = img_feature_map.contiguous().view(batch_size*convsize*convsize, -1)  # 18 x 2048
    #     f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*convsize*convsize, 1, -1).repeat(1, self.num_classes, 1) # 18 x 80 x 1024

    #     f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, 1024).repeat(batch_size*convsize*convsize,1,1) # 18 x 80 x 1024
    #     lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1,1024)) # 18 x 80 x 1024 -> 1440 x 1024
    #     coefficient = self.fc_a(lb_feature) # 1440 x 1

        # t = self.fc_a(self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).transpose(0, 1).contiguous().view(-1,1024))) # 
        
        # 1440 x 1 -> 2 x 3 x 3 x 80 -> 2 x 80 x 3 x 3 -> 2 x 80 x 9
        # coefficient = torch.transpose(torch.transpose(coefficient.view(batch_size, convsize, convsize, self.num_classes),2,3),1,2).view(batch_size, self.num_classes, -1)
        # t = coefficient
        # coefficient = F.softmax(coefficient, dim=2) 
        # coefficient = coefficient.view(batch_size, self.num_classes, convsize, convsize) # 2 x 80 x 3 x 3
        # coefficient = torch.transpose(torch.transpose(coefficient,1,2),2,3) # 2 x 3 x 3 x 80
        # coefficient = coefficient.view(batch_size, convsize, convsize, self.num_classes, 1).repeat(1,1,1,1,self.image_feature_dim)  # 2 x 3 x 3 x 80 x 2048
        # img_feature_map = img_feature_map.view(batch_size, convsize, convsize, 1, self.image_feature_dim).repeat(1, 1, 1, self.num_classes, 1)* coefficient # 2 x 3 x 3 x 80 x 2048
        # graph_net_input = torch.sum(torch.sum(img_feature_map,1) ,1)    # 2 x 80 x 2048
        # return graph_net_input, t


class GGNN(nn.Module):
    def __init__(self, input_dim, time_step, in_matrix, out_matrix):
        super(GGNN, self).__init__()
        self.input_dim = input_dim
        self.time_step = time_step
        self._in_matrix = in_matrix
        self._out_matrix = out_matrix

        self.fc_eq3_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(-1, self.input_dim)
        node_num = self._in_matrix.size(0)
        batch_aog_nodes = input.view(batch_size, node_num, self.input_dim)
        batch_in_matrix = self._in_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
        batch_out_matrix = self._out_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
        for t in range(self.time_step):
            # eq(2)
            av = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(batch_out_matrix, batch_aog_nodes)), 2)
            av = av.view(batch_size * node_num, -1)
            flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes))
            #eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes))

            flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv * hv
            batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)
        return batch_aog_nodes

class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x,2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

