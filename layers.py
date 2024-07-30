import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math

from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''
        Align the input and output
        :param c_in:
        :param c_out:
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            pad_len = int(self.c_out - self.c_in)
            return F.pad(x, [0, 0, 0, 0, pad_len, 0]) # x: (c,t,n)
        return x

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, act='relu', padding=0):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == 'GLU':
            self.conv = nn.Conv2d(c_in, c_out * 2, (Kt, 1), 1, padding)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (Kt, 1), 1, padding)

    def forward(self, x):
        x_in = self.align(x)[:, self.Kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(torch.nn.Module):
    # from bgrl
    def __init__(self, c_in, c_out, act='prelu',
                 batch_norm=True, batchnorm_mm=0.99, layer_norm=False,
                 weight_standardization=False):
        super(SpatioConvLayer, self).__init__()

        assert batch_norm != layer_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_standardization = weight_standardization
        # self.align = Align(c_in, c_out)

        # Set the activation function
        if act == 'prelu':
            self.activation = torch.nn.PReLU()
        elif act == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif act == 'softmax':
            self.activation = torch.nn.Softmax()
        else:
            raise ValueError("Invalid activation function")

        self.convg = GCNConv(c_in, c_out)

        # Add batch normalization if specified
        if self.batch_norm:
            self.norm = BatchNorm(c_out, momentum=batchnorm_mm)

        # Add layer normalization if specified
        if self.layer_norm:
            self.norm = LayerNorm(c_out)
        self.reset_parameters()


    def forward(self, x, edge_index):
        if self.weight_standardization:
            self.standardize_weights()

        x = self.convg(x, edge_index)
        # x_gc = self.norm(x)
        # x_in = self.align(x_gc)
        x = self.activation(x)
        return x

    def reset_parameters(self):
        self.convg.reset_parameters()
        self.norm.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.layers.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'