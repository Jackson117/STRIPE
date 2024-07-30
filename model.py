import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import SpatioConvLayer, TemporalConvLayer, GraphConvolution
from prototype import Prototype
from utils import fea_reshape_3d, fea_reshape_2d, normalize_adj
import scipy.sparse as sp

from torch_geometric.utils import to_dense_adj

class STRIPE(nn.Module):
    def __init__(self, args):
        super(STRIPE, self).__init__()
        self.encoder = STEncoder(Kt=args.Kt, blocks=[[args.input_dim, int(args.hid_dim), args.hid_dim], [args.hid_dim, int(args.hid_dim), args.hid_dim]],
                                 time_stamps=args.batch_size, num_nodes=args.num_nodes, dropout=args.dropout)
        self.decoder1 = Decoder1(Kt=args.Kt, blocks=[[args.hid_dim * 2, args.hid_dim, int(args.hid_dim * 2)], [args.hid_dim, args.input_dim]],
                               time_stamps=args.batch_size, num_nodes=args.num_nodes, dropout=args.dropout)
        self.decoder2 = Decoder2(Kt=args.Kt, blocks=[[args.hid_dim * 2, args.hid_dim, int(args.hid_dim * 2)],
                                                     [args.hid_dim, args.input_dim]],
                                 time_stamps=args.batch_size, num_nodes=args.num_nodes, dropout=args.dropout)
        self.prototype = Prototype(proto_size=args.proto_size, fea_dim=args.hid_dim, key_dim=args.hid_dim,
                                   temp_update=args.temp_update, temp_gather=args.temp_gather, cudaID=args.cudaID)

    def forward(self, data, keys, train=True):
        fea = self.encoder(data)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss \
                = self.prototype(fea, keys, train)

            output, A_hat = self.decoder1(updated_fea, data)
            # A_hat = self.decoder2(updated_fea, data)

            return output, A_hat, separateness_loss, compactness_loss
        else:
            updated_fea, updated_memory, softmax_score_query, softmax_score_memory, compactness_loss = self.prototype(
                fea, keys, train)

            output, A_hat = self.decoder1(updated_fea, data)
            # A_hat = self.decoder2(updated_fea, data)

            return output, A_hat, compactness_loss

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        # list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.prototype.parameters())
        return list(self.encoder.parameters()) + list(self.decoder1.parameters()) + list(self.decoder2.parameters())

class STEncoder(nn.Module):
    def __init__(self, Kt, blocks, time_stamps, num_nodes, dropout=0.1):
        super(STEncoder, self).__init__()
        self.Kt = Kt
        self.num_nodes = num_nodes
        self.time_stamps = time_stamps
        self.dropout = dropout

        c = blocks[0]
        self.sconv1 = SpatioConvLayer(c[0], c[1])
        self.tconv1 = TemporalConvLayer(Kt, c[1], c[2], 'GLU')
        self.ln1 = nn.LayerNorm(c[2])

        c = blocks[1]
        self.sconv2 = SpatioConvLayer(c[1], c[2])
        out_len = time_stamps - (Kt - 1) if time_stamps - (Kt - 1) > 1 else 1
        self.tconv2 = TemporalConvLayer(out_len, c[2], c[2], 'GLU')
        self.ln2 = nn.LayerNorm(c[2])

    def forward(self, data):
        # in_times = x.size(1)
        # if in_times < self.receptive_field:
        #     x = F.pad(x, (0,0,0,0, self.receptive_field - in_times, 0))

        x, edge_index = data.x, data.edge_index
        # n_idx, batch = data.n_idx, data.batch

        # Sconv block 1
        x = self.sconv1(x, edge_index)  # x.size(): batch_num_nodes * fea_dim
        x = F.dropout(x, self.dropout, training=self.training)

        # Sconv block 2
        x = self.sconv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # Tconv block 1
        x = torch.reshape(torch.unsqueeze(x, dim=0), (self.time_stamps, self.num_nodes, x.size(-1)))
        # x.size(): fea_dim * time_stamps * num_nodes
        x = x.permute(2, 0, 1)
        x = self.tconv1(x)  # T_out = T_in - (K_t - 1)
        x = self.ln1(x.permute(1, 2, 0))

        # Tconv block 2
        # x.size(): fea_dim * time_stamps * num_nodes
        x = x.permute(2, 0, 1)
        x = self.tconv2(x)  # T_out = T_in - (K_t - 1)
        x = self.ln2(x.permute(1, 2, 0))


        x = x.permute(2, 0, 1) # x.size(): D*1*N

        return x

class Decoder1(nn.Module):
    def __init__(self, Kt, blocks, time_stamps, num_nodes, dropout=0.1):
        super(Decoder1, self).__init__()
        
        self.Kt = Kt
        self.num_nodes = num_nodes
        self.time_stamps = time_stamps
        self.dropout = dropout

        c = blocks[0]
        out_len = time_stamps - (Kt - 1) if time_stamps - (Kt - 1) > 1 else 1

        # spatial decoder
        self.sconv1 = SpatioConvLayer(c[0], c[1])
        self.sconv3 = SpatioConvLayer(c[0], c[1])

        self.dec_tconv1 = TemporalConvLayer(1, c[0], c[1], 'GLU')
        self.upsample1 = nn.ConvTranspose2d(c[1], c[1], (out_len, 1), 1)

        self.dec_tconv2 = TemporalConvLayer(1, c[1], c[2], 'GLU')
        self.upsample2 = nn.ConvTranspose2d(c[2], c[2], (Kt, 1), 1)

        c = blocks[1]
        self.sconv2 = SpatioConvLayer(c[0], c[1])


    def forward(self, x, data):

        x = self.dec_tconv1(x)  # T_out = T_in
        x = self.upsample1(x)   # T_out = T_in + (out_len - 1)

        x = self.dec_tconv2(x)
        x = self.upsample2(x)

        x_in = torch.reshape(x, (self.time_stamps * self.num_nodes, x.size(0)))

        x = self.sconv1(x_in, data.edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.sconv2(x, data.edge_index)
        out = torch.reshape(out, (self.time_stamps, self.num_nodes, out.size(-1)))

        x = self.sconv3(x_in, data.edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        xs = torch.reshape(x, (self.time_stamps, self.num_nodes, x.size(-1)))
        A_hats = []
        for i in range(self.time_stamps):
            x = torch.squeeze(xs[i, :, :])
            a_hat = x @ x.T
            A_hats.append(torch.unsqueeze(a_hat, dim=0))

        A_hats = torch.concat(A_hats, dim=0)

        return out, A_hats


class Decoder2(nn.Module):
    def __init__(self, Kt, blocks, time_stamps, num_nodes, dropout=0.1):
        super(Decoder2, self).__init__()

        self.Kt = Kt
        self.num_nodes = num_nodes
        self.dropout = dropout

        c = blocks[0]
        self.sconv1 = SpatioConvLayer(c[0], c[1])
        # self.sconv1 = GraphConvolution(c[0], c[1])

    def forward(self, x, data):
        # x = fea_reshape_2d(x.permute(1, 2, 0), data.n_idx, data.batch, self.num_nodes)
        x = torch.squeeze(x.permute(1, 2, 0))
        x = self.sconv1(x, data.edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x





