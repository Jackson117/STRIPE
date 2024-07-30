import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Prototype(nn.Module):
    def __init__(self, proto_size, fea_dim, key_dim, temp_update, temp_gather, cudaID):
        super(Prototype, self).__init__()
        self.proto_size = proto_size
        self.fea_dim = fea_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        self.cudaID = cudaID

    def hard_neg_proto(self, proto, i):
        similarity = torch.matmul(proto, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys_var[max_idx]

    def random_pick_memory(self, proto, max_indices):

        m, d = proto.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, proto, max_indices, update_indices, score, query, train):
        device = torch.device(self.cudaID) if torch.cuda.is_available() else torch.device('cpu')

        m, d = proto.size()
        if train:
            query_update = torch.zeros((m, d)).to(device)
            random_update = torch.zeros((m, d)).to(device)
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # random_idx = torch.randperm(a)[0]
                    # idx = idx[idx != ex]
                    #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                    # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0
                    # random_update[i] = 0

            return query_update

        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # idx = idx[idx != ex]
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                else:
                    query_update[i] = 0

            return query_update

    def get_score(self, proto, query):
        h, w, d = query.size()
        m, d = proto.size()

        # query and mem are both matrices
        score = torch.matmul(query, torch.t(proto))  # h X w X m
        score = score.view(h * w, m)  # (h X w) X m

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def forward(self, query, keys, train=True):

        # query = torch.unsqueeze(query, dim=0)
        dims, t, n = query.size()  # d X T X N
        # query = F.normalize(query, dim=0)
        query = query.permute(1, 2, 0)  # T X N X d

        # train
        if train:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)
            # spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            # update
            updated_memory = self.update(query, keys, train)

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, spreading_loss, gathering_loss

        # test
        else:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)

            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)

            # update
            updated_memory = keys

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

    def update(self, query, keys, train):

        h, w, dims = query.size()  #h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        else:
            # only weighted sum update when test
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        # top-1 update
        # query_update = query_reshape[updating_indices][0]
        # updated_memory = F.normalize(query_update + keys, dim=1)

        return updated_memory.detach()

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self, query, keys, train):
        h, w, dims = query.size()  # h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0, reduction='none')

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys, train):

        h, w, dims = query.size()  # h X w X d

        loss_mse = torch.nn.MSELoss(reduction='none')

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)  # weights for q and p

        query_reshape = query.contiguous().view(h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  # the largest memory score

        # ||q_{ij} - d_m||_2
        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, updated_proto):
        h, w, dims = query.size()  # h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_proto, query)

        query_reshape = query.contiguous().view(h * w, dims)

        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_proto)  # (h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (h X w) X 2d
        updated_query = updated_query.view(h, w, 2 * dims)
        updated_query = updated_query.permute(2, 0, 1)

        return updated_query, softmax_score_query, softmax_score_memory