import time
import yaml
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import AdamW

from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from model import STRIPE
from utils import set_random_seeds, accuracy, generate_dynamic_data

from dgraphfin import DGraphFin
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse

args = argparse.ArgumentParser()

def model_setting(args):
    # initialize seed
    if args.model_seed is not None:
        set_random_seeds(args.model_seed)
    else:
        set_random_seeds()

    # set device
    device = torch.device(args.cudaID) if torch.cuda.is_available() else torch.device('cpu')

    train_data = generate_dynamic_data(args.dataset, anomaly_per=args.anomaly_rate, mode='train')
    test_data = generate_dynamic_data(args.dataset, anomaly_per=args.anomaly_rate, mode='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, drop_last=True)


    # init model and set optimizer
    model = STRIPE(args).to(device)

    return train_loader, test_loader, model, device

def loss_func(adjs, A_hats, attrs_l, X_hats, alpha):

    attribute_cost_list, structure_cost_list, cost_list = [], [], []
    for adj, A_hat, attrs, X_hat in zip(adjs, A_hats, attrs_l, X_hats):
        # Attribute reconstruction loss
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)

        cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

        cost_list.append(torch.unsqueeze(cost, dim=1))
        attribute_cost_list.append(attribute_cost.detach().cpu().numpy())
        structure_cost_list.append(structure_cost.detach().cpu().numpy())

    cost = torch.mean(torch.concat(cost_list, dim=1), dim=1)
    structure_cost = np.mean(structure_cost_list)
    attribute_cost = np.mean(attribute_cost_list)


    return cost, structure_cost, attribute_cost

def train(b, model, p_items):
    model.train()
    b = b.to(device)

    optimizer.zero_grad()

    outputs, A_hats, separateness_loss, compactness_loss = model(b, p_items, train=True)

    loss_pro = args.beta * torch.mean(separateness_loss) + (1 - args.beta) * torch.mean(compactness_loss)


    adjs = to_dense_adj(b.edge_index, batch=b.batch).to(b.x.device)
    attrs_l = torch.reshape(b.x, outputs.size())

    loss_rec, stru_loss, attr_loss = loss_func(adjs, A_hats, attrs_l, outputs, args.alpha)

    loss = args.gamma * torch.mean(loss_rec) + (1 - args.gamma) * loss_pro

    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy(), loss_rec.detach().cpu().numpy(), loss_pro.detach().cpu().numpy()

def eval(b, model):
    model.eval()
    b = b.to(device)

    with torch.no_grad():
        outputs, A_hats, compactness_loss = model(b, p_items, train=False)

        adjs = to_dense_adj(b.edge_index, batch=b.batch).to(b.x.device)
        attrs_l = torch.reshape(b.x, outputs.size())

        loss, _, _ = loss_func(adjs, A_hats, attrs_l, outputs, args.alpha)
        compactness_loss = torch.mean(compactness_loss, dim=1)
        loss = args.gamma * loss + (1 - args.gamma) * compactness_loss

    # ano_score, ano_label = loss[:b.batch_size], b.y[:b.batch_size]
    ano_score, ano_label = loss, b.y[:args.num_nodes]
    ano_score = ano_score.detach().cpu().numpy()
    ano_label = ano_label.detach().cpu().numpy()

    return ano_score, ano_label

def save_model(epoch):

    saved_content = {}

    saved_content['model_para'] = model.state_dict()
    saved_content['epoch'] = epoch

    # torch.save(saved_content, 'checkpoint/{}/{}_{}.pth'.format(args.dataset,args.setting, gen_num))
    torch.save(saved_content, 'checkpoint/{}/model_para_{}.pth'.format(args.dataset, args.epochs))

    return


def load_model(epoch):

    loaded_content = torch.load('checkpoint/{}/model_para_{}.pth'.format(args.dataset, epoch),
                                    map_location=lambda storage, loc: storage)

    model.load_state_dict(loaded_content['model_para'])

    # print("Loaded epoch number: " + str(loaded_content['epoch']))
    # print("Successfully loaded!")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='./config/REDDIT.yaml',
                        type=str, help='the configuration to use')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_name}...')
    config = yaml.load(
        open(args.config_name),
        Loader=yaml.FullLoader
    )

    args = argparse.Namespace(**config)
    train_loader, test_loader, model, device = model_setting(args)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)


    p_items = F.normalize(torch.rand((args.proto_size, args.proto_dim), dtype=torch.float),
                          dim=1).to(device)  # Initialize the memory items

    best_auc = 0.
    train_time_l =[]
    print('Training begins')
    for epoch in tqdm(range(1, args.epochs + 1)):
        l_ano_score, l_ano_label, l_loss = [], [], []
        l_rec_loss, l_pro_loss = [], []

        # Training
        train_begin = time.time()
        for b in train_loader:
            loss, rec_loss, pro_loss = train(b, model, p_items)
            l_loss.append(loss)
            l_rec_loss.append(rec_loss)
            l_pro_loss.append(pro_loss)
        # loss, rec_loss, pro_loss = train(data, model, p_items)
        # l_loss.append(loss)
        # l_rec_loss.append(rec_loss)
        # l_pro_loss.append(pro_loss)

        if epoch % args.eval_epoch == 0 or epoch == args.epochs + 1:
            for b in test_loader:
                ano_score, ano_label = eval(b, model)
                # print('ano_score', ano_score.shape)
                # print('ano_label', ano_label.shape)
                l_ano_score.append(ano_score), l_ano_label.append(ano_label)
            # ano_score, ano_label = eval(data, model)
            # l_ano_score.append(ano_score), l_ano_label.append(ano_label)

            loss_avg = np.mean(l_loss)

            all_ano_score = np.concatenate(l_ano_score, axis=0)
            all_ano_label = np.concatenate(l_ano_label, axis=0).astype(int)
            loss_rec = np.concatenate(l_rec_loss, axis=0)

            recall, macro_f1, AUC, acc, precision = accuracy(all_ano_score, all_ano_label)

            if AUC > best_auc:
                best_auc = AUC
                save_model(epoch)
            train_time_l.append(time.time() - train_begin)

    print('Training finished!')
    print('Inference begins')
    auc_max, pre_rec, recall_rec, acc_rec, f1_rec = 0., 0., 0., 0., 0.
    inf_time_l = []
    for round in tqdm(range(1, args.eval_rounds + 1)):
        test_begin = time.time()
        load_model(args.epochs)
        l_ano_score, l_ano_label, l_loss = [], [], []
        for b in test_loader:
            ano_score, ano_label = eval(b, model)
            l_ano_score.append(ano_score)
            l_ano_label.append(ano_label)
        # ano_score, ano_label = eval(data, model)
        # l_ano_score.append(ano_score)
        # l_ano_label.append(ano_label)
        all_ano_score = np.concatenate(l_ano_score, axis=0)
        all_ano_label = np.concatenate(l_ano_label, axis=0)

        recall, macro_f1, AUC, acc, precision = accuracy(all_ano_score, all_ano_label)
        if AUC > auc_max:
            saved_dict = {'ano_score': all_ano_score, 'ano_label': all_ano_label}
            np.save('./eval/{}/saved.npy'.format(args.dataset), saved_dict)
            auc_max = AUC
            pre_rec, recall_rec, acc_rec, f1_rec = precision, recall, acc, macro_f1

        inf_time_l.append(time.time() - test_begin)

    print('Best Inference AUC: ', auc_max)
    print('Inference Precision: {:.4f}'.format(pre_rec), flush=True)
    print('Inference Recall: {:.4f}'.format(recall_rec), flush=True)
    print('Inference ACC: {:4f}'.format(acc_rec), flush=True)
    print('Inference Mac_F1: {:4f}'.format(f1_rec), flush=True)
    print('Avg Train_time_per_epoch: {:.4f}s'.format(np.mean(train_time_l)))
    print('Avg Infer_time_per_epoch: {:.4f}s'.format(np.mean(inf_time_l)))
