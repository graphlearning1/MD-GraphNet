from __future__ import division
from __future__ import print_function

import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import MD_GraphNet
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt


def generate_labelid(lables, num, idx, label_seed):
    index = torch.zeros(lables.max()+1)
    train_id = []

    idx = idx.cpu().numpy()
    np.random.seed(label_seed)
    np.random.shuffle(idx)
    idx = torch.tensor(idx, dtype=torch.long)
    for l in idx:
    # for i,l in enumerate(lables):
        if index[lables[l]]<num:
            train_id.append(l)
            index[lables[l]] += 1
        else:
            continue
        if index.min()==num:
            train_id = np.array(train_id)
            return train_id

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", default='acm',help="dataset", type=str)  #citeseer，BlogCatalog,flickr,| uai， acm，
    parse.add_argument("-l", "--labelrate", default=20, help="labeled data for train per class", type = int)
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed

    seed = config.seed
    if use_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)

   
    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test, idx_val = load_data(config)

    model = MD_GraphNet(nfeat = config.fdim,
              nhid1 = config.nhid1,
              nhid2 = config.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout)
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, att, shared_loss, spec_loss, disttill_loss, vis_emb = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])

        loss = loss_class + config.share_v*shared_loss + config.spec_v*spec_loss + config.kl_loss*disttill_loss
        loss.backward()
        optimizer.step()

        acc_test, macro_f1 = main_test(model, idx_test)
        acc_val, _ = main_test(model, idx_val)

        print(epochs, "%.4f" % (acc_val.item()), "%.4f" % (acc_test.item()))
        return loss.item(), acc_test.item(), macro_f1.item(), acc_val.item()

    def eval_edge_pred(emb, sadj, fadj):
        emb = emb - emb.mean()
        emb = F.normalize(emb, p=2, dim=1)
        # Dot product
        adj_pred = torch.mm(emb, emb.T)
        edge_s = torch.where(sadj.to_dense()>0)
        edge_f = torch.where(fadj.to_dense()>0)
        edge_f = torch.cat((edge_f[0], edge_f[1]), dim=0).reshape(2, -1).cpu().numpy()
        edge_s= torch.cat((edge_s[0], edge_s[1]), dim=0).reshape(2, -1).cpu().numpy()
        logits_s = adj_pred[edge_s].detach().cpu().numpy()
        logits_f = adj_pred[edge_f].detach().cpu().numpy()

        edge_labels_s = np.ones(logits_s.shape[0])
        edge_labels_f = np.ones(logits_f.shape[0])

        logits_s = np.nan_to_num(logits_s)
        logits_f = np.nan_to_num(logits_f)
        ap_score_s = average_precision_score(edge_labels_s, logits_s)
        ap_score_f = average_precision_score(edge_labels_f, logits_f)
        return ap_score_s, ap_score_f


    def main_test(model, id_t):
        model.eval()
        output, _, _, _, _, _ = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn)


        acc_test = accuracy(output[id_t], labels[id_t])
        label_max = []
        for idx in id_t:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[id_t].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1

    def get_accuracy(results):
        epoch_run, best_dev, acc_test, f1_test = 0.0, 0.0, 0.0, 0.0
        for e, d, t, f in results:
            if d >= best_dev:
                best_dev, acc_test, f1_test, epoch_run = d, t, f, e
        return acc_test, f1_test, epoch_run, best_dev

    acc_max = 0
    f1_max = 0
    epoch_max = 0


    results = []
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, acc_val = train(model, epoch)
        results += [(epoch, acc_val, acc_test, macro_f1)]
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    acc_test_val, f1_test_val, epoch_max_val, _ = get_accuracy(results)
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max),
          epoch_max_val, acc_test_val, f1_test_val
          )


    
    
