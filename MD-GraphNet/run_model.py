from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import MDRL
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.manifold import TSNE

# if __name__ == "__main__":
def run_model_per(kl_loss,s_rec, sim_v, spec_v, dataset, labelrate):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_file = "./config/" + str(labelrate) + str(dataset) + ".ini"
    config = Config(config_file)

    config.s_rec = s_rec
    config.sim_v = sim_v
    config.spec_v = spec_v
    config.kl_loss = kl_loss

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

    sadj, fadj = load_graph(labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)

    model = MDRL(nfeat=config.fdim,
                 nhid1=config.nhid1,
                 nhid2=config.nhid2,
                 nclass=config.class_num,
                 n=config.n,
                 dropout=config.dropout)
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
        output, att, shared_loss, spec_loss, disttill_loss = model(features, sadj, fadj, config.s_rec, config.sim_v,
                                                                   config.l_knn, train=True)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])

        loss = loss_class + config.share_v * shared_loss + config.spec_v * spec_loss + config.kl_loss * disttill_loss
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        acc_test, macro_f1 = main_test(model)

        return loss.item(), acc_test.item(), macro_f1.item()


    def eval_edge_pred(emb, sadj, fadj):
        emb = emb - emb.mean()
        emb = F.normalize(emb, p=2, dim=1)
        # Dot product
        adj_pred = torch.mm(emb, emb.T)
        edge_s = torch.where(sadj.to_dense() > 0)
        edge_f = torch.where(fadj.to_dense() > 0)
        edge_f = torch.cat((edge_f[0], edge_f[1]), dim=0).reshape(2, -1).cpu().numpy()
        edge_s = torch.cat((edge_s[0], edge_s[1]), dim=0).reshape(2, -1).cpu().numpy()
        logits_s = adj_pred[edge_s].detach().cpu().numpy()
        logits_f = adj_pred[edge_f].detach().cpu().numpy()

        edge_labels_s = np.ones(logits_s.shape[0])
        edge_labels_f = np.ones(logits_f.shape[0])

        logits_s = np.nan_to_num(logits_s)
        logits_f = np.nan_to_num(logits_f)
        ap_score_s = average_precision_score(edge_labels_s, logits_s)
        ap_score_f = average_precision_score(edge_labels_f, logits_f)
        return ap_score_s, ap_score_f


    def main_test(model):
        model.eval()
        output = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn)

        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1 = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))
    return acc_max



