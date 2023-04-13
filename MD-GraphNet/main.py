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
import time

from sklearn.manifold import TSNE
import tensorboardX
from tensorboardX import SummaryWriter

def mad_gap_regularizer(intensor, neb_mask, rmt_mask):
    node_num, feat_num = intensor.size()

    input1 = intensor.expand(node_num, node_num, feat_num)
    input2 = input1.transpose(0, 1)

    input1 = input1.contiguous().view(-1, feat_num)
    input2 = input2.contiguous().view(-1, feat_num)

    simi_tensor = F.cosine_similarity(input1, input2, dim=1, eps=1e-8).view(node_num, node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor, neb_mask)
    rmt_dist = torch.mul(dist_tensor, rmt_mask)

    divide_neb = (neb_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list)
    rmt_mad = torch.mean(rmt_mean_list)

    mad_gap = rmt_mad - neb_mad

    return mad_gap

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout rate (1 - keep probability).')


    parse.add_argument("-d", "--dataset", default='cora',help="dataset", type=str)  #citeseer，BlogCatalog,flickr,| uai， acm，
    parse.add_argument("-l", "--labelrate", default=20, help="labeled data for train per class", type = int)
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

   
    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)
    # idx_val = torch.range(idx_train.max()+1, idx_test.min()-1).cuda()
    # idx_val = torch.tensor(idx_val, dtype=torch.long)

    # label_onehot = torch.zeros(features.shape[0], labels.max() + 1)
    # label_onehot.scatter_(1, torch.unsqueeze(labels, dim=1), 1)
    # sadj_hot = torch.where(sadj.to_dense() > 0, 1, 0)
    # labeled_m = torch.mm(label_onehot, label_onehot.T)
    #
    # sadj_hot_er = torch.mm(sadj_hot, sadj_hot.T)
    #
    # # neb_mask = sadj_hot_er * labeled_m
    # # rmt_mask = sadj_hot_er - neb_mask
    #
    # neb_mask = labeled_m
    # rmt_mask = 1 - labeled_m



    model = MDRL(nfeat = config.fdim,
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
        label_onehot = torch.zeros(features.shape[0]).cuda()

        # neb_mask = neb_mask.cuda()
        # rmt_mask = rmt_mask.cuda()
    # label_onehot = label_onehot.scatter_(1, labels[idx_train], 1)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)



    def train(model, epochs,  first_tl, first_vl, batch_size):
        model.train()
        optimizer.zero_grad()

        output, att, shared_loss, spec_loss, disttill_loss = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn, batch_size, train=True)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        loss = loss_class + config.share_v*shared_loss + config.spec_v*spec_loss + config.kl_loss*disttill_loss
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()


        pred_node = torch.exp(output)

        acc_test, macro_f1, reftime = main_test(model)

        # val_loss = main_val(model)

        # first_tl.append(loss.item())
        # first_vl.append(val_loss.item())

        # if first_tl == 0:
        #     first_tl = loss
        #     first_vl = val_loss

        # writer.add_scalars("loss", {"train": loss, "dev": val_loss}, epochs + 1)
        # print(loss, val_loss)

        return loss.item(), acc_test.item(), macro_f1.item(), reftime, first_tl, first_vl

    def train_batch(model, epochs,first_tl, first_vl, batch_size):
        model.train()

        device = sadj.device
        num_nodes = sadj.shape[0]
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)

        true_loss = []
        true_false_loss2 = []
        true_false_loss = []

        cls_weight = 1 / num_batches
        for i in range(num_batches):
            optimizer.zero_grad()
            mask = indices[i * batch_size:(i + 1) * batch_size]

            emb1, emb2, emb3, emb, tadj = model.forward_batch(features, sadj, fadj, config.s_rec, config.sim_v,
                                                              config.l_knn, batch_size, train=True)

            loss_class = F.nll_loss(emb[idx_train], labels[idx_train])

            output = emb

            sadj_f = torch.where(sadj.to_dense() > 0, 1.0, 0.0)
            fadj_f = torch.where(fadj.to_dense() > 0, 1.0, 0.0)
            tadj_f = torch.where(tadj > 0, 1.0, 0.0)


            #ML loss
            same_adj_loss1, diff1_adj_loss1 = model.rec_A_loss(emb1[:,:model.cut_pos], sadj_f, fadj_f, tadj_f, mask)
            same_adj_loss2, diff1_adj_loss2 = model.rec_A_loss(emb2[:,:model.cut_pos], fadj_f, sadj_f, tadj_f, mask)
            same_adj_loss3, diff1_adj_loss3 = model.rec_A_loss(emb3[:,:model.cut_pos], tadj_f, sadj_f, fadj_f, mask)
            rec_loss = (same_adj_loss1 + diff1_adj_loss1 + same_adj_loss2 + diff1_adj_loss2 + same_adj_loss3 + diff1_adj_loss3)

            similarity_loss1 = 1 - model.dot_product_normalize(emb1[:,:model.cut_pos][mask], emb2[:,:model.cut_pos]).mean()
            similarity_loss2 = 1 - model.dot_product_normalize(emb1[:,:model.cut_pos][mask], emb3[:,:model.cut_pos]).mean()
            similarity_loss3 = 1 - model.dot_product_normalize(emb3[:,:model.cut_pos][mask], emb2[:,:model.cut_pos]).mean()
            similarity_loss = (similarity_loss1 + similarity_loss2 + similarity_loss3)
            shared_loss = config.s_rec * rec_loss + config.sim_v*similarity_loss

            #SL loss
            recself_loss1, recdiff_loss1 = model.rec_A_loss(emb1, sadj_f, fadj_f, tadj_f, mask)
            recself_loss2, recdiff_loss2 = model.rec_A_loss(emb2, fadj_f, sadj_f, tadj_f, mask)
            recself_loss3, recdiff_loss3 = model.rec_A_loss(emb3, tadj_f, sadj_f, fadj_f, mask)
            spec_loss = (recself_loss1) + (recself_loss2) + (recself_loss3)

            ##TD loss
            diss_loss1 = model.distill_loss(emb1[mask], emb[mask])
            diss_loss2 = model.distill_loss(emb2[mask], emb[mask])
            diss_loss3 = model.distill_loss(emb3[mask], emb[mask])
            disttill_loss = (diss_loss1 + diss_loss2 + diss_loss3)/3

            loss = cls_weight * loss_class + config.share_v*shared_loss + config.spec_v*spec_loss + config.kl_loss*disttill_loss

            loss.backward()
            optimizer.step()

        acc = accuracy(output[idx_train], labels[idx_train])
        pred_node = torch.exp(output)

        acc_test, macro_f1, reftime = main_test(model)

        # val_loss = main_val(model)

        # first_tl.append(loss.item())
        # first_vl.append(val_loss.item())

        # if first_tl == 0:
        #     first_tl = loss
        #     first_vl = val_loss

        # writer.add_scalars("loss", {"train": loss, "dev": val_loss}, epochs + 1)
        # print(loss, val_loss)

        return loss.item(), acc_test.item(), macro_f1.item(), reftime, first_tl, first_vl

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

    def main_val(model):
        model.eval()

        start = time.time()
        output,att, shared_loss, spec_loss, disttill_loss = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn)
        loss_class = F.nll_loss(output[idx_val], labels[idx_val])
        loss = loss_class + config.share_v * shared_loss + config.spec_v * spec_loss + config.kl_loss * disttill_loss
        return loss


    def main_test(model):
        model.eval()

        start = time.time()
        output = model(features, sadj, fadj, config.s_rec, config.sim_v, config.l_knn)
        end = time.time()
        total = end - start

        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')

        # mad_eval = mad_gap_regularizer(output, neb_mask, rmt_mask)
        # print("mad_gap的值", mad_eval, "acc", acc_test)
        print("acc", acc_test)
        return acc_test, macro_f1, total
    acc_max = 0
    f1_max = 0
    epoch_max = 0

    start = time.time()
    total = 0

    first_tl = []
    first_vl = []

    if args.dataset in ['citeseer', 'BlogCatalog', 'flickr', 'uai', 'cora', 'acm']:
        batch_size = None
    else:
        batch_size = 1024
    for epoch in range(config.epochs):
        if batch_size == None:
            loss, acc_test, macro_f1, reftime, first_tl, first_vl = train(model, epoch, first_tl, first_vl, batch_size)
        else:
            loss, acc_test, macro_f1, reftime, first_tl, first_vl = train_batch(model, epoch, first_tl, first_vl, batch_size)
        total = total + reftime
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    end = time.time()



    print(total/config.epochs)
    print("time:{:.4f}".format((end-start)/config.epochs))
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))


    
    
