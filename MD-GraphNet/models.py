import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc3 = GraphConvolution(nhid, nhid)
        # self.gc4 = GraphConvolution(nhid, nhid)
        # self.gc5= GraphConvolution(nhid, nhid)
        # self.gc6 = GraphConvolution(nhid, nhid)
        # self.gc7 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)

        # x = self.gc3(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc4(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc5(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc6(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc7(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta




class MDRL(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(MDRL, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN1(nfeat, nhid1, nhid2, dropout)
        self.SGCN3 = GCN1(nfeat, nhid1, nhid2, dropout)

        self.cut_pos = nhid2//2

        self.dropout = dropout

        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )
        # self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def dot_product_normalize(self, shared_1, shared_2):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        num_of_samples = shared_1.size(0)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        # Dot product
        match_map = torch.mm(shared_1, shared_2.T)
        return match_map

    def distill_loss(self, y_s, y_t, T=1):
        p_s = F.log_softmax(y_s / T, dim=1)

        p_t = F.softmax(y_t / T, dim=1)

        loss = -(p_t * p_s).sum(dim=1).mean()
        return loss



    def tensor_normalize(self, mx):
        index = torch.where(mx > 0)
        erjie = torch.zeros_like(mx)
        erjie[index[0], index[1]] = 1
        rowsum = torch.sum(erjie, dim=0)
        r_inv = torch.pow(rowsum, -1).view(-1, 1)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diagflat(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


    def forward(self, x, sadj, fadj, s_rec, sim_v, k, batch=None, train=False):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph


        # calculate label channel
        sim1 = self.dot_product_normalize(emb1, emb1)
        sim2 = self.dot_product_normalize(emb2, emb2)
        tadj = sim1 * sim2
        tadj = tadj.detach()

        k_adj = torch.zeros(tadj.shape).cuda()
        choose_k = tadj.topk(k, dim=1)[1]
        k_adj = k_adj.scatter_(1, choose_k, 1)

        tadj_f = k_adj + k_adj.T.multiply(k_adj.T > k_adj) - k_adj.multiply(k_adj.T > k_adj)
        tadj = self.tensor_normalize(tadj_f)

        emb3 = self.SGCN3(x, tadj)  # Special_GCN out3 -- label feature graph

        emb = torch.stack([emb1, emb2, emb3], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)

        if train:
            sadj_f = torch.where(sadj.to_dense() > 0, 1.0, 0.0)
            fadj_f = torch.where(fadj.to_dense() > 0, 1.0, 0.0)

            ##ML loss
            same_adj_loss1, diff1_adj_loss1 = self.rec_A_loss(emb1[:,:self.cut_pos], sadj_f, fadj_f, tadj_f, batch)
            same_adj_loss2, diff1_adj_loss2 = self.rec_A_loss(emb2[:,:self.cut_pos], fadj_f, sadj_f, tadj_f, batch)
            same_adj_loss3, diff1_adj_loss3 = self.rec_A_loss(emb3[:,:self.cut_pos], tadj_f, sadj_f, fadj_f, batch)
            rec_loss = (same_adj_loss1 + diff1_adj_loss1 + same_adj_loss2 + diff1_adj_loss2 + same_adj_loss3 + diff1_adj_loss3)

            similarity_loss1 = 1 - self.dot_product_normalize(emb1[:,:self.cut_pos], emb2[:,:self.cut_pos]).mean()
            similarity_loss2 = 1 - self.dot_product_normalize(emb1[:,:self.cut_pos], emb3[:,:self.cut_pos]).mean()
            similarity_loss3 = 1 - self.dot_product_normalize(emb3[:,:self.cut_pos], emb2[:,:self.cut_pos]).mean()
            similarity_loss = (similarity_loss1 + similarity_loss2 + similarity_loss3)
            shared_loss = s_rec * rec_loss + sim_v*similarity_loss

            #SL loss
            recself_loss1, recdiff_loss1 = self.rec_A_loss(emb1, sadj_f, fadj_f, tadj_f, batch)
            recself_loss2, recdiff_loss2 = self.rec_A_loss(emb2, fadj_f, sadj_f, tadj_f, batch)
            recself_loss3, recdiff_loss3 = self.rec_A_loss(emb3, tadj_f, sadj_f, fadj_f, batch)
            spec_loss = (recself_loss1) + (recself_loss2) + (recself_loss3)

            ##TD loss
            diss_loss1 = self.distill_loss(emb1, emb)
            diss_loss2 = self.distill_loss(emb2, emb)
            diss_loss3 = self.distill_loss(emb3, emb)
            disttill_loss = (diss_loss1 + diss_loss2 + diss_loss3)/3

            return output, att, shared_loss, spec_loss, disttill_loss
        else:
            return output

    def forward_batch(self, x, sadj, fadj, s_rec, sim_v, k, batch=None, train=False):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph


        # calculate label channel
        sim1 = self.dot_product_normalize(emb1, emb1)
        sim2 = self.dot_product_normalize(emb2, emb2)
        tadj = sim1 * sim2
        tadj = tadj.detach()

        k_adj = torch.zeros(tadj.shape).cuda()
        choose_k = tadj.topk(k, dim=1)[1]
        k_adj = k_adj.scatter_(1, choose_k, 1)

        tadj_f = k_adj + k_adj.T.multiply(k_adj.T > k_adj) - k_adj.multiply(k_adj.T > k_adj)
        tadj = self.tensor_normalize(tadj_f)

        emb3 = self.SGCN3(x, tadj)  # Special_GCN out3 -- label feature graph

        emb = torch.stack([emb1, emb2, emb3], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)

        return emb1, emb2, emb3, emb, tadj

        # if train:
        #     sadj_f = torch.where(sadj.to_dense() > 0, 1.0, 0.0)
        #     fadj_f = torch.where(fadj.to_dense() > 0, 1.0, 0.0)
        #
        #     ##ML loss
        #     same_adj_loss1, diff1_adj_loss1 = self.rec_A_loss(emb1[:,:self.cut_pos], sadj_f, fadj_f, tadj_f, batch)
        #     same_adj_loss2, diff1_adj_loss2 = self.rec_A_loss(emb2[:,:self.cut_pos], fadj_f, sadj_f, tadj_f, batch)
        #     same_adj_loss3, diff1_adj_loss3 = self.rec_A_loss(emb3[:,:self.cut_pos], tadj_f, sadj_f, fadj_f, batch)
        #     rec_loss = (same_adj_loss1 + diff1_adj_loss1 + same_adj_loss2 + diff1_adj_loss2 + same_adj_loss3 + diff1_adj_loss3)
        #
        #     similarity_loss1 = 1 - self.dot_product_normalize(emb1[:,:self.cut_pos], emb2[:,:self.cut_pos]).mean()
        #     similarity_loss2 = 1 - self.dot_product_normalize(emb1[:,:self.cut_pos], emb3[:,:self.cut_pos]).mean()
        #     similarity_loss3 = 1 - self.dot_product_normalize(emb3[:,:self.cut_pos], emb2[:,:self.cut_pos]).mean()
        #     similarity_loss = (similarity_loss1 + similarity_loss2 + similarity_loss3)
        #     shared_loss = s_rec * rec_loss + sim_v*similarity_loss
        #
        #     #SL loss
        #     recself_loss1, recdiff_loss1 = self.rec_A_loss(emb1, sadj_f, fadj_f, tadj_f, batch)
        #     recself_loss2, recdiff_loss2 = self.rec_A_loss(emb2, fadj_f, sadj_f, tadj_f, batch)
        #     recself_loss3, recdiff_loss3 = self.rec_A_loss(emb3, tadj_f, sadj_f, fadj_f, batch)
        #     spec_loss = (recself_loss1) + (recself_loss2) + (recself_loss3)
        #
        #     ##TD loss
        #     diss_loss1 = self.distill_loss(emb1, emb)
        #     diss_loss2 = self.distill_loss(emb2, emb)
        #     diss_loss3 = self.distill_loss(emb3, emb)
        #     disttill_loss = (diss_loss1 + diss_loss2 + diss_loss3)/3
        #
        #     return output, att, shared_loss, spec_loss, disttill_loss


    def rec_A_loss(self, x, sadj, fadj, adj2, mask=None):

        norm_w_s = sadj.shape[0] ** 2 / float((sadj.shape[0] ** 2 - sadj.sum()) * 2)
        pos_weight_s = torch.FloatTensor([float(sadj.shape[0] ** 2 - sadj.sum()) / sadj.sum()]).cuda()

        norm_w_f = fadj.shape[0] ** 2 / float((fadj.shape[0] ** 2 - fadj.sum()) * 2)
        pos_weight_f = torch.FloatTensor([float(fadj.shape[0] ** 2 - fadj.sum()) / fadj.sum()]).cuda()

        norm_w_f2 = adj2.shape[0] ** 2 / float((adj2.shape[0] ** 2 - adj2.sum()) * 2)
        pos_weight_f2 = torch.FloatTensor([float(adj2.shape[0] ** 2 - adj2.sum()) / adj2.sum()]).cuda()

        if mask == None:

            edge_pre = self.dot_product_normalize(x, x)

            true_loss = norm_w_s * F.binary_cross_entropy_with_logits(edge_pre, sadj, pos_weight=pos_weight_s)
            true_false_loss = norm_w_f * F.binary_cross_entropy_with_logits(edge_pre, fadj, pos_weight=pos_weight_f)
            true_false_loss2 = norm_w_f2 * F.binary_cross_entropy_with_logits(edge_pre, adj2, pos_weight=pos_weight_f2)
            true_false_loss = true_false_loss + true_false_loss2

        else:

            edge_pre_batch = self.dot_product_normalize(x[mask], x)

            # true_loss.append(F.binary_cross_entropy_with_logits(edge_pre_batch, sadj[mask], pos_weight=pos_weight_s))
            # true_false_loss2.append(F.binary_cross_entropy_with_logits(edge_pre_batch, fadj[mask], pos_weight=pos_weight_f))
            # true_false_loss.append(F.binary_cross_entropy_with_logits(edge_pre_batch, adj2[mask], pos_weight=pos_weight_f2))

            true_loss = F.binary_cross_entropy_with_logits(edge_pre_batch, sadj[mask], pos_weight=pos_weight_s)
            true_false_loss = F.binary_cross_entropy_with_logits(edge_pre_batch, fadj[mask], pos_weight=pos_weight_f)
            true_false_loss2 = F.binary_cross_entropy_with_logits(edge_pre_batch, adj2[mask], pos_weight=pos_weight_f2)
            # true_loss = norm_w_s * (true_loss/num_batches)
            # true_false_loss = norm_w_f * (true_false_loss/num_batches)
            # true_false_loss2 = norm_w_f2 * (true_false_loss2/num_batches)

            true_loss = norm_w_s * true_loss.mean()
            true_false_loss = norm_w_f * true_false_loss.mean()
            true_false_loss2 = norm_w_f2 * true_false_loss2.mean()
            true_false_loss = true_false_loss + true_false_loss2

        return true_loss, true_false_loss
'''
0.06589440822601318
time:0.3231
epoch:35 acc_max: 0.7510 f1_max: 0.7064
'''