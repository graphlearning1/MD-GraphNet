import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx
# from scipy.linalg import fractional_matrix_power, inv
from scipy.sparse.linalg import inv
import os

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def tensor_normalize(mx):
    index = torch.where(mx > 0)
    erjie = torch.zeros_like(mx)
    erjie[index[0], index[1]] = 1
    rowsum = torch.sum(erjie, dim=0)
    r_inv = torch.pow(rowsum, -1).view(-1, 1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diagflat(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx

def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + sp.eye(a.shape[0])  # A^ = A + I_n
    row_sum = np.array(np.sum(a, 1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^(-1/2)
    at = d_mat_inv_sqrt.dot(a).dot(d_mat_inv_sqrt)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((sp.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test


def load_graph(dataset, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'


    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    # att = config.ifatt
    #
    # if att == False:
    #     struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    #     sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    #     sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # else:
    #     sadj = sp.load_npz(config.structattgraph_path)
    # sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize(sadj+sp.eye(sadj.shape[0]))


    # if not os.path.exists(config.diff_path):
    #     tadj = compute_ppr(sadj)
    #     tadj = sparse_mx_to_torch_sparse_tensor(tadj)
    #     torch.save(tadj, config.diff_path)
    # else:
    #     tadj = torch.load(config.diff_path)
    #
    # if not os.path.exists(config.tadj_f):
    #     tadj_f = tadj.clone()
    #     tadj_f = tadj_f.to_dense()
    #     for i in range(tadj_f.shape[0]):
    #         tadj_f[i, tadj_f[i].argsort()[:-5]] = 0
    #         tadj_f[i, tadj_f[i].argsort()[-5:]] = 1
    #     tadj_f = tadj_f
    #     torch.save(tadj_f, config.tadj_f)
    # else:
    #     tadj_f = torch.load(config.tadj_f)
    #
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj

