import numpy as np
import scipy
import scipy.sparse as sp
import torch
import dgl
import torch as th
from sklearn.preprocessing import OneHotEncoder



# def encode_onehot(labels):
#     labels = labels.reshape(-1, 1)
#     enc = OneHotEncoder()
#     enc.fit(labels)
#     labels_onehot = enc.transform(labels).toarray()
#     return labels_onehot
#
#
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()



def load_Freebase_data(prefix=r'..\dataset\freebase'):

    # 结构特征提取
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    np.fill_diagonal(adjM, adjM.diagonal() + 1) # 直接在原始邻接矩阵的对角线上加1 实现关系特征和身份编码的相加
    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:3492, :]
    M_nodes_rows = torch.FloatTensor(M_nodes_rows)

    features = M_nodes_rows

    labels = np.load(prefix + '/labels.npy')#加载标签，4019

    MAM = scipy.sparse.load_npz(prefix + '/mam.npz')
    MDM = scipy.sparse.load_npz(prefix + '/mdm.npz')
    MWM = scipy.sparse.load_npz(prefix + '/mwm.npz')

    g1 = dgl.DGLGraph(MAM)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(MDM)
    g3 = dgl.DGLGraph(MWM)
    g = [g1, g2, g3]

    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_Aminer_data(prefix=r'..\dataset\aminer'):

    # 结构特征提取
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    np.fill_diagonal(adjM, adjM.diagonal() + 1) # 直接在矩阵的对角线上加1 实现关系特征和身份编码的相加
    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:6564, :]
    P_nodes_rows = torch.FloatTensor(P_nodes_rows)

    # 维度不匹配 无法相加
    # feat_p = scipy.sparse.eye(6564)
    # feat_p = th.FloatTensor(preprocess_features(feat_p))

    # 内存不够用
    # feat_all = scipy.sparse.eye(55783)
    # feat_all = th.FloatTensor(preprocess_features(feat_all))
    # feat_p = feat_all[:6564, :]

    # feat = P_nodes_rows + feat_p
    # features = torch.FloatTensor(feat)

    features = P_nodes_rows


    labels = np.load(prefix + '/labels.npy')#加载标签，4019

    PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    PRP = scipy.sparse.load_npz(prefix + '/prp.npz')

    g1 = dgl.DGLGraph(PAP)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(PRP)
    g = [g1, g2]


    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx

