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



def load_ACM_data(prefix=r'.\ACM'):

    # 结构特征提取
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:4019, :]
    P_nodes_rows = torch.FloatTensor(P_nodes_rows)

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列
    features_0 = torch.FloatTensor(features_0)

    feat = torch.cat([P_nodes_rows, features_0], dim=1)
    # feat = torch.cat([features_0,P_nodes_rows], dim=1)


    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    # PAP = scipy.sparse.load_npz(prefix + '/adj_pap_one.npz')
    # PSP = scipy.sparse.load_npz(prefix + '/adj_psp_one.npz')

    PAP = scipy.sparse.load_npz(prefix + '/pap.npz') # 与上述元路径邻接矩阵相同
    PSP = scipy.sparse.load_npz(prefix + '/psp.npz')

    g1 = dgl.DGLGraph(PAP)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(PSP)
    g = [g1, g2]

    # features = torch.FloatTensor(feat)
    features = feat
    # features = P_nodes_rows

    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_IMDB_data(prefix=r'.\IMDB_processed'):
    # 结构特征提取
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:4278, :]
    M_nodes_rows = torch.FloatTensor(M_nodes_rows)

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_0 = torch.FloatTensor(features_0)

    feat = torch.cat([M_nodes_rows, features_0], dim=1)
    # feat = torch.cat([features_0,P_nodes_rows], dim=1)


    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引


    MDM = scipy.sparse.load_npz(prefix + '/0/MDM.npz') # 与上述元路径邻接矩阵相同
    MAM = scipy.sparse.load_npz(prefix + '/0/MAM.npz')

    g1 = dgl.DGLGraph(MDM)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(MAM)
    g = [g1, g2]

    features = torch.FloatTensor(feat)
    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_Freebase_data(prefix=r'.\freebase'):

    # 结构特征提取
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:3492, :]
    M_nodes_rows = torch.FloatTensor(M_nodes_rows)

    feat_m = scipy.sparse.eye(3492) #节点类型0 m的特征，3492行3492列
    feat_m = th.FloatTensor(preprocess_features(feat_m))

    feat = torch.cat([M_nodes_rows, feat_m], dim=1)


    labels = np.load(prefix + '/labels.npy')#加载标签，4019


    MAM = scipy.sparse.load_npz(prefix + '/mam.npz')
    MDM = scipy.sparse.load_npz(prefix + '/mdm.npz')
    MWM = scipy.sparse.load_npz(prefix + '/mwm.npz')

    g1 = dgl.DGLGraph(MAM)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(MDM)
    g3 = dgl.DGLGraph(MWM)
    g = [g1, g2, g3]

    # features = torch.FloatTensor(feat)
    features = M_nodes_rows

    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_Aminer_data(prefix=r'.\aminer'):

    # 结构特征提取
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:6564, :]
    P_nodes_rows = torch.FloatTensor(P_nodes_rows)

    feat_p = scipy.sparse.eye(6564) #节点类型0 m的特征，3492行3492列
    feat_p = th.FloatTensor(preprocess_features(feat_p))

    feat = torch.cat([P_nodes_rows, feat_p], dim=1)


    labels = np.load(prefix + '/labels.npy')#加载标签，4019

    PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    PRP = scipy.sparse.load_npz(prefix + '/prp.npz')

    g1 = dgl.DGLGraph(PAP)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(PRP)
    g = [g1, g2]

    # features = torch.FloatTensor(feat)
    features = P_nodes_rows

    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_DBLP_data(prefix=r'.\DBLP_processed'):

    # 结构特征提取
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    # 提取 P 类型节点的行
    A_nodes_rows = adjM[:4057, :]
    A_nodes_rows = torch.FloatTensor(A_nodes_rows)

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列
    features_0 = torch.FloatTensor(features_0)

    feat = torch.cat([A_nodes_rows, features_0], dim=1)
    features = feat

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    # PAP = scipy.sparse.load_npz(prefix + '/adj_pap_one.npz')
    # PSP = scipy.sparse.load_npz(prefix + '/adj_psp_one.npz')

    APA = scipy.sparse.load_npz(prefix + '/0/apa.npz') # 与上述元路径邻接矩阵相同
    APAPA = np.load(prefix + '/0/apapa.npy')
    APCPA = scipy.sparse.load_npz(prefix + '/0/apcpa.npz')
    APTPA = scipy.sparse.load_npz(prefix + '/0/aptpa.npz')


    g1 = dgl.DGLGraph(APA)      #存储节点、边及其特征
    src_nodes, dst_nodes = np.nonzero(APAPA)
    g2 = dgl.graph((src_nodes, dst_nodes))
    g3 = dgl.DGLGraph(APCPA)
    g4 = dgl.DGLGraph(APTPA)

    g = [g1, g2, g3, g4]

    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return g, features, labels, num_classes, train_idx, val_idx, test_idx