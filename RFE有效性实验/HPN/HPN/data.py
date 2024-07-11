import numpy as np
import scipy
import pickle
import torch
import dgl
import scipy.sparse as sp



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def load_ACM_data(prefix=r'C:\Users\Yanyeyu\Desktop\实验2\HPN\dataset/ACM'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    # PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    # PSP = scipy.sparse.load_npz(prefix + '/psp.npz')
    PAP = np.load(prefix + '/PAP_only_one.npy')
    PSP = np.load(prefix + '/PSP_only_one.npy')
    PAP = scipy.sparse.csr_matrix(PAP)
    PSP = scipy.sparse.csr_matrix(PSP)
    g1 = dgl.DGLGraph(PAP)
    g2 = dgl.DGLGraph(PSP)
    g=[g1, g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_IMDB_data(prefix=r'E:\图神经网络\图神经网络\模型及代码\实验2\HPN\dataset\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    MAM = np.load(prefix + '/MAM_only_one.npy')
    MAM = scipy.sparse.csr_matrix(MAM)
    MDM = np.load(prefix + '/MDM_only_one.npy')
    MDM = scipy.sparse.csr_matrix(MDM)


    g1 = dgl.DGLGraph(MAM)
    g2 = dgl.DGLGraph(MDM)
    g=[g1,g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes=3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    return g, features, labels, num_classes, train_idx, val_idx, test_idx


def load_DBLP_data(prefix=r'..\dataset\DBLP'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    APA = scipy.sparse.load_npz(prefix + '/apa_only_one.npz').A
    APA = scipy.sparse.csr_matrix(APA)
    APCPA = scipy.sparse.load_npz(prefix + '/apcpa_only_one.npz').A
    APCPA = scipy.sparse.csr_matrix(APCPA)
    APTPA = scipy.sparse.load_npz(prefix + '/aptpa_only_one.npz').A
    APTPA = scipy.sparse.csr_matrix(APTPA)



    g1 = dgl.DGLGraph(APA)
    g2 = dgl.DGLGraph(APCPA)
    g3 = dgl.DGLGraph(APTPA)
    g=[g1,g2,g3]
    #g = [g1, g2]
    features = torch.FloatTensor(features_0)
    labels=torch.LongTensor(labels)
    num_classes=4
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    return g, features, labels, num_classes, train_idx, val_idx, test_idx


def load_Freebase_data(prefix=r'..\dataset\freebase'):

    feat_m = scipy.sparse.eye(3492) #节点类型0 m的特征，3492行3492列
    feat_m = torch.FloatTensor(preprocess_features(feat_m))

    labels = np.load(prefix + '/labels.npy')#加载标签，4019

    MAM = scipy.sparse.load_npz(prefix + '/mam.npz')
    MDM = scipy.sparse.load_npz(prefix + '/mdm.npz')
    MWM = scipy.sparse.load_npz(prefix + '/mwm.npz')

    g1 = dgl.DGLGraph(MAM)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(MDM)
    g3 = dgl.DGLGraph(MWM)
    g = [g1, g2, g3]

    features = torch.FloatTensor(feat_m)
    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx



def load_Aminer_data(prefix=r'..\dataset\aminer'):

    feat_p = scipy.sparse.eye(6564)
    feat_p = torch.FloatTensor(preprocess_features(feat_p))

    labels = np.load(prefix + '/labels.npy')

    PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    PRP = scipy.sparse.load_npz(prefix + '/prp.npz')

    g1 = dgl.DGLGraph(PAP)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(PRP)
    g = [g1, g2]

    features = torch.FloatTensor(feat_p)
    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = np.load(prefix + "/train_60.npy")
    val_idx = np.load(prefix + "/val_60.npy")
    test_idx = np.load(prefix + "/test_60.npy")

    return g, features, labels, num_classes, train_idx, val_idx, test_idx