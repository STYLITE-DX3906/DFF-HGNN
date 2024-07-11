import numpy as np
import scipy
import pickle
import torch
import dgl


def load_ACM_data(prefix='D:/STUDY/others/data/ACM_processed'):

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    # PAP = scipy.sparse.load_npz(prefix + '/pap.npz')
    # PSP = scipy.sparse.load_npz(prefix + '/psp.npz')
    PAP = np.load(prefix + '/0/0-1-0.npy')
    PSP = np.load(prefix + '/0/0-2-0.npy')
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


def load_IMDB_data(prefix=r'E:\图神经网络\论文\HAN多版本\Dataset\IMDB'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#节点类型0的特征，4019行4000列

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    MAM = np.load(prefix + '/mam.npy')
    MAM = scipy.sparse.csr_matrix(MAM)
    MDM = np.load(prefix + '/mdm.npy')
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