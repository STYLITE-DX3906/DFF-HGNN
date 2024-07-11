# 验证RFE方法有效性的数据处理

import datetime
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path

from scipy import sparse
from scipy import io as sio
from itertools import product

import torch
import dgl

def set_random_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # pytorch-cuda


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask.byte()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# def load_acm_raw(train_split=0.2, val_split=0.3, feat=1):
#     data_folder = './data/acm/'
#     data_path = osp.join(data_folder, 'ACM.mat')
#     data = sio.loadmat(data_path)
#     target = 'paper'
#
#     p_vs_l = data['PvsL']
#     p_vs_a = data['PvsA']
#     p_vs_t = data['PvsT']
#     p_vs_p = data['PvsP']
#     p_vs_c = data['PvsC']
#
#     conf_ids = [0, 1, 9, 10, 13]
#     label_ids = [0, 1, 2, 2, 1]
#
#     p_vs_c_filter = p_vs_c[:, conf_ids]
#     p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
#     p_vs_c = p_vs_c[p_selected]
#     # p_vs_t = p_vs_t[p_selected]
#     # p_vs_a = p_vs_a[p_selected]
#     # p_vs_t = p_vs_t[p_selected]
#     # p_vs_p = p_vs_p[p_selected]
#     p_vs_p = p_vs_p[p_selected].T[p_selected]
#     a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
#     p_vs_a = p_vs_a[p_selected].T[a_selected].T
#     l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
#     p_vs_l = p_vs_l[p_selected].T[l_selected].T
#     t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
#     p_vs_t = p_vs_t[p_selected].T[t_selected].T
#
#     if feat == 1 or feat == 3:
#         hg = dgl.heterograph({
#             ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
#             ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
#             ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
#             ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
#             ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
#             ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
#         })
#
#         paper_feats = torch.FloatTensor(p_vs_t.toarray())
#         features = {
#             'paper': paper_feats
#         }
#     elif feat == 2:
#         hg = dgl.heterograph({
#             ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
#             ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
#             ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
#             ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
#             ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
#             ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
#             ('paper', 'paper_term', 'term'): p_vs_t.nonzero(),
#             ('term', 'term_paper', 'paper'): p_vs_t.transpose().nonzero()
#         })
#         features = {}
#     elif feat == 4:
#         hg = dgl.heterograph({
#             ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
#             ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
#             ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
#             ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
#             ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
#             ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
#         })
#
#         paper_feats = torch.FloatTensor(p_vs_t.toarray())
#         features = {}
#     print(hg)
#
#     pc_p, pc_c = p_vs_c.nonzero()
#     labels = np.zeros(len(p_selected), dtype=np.int64)
#     for conf_id, label_id in zip(conf_ids, label_ids):
#         labels[pc_p[pc_c == conf_id]] = label_id
#     labels = torch.LongTensor(labels)
#
#     num_classes = 3
#
#     split = np.load(osp.join(data_folder, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
#     train_idx = split['train_idx']
#     val_idx = split['val_idx']
#     test_idx = split['test_idx']
#
#     num_nodes = hg.number_of_nodes('paper')
#     train_mask = get_binary_mask(num_nodes, train_idx)
#     val_mask = get_binary_mask(num_nodes, val_idx)
#     test_mask = get_binary_mask(num_nodes, test_idx)
#
#     node_dict = {}
#     edge_dict = {}
#     for ntype in hg.ntypes:
#         node_dict[ntype] = len(node_dict)
#     for etype in hg.etypes:
#         edge_dict[etype] = len(edge_dict)
#         hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
#
#     return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



def load_acm_precessed(feat=1):
    raw_dir = './data/acm/'

    node_types = ['paper', 'author', 'subject']
    paper_feats = sparse.load_npz(osp.join(raw_dir, 'features_0_p.npz'))
    author_feats = sparse.load_npz(osp.join(raw_dir, 'features_1_a.npz'))
    subject_feats = sparse.load_npz(osp.join(raw_dir, 'features_2_s.npz'))
    target = 'paper'

    if feat == 1:
        paper_feats = torch.from_numpy(paper_feats.todense()).to(torch.float)
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        subject_feats = torch.from_numpy(subject_feats.todense()).to(torch.float)

        # 结构特征提取
        adjM = sparse.load_npz(raw_dir + '/adjM.npz').toarray()
        # 提取 P 类型节点的行
        P_nodes_rows = adjM[:4019, :]
        # 提取 A 类型节点的行
        A_nodes_rows = adjM[4019:4019 + 7167, :]
        # 提取 S 类型节点的行
        S_nodes_rows = adjM[4019 + 7167:4019 + 7167 + 60, :]
        P_nodes_rows = torch.FloatTensor(P_nodes_rows)
        A_nodes_rows = torch.FloatTensor(A_nodes_rows)
        S_nodes_rows = torch.FloatTensor(S_nodes_rows)
        paper_feats = torch.cat([P_nodes_rows, paper_feats], dim=1)
        author_feats = torch.cat([A_nodes_rows, author_feats], dim=1)
        subject_feats = torch.cat([S_nodes_rows, subject_feats], dim=1)

        features = {
            'paper': paper_feats,
            'author': author_feats,
            'subject': subject_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        paper_feats = torch.from_numpy(paper_feats.todense()).to(torch.float)
        features = {
            'paper': paper_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    ###################################################构建DGL异质图
    # 初始化节点分区
    s = {}
    N_p = paper_feats.shape[0]
    N_a = author_feats.shape[0]
    N_s = subject_feats.shape[0]
    s['paper'] = (0, N_p)
    s['author'] = (N_p, N_p + N_a)
    s['subject'] = (N_p + N_a, N_p + N_a + N_s)

    # 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

    # 构建异质图
    hg_data = dict()  # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]  # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0:  # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[
                src, src + '_' + dst, dst] = A_sub.nonzero()  # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['paper', 'movie_keyword', 'keyword'] = paper_feats.nonzero()
        hg_data['keyword', 'keyword_movie', 'paper'] = paper_feats.transpose().nonzero()

    hg = dgl.heterograph(hg_data)
    print(hg)
    ###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    split = np.load(osp.join(raw_dir, f'train_val_test_idx.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

    # 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



# def load_dblp_precessed(train_split=0.2, val_split=0.3, feat=1):
def load_dblp_precessed(feat=1):

    raw_dir = './data/dblp/'

    author_feats = sparse.load_npz(osp.join(raw_dir, 'features_0.npz')) # author to keyword
    paper_feats = sparse.load_npz(osp.join(raw_dir, 'features_1.npz')) # paper to words in title
    term_feats = np.load(osp.join(raw_dir, 'features_2.npy'))
    node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
    target = 'author'

    if feat == 1:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        paper_feats = torch.from_numpy(paper_feats.todense()).to(torch.float)
        term_feats = torch.from_numpy(term_feats).to(torch.float)

        # 尝试使用ONE-HOT编码设置conference的特征 效果不好 可能就是原文不进行此操作的原因
        # conference_feats = np.eye(20)
        # conference_feats = torch.from_numpy(conference_feats).to(torch.float)

        features = {
            'author': author_feats,
            'paper': paper_feats,
            'term': term_feats
            # 'conference': conference_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        features = {
            'author': author_feats,
        }
    node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
    num_confs = int((node_type_idx == 3).sum())

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    s = {}
    N_a = author_feats.shape[0]
    N_p = paper_feats.shape[0]
    N_t = term_feats.shape[0]
    N_c = num_confs
    s['author'] = (0, N_a)
    s['paper'] = (N_a, N_a + N_p)
    s['term'] = (N_a + N_p, N_a + N_p + N_t)
    s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

    node_types = ['author', 'paper', 'term', 'conference']

    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))
    hg_data = dict()

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]
        if A_sub.nnz > 0:
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero()

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['author', 'author_keyword', 'keyword'] = author_feats.nonzero()
        hg_data['keyword', 'keyword_author', 'author'] = author_feats.transpose().nonzero()
        hg_data['paper', 'paper_title', 'title-word'] = paper_feats.nonzero()
        hg_data['title-word', 'title_paper', 'paper'] = paper_feats.transpose().nonzero()
    hg = dgl.heterograph(hg_data)
    print(hg)

    split = np.load(osp.join(raw_dir, f'train_val_test_idx.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



# def load_imdb_precessed(train_split=0.2, val_split=0.3, feat=1):
def load_imdb_precessed(feat=1):
    raw_dir = './data/imdb/'

    node_types = ['movie', 'director', 'actor']
    movie_feats = sparse.load_npz(osp.join(raw_dir, 'features_0_M.npz'))
    director_feats = sparse.load_npz(osp.join(raw_dir, 'features_1_D.npz'))
    actor_feats = sparse.load_npz(osp.join(raw_dir, 'features_2_A.npz'))
    target = 'movie'

    if feat == 1:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        director_feats = torch.from_numpy(director_feats.todense()).to(torch.float)
        actor_feats = torch.from_numpy(actor_feats.todense()).to(torch.float)
        features = {
            'movie': movie_feats,
            'director': director_feats,
            'actor': actor_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        features = {
            'movie': movie_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

###################################################构建DGL异质图
# 初始化节点分区
    s = {}
    N_m = movie_feats.shape[0]
    N_d = director_feats.shape[0]
    N_a = actor_feats.shape[0]
    s['movie'] = (0, N_m)
    s['director'] = (N_m, N_m + N_d)
    s['actor'] = (N_m + N_d, N_m + N_d + N_a)

# 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

# 构建异质图
    hg_data = dict() # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]] # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0: # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero() # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['movie', 'movie_keyword', 'keyword'] = movie_feats.nonzero()
        hg_data['keyword', 'keyword_movie', 'movie'] = movie_feats.transpose().nonzero()

    hg = dgl.heterograph(hg_data)
    print(hg)
###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    split = np.load(osp.join(raw_dir, f'train_val_test_idx.npz'))
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

# 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



def load_yelp_precessed(feat=1):
    raw_dir = './data/yelp/'

    node_types = ['business', 'user', 'service', 'rating_levels']
    business_feats = sparse.load_npz(osp.join(raw_dir, 'features_0_b.npz'))
    user_feats = sparse.load_npz(osp.join(raw_dir, 'features_1_u.npz'))
    service_feats = sparse.load_npz(osp.join(raw_dir, 'features_2_s.npz'))
    rating_levels_feats = sparse.load_npz(osp.join(raw_dir, 'features_3_l.npz'))
    target = 'business'

    if feat == 1:
        business_feats = torch.from_numpy(business_feats.todense()).to(torch.float)
        user_feats = torch.from_numpy(user_feats.todense()).to(torch.float)
        service_feats = torch.from_numpy(service_feats.todense()).to(torch.float)
        rating_levels_feats = torch.from_numpy(rating_levels_feats.todense()).to(torch.float)

        features = {
            'business': business_feats,
            'user': user_feats,
            'service': service_feats,
            'rating_levels': rating_levels_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        business_feats = torch.from_numpy(business_feats.todense()).to(torch.float)
        features = {
            'business': business_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

###################################################构建DGL异质图
# 初始化节点分区
    s = {}
    N_b = business_feats.shape[0]
    N_u = user_feats.shape[0]
    N_s = service_feats.shape[0]
    N_l = rating_levels_feats.shape[0]


    s['business'] = (0, N_b)
    s['user'] = (N_b, N_b + N_u)
    s['service'] = (N_b + N_u, N_b + N_u + N_s)
    s['rating_levels'] = (N_b + N_u + N_s, N_b + N_u + N_s + N_l)


# 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

# 构建异质图
    hg_data = dict() # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]] # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0: # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero() # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    hg = dgl.heterograph(hg_data)
    print(hg)
###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    split = np.load(osp.join(raw_dir, f'train_val_test_idx.npy'), allow_pickle=True)
    split = split.item()
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    test_idx = split['test_idx']

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

# 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



def load_freebase_precessed(feat=1):
    raw_dir = './data/freebase/'

    node_types = ['movie', 'direct', 'actor', 'writer']

    movie_feats = torch.FloatTensor((sparse.eye(3492)).todense())
    direct_feats = torch.FloatTensor((sparse.eye(2502)).todense())
    actor_feats = torch.FloatTensor((sparse.eye(33401)).todense())
    writer_feats = torch.FloatTensor((sparse.eye(4459)).todense())

    # # 结构特征提取
    # adjM = sparse.load_npz(raw_dir + "/adjM.npz").toarray()
    # # 提取 M 类型节点的行
    # M_nodes_rows = adjM[:3492, :]
    # # 提取 D 类型节点的行
    # D_nodes_rows = adjM[3492:3492 + 2502, :]
    # # 提取 A 类型节点的行
    # A_nodes_rows = adjM[3492 + 2502:3492 + 2502 + 33401, :]
    # # 提取 W 类型节点的行
    # W_nodes_rows = adjM[3492 + 2502 + 33401:3492 + 2502 + 33401 + 4459, :]
    # M_nodes_rows = torch.FloatTensor(M_nodes_rows)
    # D_nodes_rows = torch.FloatTensor(D_nodes_rows)
    # A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    # W_nodes_rows = torch.FloatTensor(W_nodes_rows)
    #
    # # movie_feats = torch.cat([M_nodes_rows, movie_feats], dim=1)
    # # direct_feats = torch.cat([D_nodes_rows, direct_feats], dim=1)
    # # actor_feats = torch.cat([A_nodes_rows, actor_feats], dim=1)
    # # writer_feats = torch.cat([W_nodes_rows, writer_feats], dim=1)
    # movie_feats = M_nodes_rows
    # direct_feats = D_nodes_rows
    # actor_feats = A_nodes_rows
    # writer_feats = W_nodes_rows



    target = 'movie'

    if feat == 1:

        features = {
            'movie': movie_feats,
            'direct': direct_feats,
            'actor': actor_feats,
            'writer': writer_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        features = {
            'movie': movie_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

###################################################构建DGL异质图
# 初始化节点分区
    s = {}
    N_m = movie_feats.shape[0]
    N_d = direct_feats.shape[0]
    N_a = actor_feats.shape[0]
    N_w = writer_feats.shape[0]


    s['movie'] = (0, N_m)
    s['direct'] = (N_m, N_m + N_d)
    s['actor'] = (N_m + N_d, N_m + N_d + N_a)
    s['writer'] = (N_m + N_d + N_a, N_m + N_d + N_a + N_w)


# 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

# 构建异质图
    hg_data = dict() # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]] # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0: # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero() # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    hg = dgl.heterograph(hg_data)
    print(hg)
###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    train_idx = np.load(raw_dir + "/train_60.npy")
    val_idx = np.load(raw_dir + "/val_60.npy")
    test_idx = np.load(raw_dir + "/test_60.npy")

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

# 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target



def load_freebase_RFE_precessed(feat=1):
    raw_dir = './data/freebase/'

    node_types = ['movie', 'direct', 'actor', 'writer']

    # 不需要进行单独类别的one-hot编码的设置了，直接使用下面的原始邻接矩阵对角线元素加1！
    # movie_feats = torch.FloatTensor((sparse.eye(3492)).todense())
    # direct_feats = torch.FloatTensor((sparse.eye(2502)).todense())
    # actor_feats = torch.FloatTensor((sparse.eye(33401)).todense())
    # writer_feats = torch.FloatTensor((sparse.eye(4459)).todense())

    # 结构特征提取
    adjM = sparse.load_npz(raw_dir + "/adjM.npz").toarray()
    np.fill_diagonal(adjM, adjM.diagonal() + 1) # 直接在矩阵的对角线上加1 实现关系特征和身份编码的相加
    # 提取 M 类型节点的行
    M_nodes_rows = adjM[:3492, :]
    # 提取 D 类型节点的行
    D_nodes_rows = adjM[3492:3492 + 2502, :]
    # 提取 A 类型节点的行
    A_nodes_rows = adjM[3492 + 2502:3492 + 2502 + 33401, :]
    # 提取 W 类型节点的行
    W_nodes_rows = adjM[3492 + 2502 + 33401:3492 + 2502 + 33401 + 4459, :]
    M_nodes_rows = torch.FloatTensor(M_nodes_rows)
    D_nodes_rows = torch.FloatTensor(D_nodes_rows)
    A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    W_nodes_rows = torch.FloatTensor(W_nodes_rows)

    movie_feats = M_nodes_rows
    direct_feats = D_nodes_rows
    actor_feats = A_nodes_rows
    writer_feats = W_nodes_rows

    target = 'movie'

    if feat == 1:

        features = {
            'movie': movie_feats,
            'direct': direct_feats,
            'actor': actor_feats,
            'writer': writer_feats
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        features = {
            'movie': movie_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    ###################################################构建DGL异质图
    # 初始化节点分区
    s = {}
    N_m = movie_feats.shape[0]
    N_d = direct_feats.shape[0]
    N_a = actor_feats.shape[0]
    N_w = writer_feats.shape[0]

    s['movie'] = (0, N_m)
    s['direct'] = (N_m, N_m + N_d)
    s['actor'] = (N_m + N_d, N_m + N_d + N_a)
    s['writer'] = (N_m + N_d + N_a, N_m + N_d + N_a + N_w)

    # 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

    # 构建异质图
    hg_data = dict()  # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]  # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0:  # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[
                src, src + '_' + dst, dst] = A_sub.nonzero()  # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    hg = dgl.heterograph(hg_data)
    print(hg)
    ###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    train_idx = np.load(raw_dir + "/train_60.npy")
    val_idx = np.load(raw_dir + "/val_60.npy")
    test_idx = np.load(raw_dir + "/test_60.npy")

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

    # 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target


def load_aminer_RFE_precessed(feat=1):
    raw_dir = './data/aminer/'

    node_types = ['paper', 'author', 'reference']

    # paper_feats = torch.FloatTensor((sparse.eye(6564)).todense())
    # author_feats = torch.FloatTensor((sparse.eye(13329)).todense())
    # reference_feats = torch.FloatTensor((sparse.eye(35890)).todense())

    # 结构特征提取
    adjM = sparse.load_npz(raw_dir + "/adjM.npz").toarray()
    np.fill_diagonal(adjM, adjM.diagonal() + 1) # 直接在矩阵的对角线上加1 实现关系特征和身份编码的相加
    # 提取 P 类型节点的行
    P_nodes_rows = adjM[:6564, :]
    # 提取 A 类型节点的行
    A_nodes_rows = adjM[6564:6564 + 13329, :]
    # 提取 R 类型节点的行
    R_nodes_rows = adjM[6564 + 13329:6564 + 13329 + 35890, :]

    P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    R_nodes_rows = torch.FloatTensor(R_nodes_rows)

    # movie_feats = torch.cat([M_nodes_rows, movie_feats], dim=1)
    # direct_feats = torch.cat([D_nodes_rows, direct_feats], dim=1)
    # actor_feats = torch.cat([A_nodes_rows, actor_feats], dim=1)
    # paper_feats = paper_feats
    # author_feats = author_feats
    # reference_feats = reference_feats

    paper_feats = P_nodes_rows
    author_feats = A_nodes_rows
    reference_feats = R_nodes_rows



    target = 'paper'

    if feat == 1:

        features = {
            'paper': paper_feats,
            'author': author_feats,
            'reference': reference_feats,
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        features = {
            'paper': paper_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

###################################################构建DGL异质图
# 初始化节点分区
    s = {}
    N_p = paper_feats.shape[0]
    N_a= author_feats.shape[0]
    N_r = reference_feats.shape[0]


    s['paper'] = (0, N_p)
    s['author'] = (N_p, N_p + N_a)
    s['reference'] = (N_p + N_a, N_p + N_a + N_r)


# 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

# 构建异质图
    hg_data = dict() # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]] # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0: # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero() # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    hg = dgl.heterograph(hg_data)
    print(hg)
###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    train_idx = np.load(raw_dir + "/train_60.npy")
    val_idx = np.load(raw_dir + "/val_60.npy")
    test_idx = np.load(raw_dir + "/test_60.npy")

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

# 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target


def load_aminer_precessed(feat=1):
    raw_dir = './data/aminer/'

    node_types = ['paper', 'author', 'reference']

    paper_feats = torch.FloatTensor((sparse.eye(6564)).todense())
    author_feats = torch.FloatTensor((sparse.eye(13329)).todense())
    reference_feats = torch.FloatTensor((sparse.eye(35890)).todense())

    # # 结构特征提取
    # adjM = sparse.load_npz(raw_dir + "/adjM.npz").toarray()
    # # 提取 P 类型节点的行
    # P_nodes_rows = adjM[:6564, :]
    # # 提取 A 类型节点的行
    # A_nodes_rows = adjM[6564:6564 + 13329, :]
    # # 提取 R 类型节点的行
    # R_nodes_rows = adjM[6564 + 13329:6564 + 13329 + 35890, :]
    #
    # P_nodes_rows = torch.FloatTensor(P_nodes_rows)
    # A_nodes_rows = torch.FloatTensor(A_nodes_rows)
    # R_nodes_rows = torch.FloatTensor(R_nodes_rows)

    # movie_feats = torch.cat([M_nodes_rows, movie_feats], dim=1)
    # direct_feats = torch.cat([D_nodes_rows, direct_feats], dim=1)
    # actor_feats = torch.cat([A_nodes_rows, actor_feats], dim=1)
    # paper_feats = paper_feats
    # author_feats = author_feats
    # reference_feats = reference_feats
    # paper_feats = P_nodes_rows
    # author_feats = A_nodes_rows
    # reference_feats = R_nodes_rows



    target = 'paper'

    if feat == 1:

        features = {
            'paper': paper_feats,
            'author': author_feats,
            'reference': reference_feats,
        }
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        features = {
            'paper': paper_feats
        }

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

###################################################构建DGL异质图
# 初始化节点分区
    s = {}
    N_p = paper_feats.shape[0]
    N_a= author_feats.shape[0]
    N_r = reference_feats.shape[0]


    s['paper'] = (0, N_p)
    s['author'] = (N_p, N_p + N_a)
    s['reference'] = (N_p + N_a, N_p + N_a + N_r)


# 加载邻接矩阵
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))

# 构建异质图
    hg_data = dict() # 初始化空字典，用于存储异质图中各种类型的边。

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]] # s[src][0] 和 s[src][1] 是源节点类型在大邻接矩阵中的起始和结束索引
        if A_sub.nnz > 0: # 检查子矩阵 A_sub 是否包含非零项。nnz 属性返回稀疏矩阵中非零元素的数量。如果子矩阵中存在非零元素，表示存在 src 类型到 dst 类型的边。
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero() # 将非零元素的位置（即边的存在）添加到 hg_data 字典中。nonzero() 函数返回一个包含两个数组的元组，第一个数组是非零元素的行索引，第二个数组是非零元素的列索引，这些索引对应于图中的边。

    hg = dgl.heterograph(hg_data)
    print(hg)
###################################################

    # split = np.load(osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz'))
    train_idx = np.load(raw_dir + "/train_60.npy")
    val_idx = np.load(raw_dir + "/val_60.npy")
    test_idx = np.load(raw_dir + "/test_60.npy")

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]

# 节点 / 边映射
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target


# def load_data(dataset, train_split, val_split, feat=1):
#     if dataset == 'acm':
#         return load_acm_raw(train_split, val_split, feat=feat)
#     elif dataset == 'dblp':
#         return load_dblp_precessed(train_split, val_split, feat=feat)
#     elif dataset == 'imdb':
#         return load_imdb_precessed(train_split, val_split, feat=feat)
#     else:
#         return NotImplementedError('Unsupported dataset {}'.format(dataset))
def load_data(dataset, feat=1):
    if dataset == 'acm':
        # return load_acm_raw(feat=feat)
        return load_acm_precessed(feat=feat)
    elif dataset == 'dblp':
        return load_dblp_precessed(feat=feat)
    elif dataset == 'imdb':
        return load_imdb_precessed(feat=feat)
    elif dataset == 'yelp':
        return load_yelp_precessed(feat=feat)
    elif dataset == 'freebase':
        return load_freebase_precessed(feat=feat)
    elif dataset == 'freebase_RFE':
        return load_freebase_RFE_precessed(feat=feat)
    elif dataset == 'aminer':
        return load_aminer_precessed(feat=feat)
    elif dataset == 'aminer_RFE':
        return load_aminer_RFE_precessed(feat=feat)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))


def set_logger(my_str):
    task_time = get_date_postfix()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"log/{my_str}_{task_time}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_checkpoint_path(args, epoch):
    Path('checkpoint').mkdir(parents=True, exist_ok=True)
    checkpoint_path = './checkpoint/{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
        args.prefix,
        args.dataset,
        args.feat,
        args.train_split,
        args.seed,
        args.n_hid,
        args.n_layers,
        epoch,
        args.max_lr)
    return checkpoint_path