import os.path as osp
import shutil
from pathlib import Path
import argparse

import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.nn import functional as F

from model import SRHGN
from utils import set_random_seed, load_data, get_n_params, set_logger

from tools import evaluate_results_nc
import psutil
import os



def load_params():
    parser = argparse.ArgumentParser(description='Training SR-HGN')
    parser.add_argument('--prefix', type=str, default='SR-HGN')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='aminer_RFE')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--verbose', type=int, default=10)
    # parser.add_argument('--train_split', type=float, default=0.2) #原文数据集设置
    # parser.add_argument('--val_split', type=float, default=0.3) #原文数据集设置
    parser.add_argument('--max_lr', type=float, default=1e-3) # 源码设置

    parser.add_argument('--clip', type=int, default=1.0)
    # parser.add_argument('--weight_decay', type=float, default=1e-5) #原文设置
    parser.add_argument('--weight_decay', type=float, default=0.001) #作为对比实验设置
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_node_heads', type=int, default=4)
    parser.add_argument('--num_type_heads', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.4) # 原文设置

    parser.add_argument('--repeat', type=int, default=10, help='重复训练和测试次数')


    # parser.add_argument('--cluster', action='store_true') #原文数据集设置

    args = parser.parse_args()
    args = vars(args)

    return args


def init_feat(G, n_inp, features):
    # Randomly initialize features if features don't exist
    input_dims = {}

    for ntype in G.ntypes:
        
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_inp), requires_grad=True)
        nn.init.xavier_uniform_(emb)

        feats = features.get(ntype, emb)
        G.nodes[ntype].data['x'] = feats
        input_dims[ntype] = feats.shape[1]

    return G, input_dims


def train(model, G, labels, target, optimizer, scheduler, train_idx, clip=1.0):
    model.train()

    logits, _, _ = model(G, target)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()

    return loss.item()


def eval(model, G, labels, target, train_idx, val_idx, test_idx):
    model.eval()

    logits, _, _ = model(G, target)
    pred = logits.argmax(1).detach().cpu().numpy()

    train_macro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='macro')
    train_micro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='micro')
    val_macro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='macro')
    val_micro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='micro')
    test_macro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='macro')
    test_micro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='micro')

    return {
        'train_maf1': train_macro_f1,
        'train_mif1': train_micro_f1,
        'val_maf1': val_macro_f1,
        'val_mif1': val_micro_f1,
        'test_maf1': test_macro_f1,
        'test_mif1': test_micro_f1
    }


def cluster(model, G, target, labels):
    model.eval()

    _, embedding, attns = model(G, target)
    embedding = embedding.detach().cpu().numpy()
    labels = labels.cpu()

    kmeans = KMeans(n_clusters=len(torch.unique(labels)), random_state=42).fit(embedding)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)
    ari = adjusted_rand_score(labels, kmeans.labels_)

    return {
        'nmi': nmi,
        'ari': ari
    }


def main(params):
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else 'cpu')

    my_str = f"{params['prefix']}_{params['dataset']}"

    logger = set_logger(my_str)
    logger.info(params)

    checkpoints_path = f'checkpoints'
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)

    # G, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(params['dataset'], params['train_split'], params['val_split'], params['feat'])
    G, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(params['dataset'], params['feat'])

    G, input_dims = init_feat(G, params['input_dim'], features)
    G = G.to(device)
    labels = labels.to(device)

    svm_macro_avg = np.zeros((7,), dtype=float)
    svm_micro_avg = np.zeros((7,), dtype=float)
    nmi_avg = 0
    ari_avg = 0
    print('start train with repeat = {}\n'.format(params['repeat']))
    for cur_repeat in range(params['repeat']):
        print('cur_repeat = {}   ==============================================================='.format(params['repeat']))

        model = SRHGN(G,
                      node_dict, edge_dict,
                      input_dims=input_dims,
                      hidden_dim=params['hidden_dim'],
                      output_dim=labels.max().item() + 1,
                      num_layers=params['num_layers'],
                      num_node_heads=params['num_node_heads'],
                      num_type_heads=params['num_type_heads'],
                      alpha=params['alpha']).to(device)

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=params['epochs'], max_lr=params['max_lr'])

        logger.info('Training SR-HGN with #param: {:d}'.format(get_n_params(model)))

        best_val_mif1 = 0
        best_epoch = 0

        b = 0
        a = 0
        import time
        for epoch in range(1, params['epochs'] + 1):
            b = b + 1
            t = time.time()  # 计算开始的时间
            loss = train(model, G, labels, target, optimizer, scheduler, train_idx, clip=params['clip'])

            if epoch % params['verbose'] == 0:

                t2 = time.time()
                a = a + (t2 - t)
                print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

                results = eval(model, G, labels, target, train_idx, val_idx, test_idx)

                if results['val_mif1'] > best_val_mif1:
                    best_val_mif1 = results['val_mif1']
                    best_results = results
                    best_epoch = epoch

                logger.info('Epoch: {:d} | LR: {:.4f} | Loss {:.4f} | Val MiF1: {:.4f} (Best: {:.4f}) | Test MiF1: {:.4f} (Best: {:.4f})'.format(
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    results['val_mif1'],
                    best_results['val_mif1'],
                    results['test_mif1'],
                    best_results['test_mif1']
                ))

                torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}_{epoch}.pkl'))

        logger.info('Best Epoch: {:d} | Train MiF1: {:.4f},  MaF1: {:.4f} | Val MiF1: {:.4f}, MaF1: {:.4f} | Test MiF1: {:.4f}, MaF1: {:.4f}'.format(
            best_epoch,
            best_results['train_mif1'],
            best_results['train_maf1'],
            best_results['val_mif1'],
            best_results['val_maf1'],
            best_results['test_mif1'],
            best_results['test_maf1']
        ))

        # if params['cluster']:
        #     model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}_{best_epoch}.pkl')))
        #     cluster_results = cluster(model, G, target, labels)
        #
        #     logger.info('NMI: {:.4f} | ARI: {:.4f}'.format(cluster_results['nmi'], cluster_results['ari']))

        print('平均时间：', str(a / b))
        print('\ntesting...')
        model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}_{best_epoch}.pkl')))
        # cluster_results = cluster(model, G, target, labels)
        # logger.info('NMI: {:.4f} | ARI: {:.4f}'.format(cluster_results['nmi'], cluster_results['ari']))

        model.eval()
        logits, h, _ = model(G, target)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test_idx].detach().cpu().numpy(),labels[test_idx].cpu().numpy(),int(labels.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari

    svm_macro_avg = svm_macro_avg / params['repeat']
    svm_micro_avg = svm_micro_avg / params['repeat']
    nmi_avg /= params['repeat']
    ari_avg /= params['repeat']
    print('---\nThe average of {} results:'.format(params['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')



if __name__ == '__main__':
    params = load_params()
    set_random_seed(params['seed'])
    main(params)
