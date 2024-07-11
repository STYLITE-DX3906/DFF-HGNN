import argparse
import torch
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from aminer.aminer_dataloader import load_aminer
import numpy as np
import random

# from RAS_HGN import RAS_HGN
from MLP import MLP
# from MLP_2 import MLP


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def main(args):
    label, num_classes, train, val, test, feature, feature_dim, Relation_feature, Relation_feature_dim = load_aminer()

    # feature = feature.to(args['device'])
    Relation_feature = Relation_feature.to(args['device'])

    label = label.to(args['device'])


    svm_macro_avg = np.zeros((7,), dtype=float)
    svm_micro_avg = np.zeros((7,), dtype=float)
    nmi_avg = 0
    ari_avg = 0
    print('start train with repeat = {}\n'.format(args['repeat']))
    for cur_repeat in range(args['repeat']):
        print(
            'cur_repeat = {}   ==============================================================='.format(cur_repeat))

        model = MLP(args["hidden_units"], num_classes, args['feat_drop'], Relation_feature_dim)
        model = model.to(args['device'])

        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):

            model.train()
            logits, h = model(Relation_feature)
            loss = loss_fcn(logits[train], label[train])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits, h = model(Relation_feature)
            val_loss = loss_fcn(logits[val], label[val])
            test_loss = loss_fcn(logits[test], label[test])

            print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),
                                                                                        val_loss.item(),
                                                                                        test_loss.item()))
            early_stopping(val_loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        print('\ntesting...')
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
        model.eval()
        logits, h = model(Relation_feature)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test].detach().cpu().numpy(), label[test].cpu().numpy(), int(label.max()) + 1)  # 使用SVM评估节点

        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari
    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    nmi_avg /= args['repeat']
    ari_avg /= args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')

    # # # ======================#这个作用是可视化
    # labels = labels.cpu()
    # Y = labels[test_idx].numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
    # color_idx = {}
    # for i in range(len(h[test_idx].detach().cpu().numpy())):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    #
    # plt.legend()
    # plt.savefig("ACM_" + "分类图.png", dpi=1000,
    #             bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aminer')
    parser.add_argument('--dataset', default='Aminer_R', help='数据集')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0005, help='权重衰减')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--att_drop', default=0.6, help='注意力丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='特征丢弃率')
    parser.add_argument('--sample_rate', default=[27, 8], help='采样率')
    parser.add_argument('--nei_num', default=2, help='邻居数量')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=20, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=10, help='重复训练和测试次数')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

