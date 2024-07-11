"""
ACM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACM')
    parser.add_argument('--lr', default=0.005, help='学习率')
    # parser.add_argument('--weight_decay', default=0.0005, help='权重衰减')
    parser.add_argument('--weight_decay', default=0.00005, help='权重衰减') #0.0005虽然没有0.00005效果好，但是也够用，这样能够保持统一。
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--att_drop', default=0.6, help='注意力丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='特征丢弃率')
    parser.add_argument('--sample_rate', default=[27,8], help='采样率')
    parser.add_argument('--nei_num', default=2, help='邻居数量')
    # parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=20, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

DBLP
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBLP')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0005, help='权重衰减')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--att_drop', default=0.6, help='注意力丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='特征丢弃率')
    parser.add_argument('--sample_rate', default=[168], help='采样率') #现在是168 曾经是21
    parser.add_argument('--nei_num', default=1, help='邻居数量')
    # parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=25, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

Yelp
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yelp')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0005, help='权重衰减')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--att_drop', default=0.6, help='注意力丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='特征丢弃率')
    parser.add_argument('--sample_rate', default=[6,14,1], help='采样率')
    parser.add_argument('--nei_num', default=3, help='邻居数量')
    # parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=25, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

IMDB
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMDB')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0005, help='权重衰减') # 0.01 分类高但是聚类低 反正 分类也比不过 不如让聚类碾压 而且模型还统一了权重衰减
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--att_drop', default=0.6, help='注意力丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='特征丢弃率')
    parser.add_argument('--sample_rate', default=[6,10], help='采样率')
    parser.add_argument('--nei_num', default=2, help='邻居数量')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=30, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

"""
