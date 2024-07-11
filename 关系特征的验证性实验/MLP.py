import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, hidden_dim, outsize, feat_drop, input_dim):
        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim

        # dropout
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.predict2 = nn.Linear(input_dim, hidden_dim) #第一层神经网络
        self.predict3 = nn.Linear(hidden_dim, hidden_dim) #第二层神经网络

        self.activation = nn.ELU()  # 添加非线性激活函数，这里选用ELU，也可以换成其他激活函数如ReLU、Tanh等

        self.predict = nn.Linear(hidden_dim, outsize)


    def forward(self, feat):

        # 使用一层神经网络
        # h = self.predict2(feat)

        # 使用两层神经网络
        # h = F.elu(self.predict3(self.predict2(feat)))

        # h = self.predict3(self.predict2(feat))

        # 使用两层神经网络，并在每层线性变换后加入非线性激活函数
        h = self.activation(self.predict2(feat))
        h = self.activation(self.predict3(h))

        logits = self.predict(h)

        return logits, h
