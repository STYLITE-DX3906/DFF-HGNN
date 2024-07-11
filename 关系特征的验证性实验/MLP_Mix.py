import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, hidden_dim, outsize, feat_drop, input_dim_a, input_dim_s):
        super(MLP, self).__init__()

        self.hidden_dim = hidden_dim

        # dropout
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        input_dim = input_dim_a + input_dim_s

        self.predict2 = nn.Linear(input_dim, hidden_dim) #第一层神经网络
        self.predict3 = nn.Linear(hidden_dim, hidden_dim) #第二层神经网络

        self.activation = nn.ELU()  # 添加非线性激活函数，这里选用ELU，也可以换成其他激活函数如ReLU、Tanh等

        self.predict = nn.Linear(hidden_dim, outsize)


    def forward(self, feat_a,feat_s):

        # 使用一层神经网络
        # h = self.predict2(feat)

        # 使用两层神经网络(非线性)
        # h = F.elu(self.predict3(self.predict2(feat)))

        # feat = self.alpha * feat_a + (1 - self.alpha) * feat_s
        # feat = torch.cat([feat_a, feat_s], dim=1)
        feat = torch.cat([feat_s,feat_a], dim=1)

        # h = self.predict3(self.predict2(feat))

        # 使用两层神经网络，并在每层线性变换后加入非线性激活函数
        h = self.activation(self.predict2(feat))
        h = self.activation(self.predict3(h))

        logits = self.predict(h)

        return logits, h



class MLP2(nn.Module):
    def __init__(self, hidden_dim, outsize, feat_drop, input_dim_a, input_dim_s):
        super(MLP2, self).__init__()

        self.hidden_dim = hidden_dim

        # dropout
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.alpha = nn.Parameter(torch.ones(1, hidden_dim) * 0.5, requires_grad=True)

        self.predict2 = nn.Linear(input_dim_a, hidden_dim) #第一层神经网络 属性特征转换
        self.predict4 = nn.Linear(input_dim_s, hidden_dim) #第一层神经网络 结构特征转换

        self.predict3 = nn.Linear(hidden_dim, hidden_dim) #第二层神经网络

        self.activation = nn.ELU()  # 添加非线性激活函数，这里选用ELU，也可以换成其他激活函数如ReLU、Tanh等

        self.predict = nn.Linear(hidden_dim, outsize)


    def forward(self, feat_a,feat_s):

        h_a = self.activation(self.predict2(feat_a))
        h_s = self.activation(self.predict4(feat_s))
        feat = self.alpha * h_a + (1 - self.alpha) * h_s
        print("属性特征所占权重：", self.alpha.data.cpu().numpy())
        print("结构特征所占权重：", (1-self.alpha).data.cpu().numpy())
        h = self.activation(self.predict3(feat))

        logits = self.predict(h)

        return logits, h