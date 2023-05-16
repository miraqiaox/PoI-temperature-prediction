import os
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCNlayer(Module):
    """
    GCN layer
    """

    def __init__(self, in_features, out_features, nodes, device='cpu', bias=True):
        super(GCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = Parameter(torch.FloatTensor(nodes, out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Res_Block(nn.Module):
    def __init__(self, nhid, nodes, device='cpu'):
        super(Res_Block, self).__init__()
        self.gc = GCNlayer(nhid, nhid, nodes, device)
        self.bn = nn.BatchNorm1d(nodes, track_running_stats=False).to(device)
        self.relu = nn.LeakyReLU().to(device)
        self.device = device
    def forward(self, X):
        x, adj = X
        # x_temp = torch.ones_like(x).to(self.device)
        # for i in range(x.size()[0]):
        x_temp = self.gc(x, adj)
        x_temp = self.relu(self.bn(x_temp)) + x
        return (x_temp, adj)


class GCN(nn.Module):
    def __init__(self, infeat, n_in, outfeat, n_out, str_feat, hid_feat, n_hid, device='cpu'):
        super(GCN, self).__init__() 
        self.n_in = n_in
        self.bn = nn.BatchNorm1d(n_in + n_out, track_running_stats=False).to(device)
        self.relu = nn.LeakyReLU().to(device)
        self.gc_input = GCNlayer(infeat + str_feat, hid_feat, n_in + n_out, device=device)
        self.gc_output = GCNlayer(hid_feat, outfeat, n_in + n_out, device=device)
        # self.linear = nn.Linear(n_in, n_in + n_out).to(device)
        self.conv1d = nn.Conv1d(n_in, n_in + n_out, kernel_size=1).to(device)
        self.hidden_layer = nn.Sequential()
        for i in range(n_hid):
            self.hidden_layer.add_module(
                'hidden_layer_{:d}'.format(i + 1), Res_Block(hid_feat, n_in + n_out, device))

    def forward(self, x, x_struc, adj):
        # x = x.permute(0,2,1)
        x = self.conv1d(x)
        # x = x.permute(0,2,1)
        struc = x_struc.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = torch.cat((x, struc), dim=-1)
        temp_in = self.gc_input(x, adj)
        temp_in = self.relu(self.bn(temp_in))
        X = (temp_in, adj)
        X_hid = self.hidden_layer(X)
        x_out, adj = X_hid
        temp_out = self.gc_output(x_out, adj)
        temp = temp_out.squeeze(-1)
        return temp[:, self.n_in:]


if __name__ == '__main__':
    model = GCN(infeat=2, n_in=10, outfeat=1, n_out=20, 
            str_feat=4, hid_feat=256, n_hid=64)
    print(sum(param.numel() for param in model.parameters()))
    # adj = torch.rand(30, 30)
    # struc = torch.rand(30, 4)
    # x = torch.rand(4, 10, 2)
    # y = model(x, struc, adj)
    # print(y.size())
    # for i in model.state_dict():
    #     print(i)
    # print('----------------')
    # for name, param in model.named_parameters():
    #     print(name)