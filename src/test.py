import os
import time
import math
import torch
import argparse
import numpy as np
from model import GCN
import matplotlib.pyplot as plt
from function import build_graph, Graph_embedding, load_data_dis, train, test
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = "cuda"
# device = 'cpu'

parser = argparse.ArgumentParser(description='Graph_Conv')

parser.add_argument('--init_lr', default=5e-3, type=float, help='inital learning rate')
parser.add_argument('--gamma', type=float, default=0.97, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--epochs', default=200, type=int, help='training epochs')
parser.add_argument('--batchsize', default=2, type=int, help='train batch size')
parser.add_argument('--data_root', default='./inverse_dataset/Example', type=str, help='data root')
parser.add_argument('--str_root', default='./structure/1600/thershold/150', type=str, help='struc mat root')
parser.add_argument('--rate', default=0.8, type=float, help='rate of training set')
parser.add_argument('--embed_size', default=4, type=int, help='Node embedding size')
parser.add_argument('--hidden_feat', default=256, type=int, help='features of hidden layer')
parser.add_argument('--num_hid_layers', default=64, type=int, help='num of hidden layers')
args = parser.parse_args()

n_out = 400
# 定义测点与预测点的坐标信息
x_point = np.array([[50, 160], [20, 100], [160, 160], [160, 40], [30, 42], [70, 42], [105, 60], [160, 100], [110, 162], [60, 100]], dtype=int)

# x_point = np.array([[0.019, 0.092], [0.093, 0.079], [0.045, 0.015], [0.08, 0.025], [0.072, 0.089],
#                          [0.036, 0.033], [0.021, 0.066], [0.047, 0.079], [0.06, 0.055], [0.022, 0.014]])
# x_point = np.array(2000 * x_point, dtype=int)
num = int(math.sqrt(n_out))
y_point = np.zeros((num, num, 2))
temp = np.arange(0, 200, int(200 / num))
ones = np.ones_like(temp)
for i in range(num):
    y_point[i, :, 0] = ones * i * int(200 / num)
    y_point[i, :, 1] = temp
y_point = np.array(y_point.reshape(-1, 2), dtype=int)


print("load data")
trainset, testset, point, x_std = load_data_dis(x_point, y_point, 10000, args.rate, 14)

# 定义dataloader
_, test_set = TensorDataset(trainset[0], trainset[1]), TensorDataset(testset[0], testset[1])

test_dataloader = DataLoader(dataset=test_set, batch_size=2, shuffle=False)

# Graph建模
adj, graph_edge = build_graph(point, 150)
adj = adj.to(device)

# Node2Vec
struc = Graph_embedding(args.str_root, args.embed_size, len(point), graph_edge)
struc = struc.to(device)

# Model and optimizer
# for i in range(5):
model = GCN(infeat=1, n_in=10, outfeat=1, n_out=n_out, str_feat=args.embed_size, 
            hid_feat=args.hidden_feat, n_hid=64, device=device)

Loss = torch.nn.L1Loss()
# model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('./model_pkl/thershold/net_hidf256_nhid64_bs64_150_test.pth'))
model.eval()

with torch.no_grad():
    test_loss = 0
    for i, (x, y) in enumerate(test_dataloader):
        # if x.size()[0] == 1:
        #     print('yes')
        y_hat = model(x.to(device), struc.to(device), adj.to(device))
        loss = Loss(y_hat, y.to(device).squeeze(-1)) * x_std
        if loss > 10:
            print(loss, i)
        test_loss += loss.item()
    test_loss = test_loss / len(test_dataloader)

print(test_loss)
