import os
import time
import math
import torch
import argparse
import numpy as np
from model import GCN
import matplotlib.pyplot as plt
from function import build_graph, Graph_embedding, load_data, train, test
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# device = 'cpu'

parser = argparse.ArgumentParser(description='Graph_Conv')

parser.add_argument('--init_lr', default=5e-3, type=float, help='inital learning rate')
parser.add_argument('--gamma', type=float, default=0.97, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--epochs', default=5, type=int, help='training epochs')
parser.add_argument('--batchsize', default=64, type=int, help='train batch size')
parser.add_argument('--data_root', default='./example_dataset1/Example', type=str, help='data root')
parser.add_argument('--str_root', default='./structure/400/center_monitoring_uniform_PoI', type=str, help='struc mat root')
parser.add_argument('--rate', default=0.8, type=float, help='rate of training set')
parser.add_argument('--embed_size', default=4, type=int, help='Node embedding size')
parser.add_argument('--hidden_feat', default=256, type=int, help='features of hidden layer')
# parser.add_argument('--num_hid_layers', default=64, type=int, help='num of hidden layers')
# parser.add_argument('--str_root', default='/mnt/liqiao/graph/graph/Node2vec/Node_Vec', type=str, help='data root')
args = parser.parse_args()


n_out = 1600
# 定义测点与预测点的坐标信息
x_point = np.array([[50, 160], [20, 100], [160, 160], [160, 40], [30, 42], [70, 42], [105, 60], [160, 100], [110, 162], [60, 100]], dtype=int)
# x_point = np.array(2000 * x_point, dtype=int)
num = int(math.sqrt(n_out))
y_point = np.zeros((num, num, 2))
temp = np.arange(0, 200, int(200 / num))
ones = np.ones_like(temp)
for i in range(num):
    y_point[i, :, 0] = ones * i * int(200 / num)
    y_point[i, :, 1] = temp

y_point = np.array(y_point.reshape(-1, 2), dtype=int)


# 读取数据
print("load data")
trainset, testset, point, x_std = load_data(args.data_root, x_point, y_point, 10, args.rate, 'cpu')

# 定义dataloader
train_set, test_set = TensorDataset(trainset[0], trainset[1]), TensorDataset(testset[0], testset[1])
train_dataloader = DataLoader(dataset=train_set,
                            batch_size=args.batchsize,
                            shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=args.batchsize, shuffle=False)

# Graph建模
adj, graph_edge = build_graph(point, 100)
print(len(graph_edge))


# Node2Vec
struc = Graph_embedding(args.str_root, args.embed_size, len(point), graph_edge)
struc = struc.to(device)
adj = adj.to(device)

# Model and optimizer

model = GCN(infeat=1, n_in=10, outfeat=1, n_out=n_out, str_feat=args.embed_size, 
            hid_feat=args.hidden_feat, n_hid=args.num_hid_layers, device=device)

Loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr)
scheduler = ExponentialLR(optimizer, args.gamma)

# training
total_time = 0
mae = 999
for epoch in range(args.epochs):
    time_start=time.time()
    model.train()
    train_loss= train(model, struc, adj, Loss, optimizer, train_dataloader, device)

    scheduler.step()
    time_end=time.time()
    epoch_time = time_end - time_start
    total_time += epoch_time
    if train_loss < mae:
        torch.save(model.state_dict(),
                    "./model_pkl/net_hidf{:d}_nhid{:d}_bs{:d}_{:d}_test.pth".format(
                    args.hidden_feat, args.num_hid_layers, args.batchsize, n_out))
        mae = train_loss
    if (epoch + 1) % 10  == 0:
        with torch.no_grad():
            model.eval()
            test_loss = test(model, struc, adj, Loss, test_dataloader, device) * x_std
        
        print('epoch: {:d}, train loss: {:4f}, test loss: {:4f}'.format(epoch + 1, train_loss  * x_std, test_loss))
    else:
        print('epoch: {:d}, train loss: {:4f}'.format(epoch + 1, train_loss  * x_std))
    print('time per epoch:{:4f}s'.format(epoch_time))
    print('total time:{:4f}s'.format(total_time))
