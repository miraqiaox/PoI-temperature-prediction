import networkx as nx
from Node2vec import Node2Vec
import numpy as np
import scipy.io as scio
import torch
import scipy.sparse as sp
from pathlib import Path




# adj_matrix norm
def normalize(mx):
    '''
    对邻接矩阵进行归一化
    '''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# build the graph
def build_graph(point, threshold):
    '''
    Graph的建模，目前采用距离阈值进行建模，输入节点的位置信息和阈值即可
    '''
    graph_edge = []
    edge = np.zeros((point.shape[0], point.shape[0]))
    for i in range(len(point)):
        for j in range(i + 1, len(point)):
            point_dis = np.linalg.norm(point[i] - point[j])
            if point_dis < threshold:
                graph_edge.append([i, j])
                graph_edge.append([j, i])
                edge[i, j] = 1
                edge[j, i] = 1
    edge_index = normalize(edge)
    edge_index = torch.tensor(edge_index, dtype=torch.float)

    return edge_index, graph_edge
# build the graph
def build_cov_graph(point, train_data, threshold):
    x_data = train_data
    graph_edge = []
    edge = np.zeros((x_data.shape[1], x_data.shape[1]))
    for i in range(x_data.shape[1]):
        for j in range(i + 1, x_data.shape[1]):
            cov = np.cov(x_data[:, i], x_data[:, j])
            if cov[0, 1] > threshold:
                graph_edge.append([i, j])
                graph_edge.append([j, i])
                edge[i, j] = 1
                edge[j, i] = 1
    # graph_edge = torch.LongTensor(graph_edge).T
    edge_index = normalize(edge)
    edge_index = torch.tensor(edge_index, dtype=torch.float)
    return edge_index, graph_edge
# Node2Vec
def Graph_embedding(root, embed_size, n_node, graph_edge):
    '''
    采用Node2Vec的形式将graph信息进行聚合得到每个节点的结构信息。
    root为将生成的structure feature进行保存的地址，如果存在则直接读取。
    '''
    my_file = Path(root  + '/struc.mat')
    if not my_file.exists():
        G = nx.Graph()
        G.add_edges_from(graph_edge)
        embedding_model = Node2Vec(G, walk_length=10, num_walks=1,
                                p=1, q=4, workers=1, use_rejection_sampling=0)
        embedding_model.train(embed_size=embed_size, window_size=5, iter=10)
        embeddings = embedding_model.get_embeddings()
        print(len(embeddings), n_node)
        x_struc = np.empty((n_node, embed_size))
        for i in range(n_node):
            x_struc[i, :] = embeddings[i]
        scio.savemat(root + '/struc.mat', {'struc':x_struc})
        x_struc = torch.tensor(x_struc).float()

        return x_struc
    else:
        data = scio.loadmat(root + '/struc.mat')
        struc = data['struc']
        return torch.tensor(struc).float()

# load data
def load_data(x_point, pred_point, num_data, rate, num, transform=True, device='cpu'):
    '''
    读取生成器生成的原始数据
    x_point为测点的坐标，pred_point为预测点的坐标，num_data为需要多少组
    '''
    if num < 5:
        root = './example_dataset_vb/Example'
    else:
        root = './example_dataset_vp/Example'
    x_tem = np.empty((num_data, len(x_point)))
    y_tem = np.empty((num_data, len(pred_point)))
    x_t = x_point.T
    y_t = pred_point.T
    for i in range(num_data):
        data = scio.loadmat(root + '{:d}.mat'.format(i))
        data_tem = data['u200'].T
        x_tem[i] = data_tem[x_t[0], x_t[1]]
        y_tem[i] = data_tem[y_t[0], y_t[1]]
    point = np.concatenate((x_point, pred_point), axis=0)
    if transform: 
        x_mean = np.mean(x_tem)
        x_std = np.std(x_tem)
        x_tem = (x_tem - x_mean) / x_std
        y_tem = (y_tem - x_mean) / x_std
        # x_tem = (x_tem - 298) / 50
        # y_tem = (y_tem - 298) / 50
    temp = int(rate * num_data)
    x = torch.tensor(x_tem).unsqueeze(-1).float().to(device)
    y = torch.tensor(y_tem).unsqueeze(-1).float().to(device)
    if num == 0 or num == 5:
        train_x = x[2000:]
        train_y = y[2000:]
        test_x = x[:2000]
        test_y = y[:2000]
    if num == 1 or num == 6:
        train_x = torch.cat((x[:2000], x[4000:]),dim=0)
        train_y = torch.cat((y[:2000], y[4000:]),dim=0)
        test_x = x[2000: 4000]
        test_y = y[2000: 4000]
    if num == 2 or num == 7:
        train_x = torch.cat((x[:4000], x[6000:]),dim=0)
        train_y = torch.cat((y[:4000], y[6000:]),dim=0)
        test_x = x[4000: 6000]
        test_y = y[4000: 6000]
    if num == 3 or num == 8:
        train_x = torch.cat((x[:6000], x[8000:]),dim=0)
        train_y = torch.cat((y[:6000], y[8000:]),dim=0)
        test_x = x[6000: 8000]
        test_y = y[6000: 8000]
    if num == 4 or num == 9:
        train_x = x[:8000]
        train_y = y[:8000]
        test_x = x[8000:]
        test_y = y[8000:]
    # train_x = x[:temp]
    # train_y = y[:temp]
    # test_x = x[temp:]
    # test_y = y[temp:]
    # train_x = torch.cat((x[:2000], x[4000:]),dim=0)
    # train_y = torch.cat((y[:2000], y[4000:]),dim=0)
    # test_x = x[2000: 4000]
    # test_y = y[2000: 4000]
    # train_x = x[:8000]
    # train_y = y[:8000]
    # test_x = x[8000:]
    # test_y = y[8000:]
    # train_x = x[2000:]
    # train_y = y[2000:]
    # test_x = x[:2000]
    # test_y = y[:2000]

    return (train_x, train_y), (test_x, test_y), point, x_std
def load_data_dis(x_point, pred_point, num_data, rate, num, transform=True, device='cpu'):
    '''
    读取生成器生成的原始数据
    x_point为测点的坐标，pred_point为预测点的坐标，num_data为需要多少组
    '''
    
    root = './example_dataset1/Example'
    x_tem = np.empty((num_data, len(x_point)))
    y_tem = np.empty((num_data, len(pred_point)))
    x_t = x_point.T
    y_t = pred_point.T
    for i in range(num_data):
        data = scio.loadmat(root + '{:d}.mat'.format(i))
        data_tem = data['u200']
        x_tem[i] = data_tem[x_t[0], x_t[1]]
        y_tem[i] = data_tem[y_t[0], y_t[1]]
    point = np.concatenate((x_point, pred_point), axis=0)
    if transform: 
        x_mean = np.mean(x_tem)
        x_std = np.std(x_tem)
        x_tem = (x_tem - x_mean) / x_std
        y_tem = (y_tem - x_mean) / x_std
        # x_tem = (x_tem - 298) / 50
        # y_tem = (y_tem - 298) / 50
    temp = int(rate * num_data)
    x = torch.tensor(x_tem).unsqueeze(-1).float().to(device)
    y = torch.tensor(y_tem).unsqueeze(-1).float().to(device)
    if num == 10:
        train_x = x[2000:]
        train_y = y[2000:]
        test_x = x[:2000]
        test_y = y[:2000]
    if num == 11:
        train_x = torch.cat((x[:2000], x[4000:]),dim=0)
        train_y = torch.cat((y[:2000], y[4000:]),dim=0)
        test_x = x[2000: 4000]
        test_y = y[2000: 4000]
    if num == 12:
        train_x = torch.cat((x[:4000], x[6000:]),dim=0)
        train_y = torch.cat((y[:4000], y[6000:]),dim=0)
        test_x = x[4000: 6000]
        test_y = y[4000: 6000]
    if num == 13:
        train_x = torch.cat((x[:6000], x[8000:]),dim=0)
        train_y = torch.cat((y[:6000], y[8000:]),dim=0)
        test_x = x[6000: 8000]
        test_y = y[6000: 8000]
    if num == 14:
        train_x = x[:8000]
        train_y = y[:8000]
        test_x = x[8000:]
        test_y = y[8000:]

    return (train_x, train_y), (test_x, test_y), point, x_std

def load_data_test(x_point, pred_point, num_data):

    # root = './example_dataset_vb/Example'

    # root = './example_dataset_vp/Example'
    
    root = './example_dataset1/Example'
    x_tem = np.empty((num_data, len(x_point)))
    y_tem = np.empty((num_data, len(pred_point)))
    x_t = x_point.T
    y_t = pred_point.T
    for i in range(num_data):
        data = scio.loadmat(root + '{:d}.mat'.format(i+10000))
        # data_tem = data['u200'].T
        data_tem = data['u200']
        x_tem[i] = data_tem[x_t[0], x_t[1]]
        y_tem[i] = data_tem[y_t[0], y_t[1]]
    point = np.concatenate((x_point, pred_point), axis=0)

    x_mean = np.mean(x_tem)
    x_std = np.std(x_tem)
    x_tem = (x_tem - x_mean) / x_std
    y_tem = (y_tem - x_mean) / x_std
        # x_tem = (x_tem - 298) / 50
        # y_tem = (y_tem - 298) / 50
    x = torch.tensor(x_tem).unsqueeze(-1).float()
    y = torch.tensor(y_tem).unsqueeze(-1).float()


    return (x, y), point, x_std

def train(model, struc, adj, Loss, optimizer, dataloader, device='cpu'):
    train_loss = 0
    # model.train()
    for _, (x, y) in enumerate(dataloader):
        y_hat = model(x.to(device), struc, adj)
        optimizer.zero_grad()
        loss = Loss(y_hat, y.to(device).squeeze(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader)

def test(model, struc, adj, Loss, dataloader, device='cpu'):
    with torch.no_grad():
        test_loss = 0
        for _, (x, y) in enumerate(dataloader):
            y_hat = model(x.to(device), struc, adj)
            loss = Loss(y_hat, y.to(device).squeeze(-1))
            test_loss += loss.item()
        test_loss = test_loss / len(dataloader)
        return test_loss

# test
if __name__ == '__main__':
    root = './dataset/Example'
    x_point = np.array([[0.019, 0.092], [0.093, 0.079], [0.045, 0.015], [0.08, 0.025], [0.072, 0.089],
                         [0.036, 0.033], [0.021, 0.066], [0.047, 0.079], [0.09, 0.055], [0.022, 0.014]])
    x_point = np.array(2000 * x_point, dtype=int)
    y_point = np.zeros((20, 20, 2))
    temp = np.arange(0, 200, 10)
    ones = np.ones_like(temp)
    for i in range(20):
        y_point[i, :, 0] = ones * i * 10
        y_point[i, :, 1] = temp
    y_point = np.array(y_point.reshape(-1, 2), dtype=int)
    x_tem, y_tem, point = load_data(root, x_point, y_point, num_data=1000)
    print(x_tem.shape, y_tem.shape, point.shape)
