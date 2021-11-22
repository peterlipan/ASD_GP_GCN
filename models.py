import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layers import HGPSLPool
from torch_geometric.nn import GCNConv, APPNP, ClusterGCNConv, ChebConv


# Model of hierarchical graph pooling
class GPModel(torch.nn.Module):
    def __init__(self, args):
        super(GPModel, self).__init__()
        # parameters of hierarchical graph pooling
        self.args = args
        self.num_features = args.num_features
        self.pooling_ratio = args.pooling_ratio
        self.sample = True
        self.sparse = True
        self.sl = False
        self.lamb = 1.0

        # define the pooling layers
        self.pool1 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool3 = HGPSLPool(self.num_features, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # initialize edge weights
        edge_attr = None

        # hierarchical pooling
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fuse the above three pooling results
        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        # return the selected substructures
        return x


# Multilayer Perceptron
class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, args):
        super(MultilayerPerceptron, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        self.lin1 = torch.nn.Linear(self.num_features, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        # further learned features
        features = x
        # for training phase
        x = torch.flatten(self.lin3(x))
        return x, features


# Model of graph convolutional Networks run on population graph
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio

        # define the gcn layers. As stated in the paper,
        # herein, we have employed GCNConv and ClusterGCN
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = ClusterGCNConv(self.nhid, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        # store the learned node embeddings
        features = x
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.flatten(x)
        return x, features
