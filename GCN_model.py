from torch_geometric.nn import GCNConv, APPNP, ClusterGCNConv, ChebConv
import torch.nn.functional as F
import torch


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.nhid = args.nhid
        self.drop_out = args.dropout_ratio
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = ClusterGCNConv(self.nhid, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        features = x
        x = torch.flatten(x)
        return x, features
