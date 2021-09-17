import argparse
import torch
import os
import pandas as pd
from models import GCN
from construct_graph import population_graph
from kfold_eval import kfold_mlp, kfold_gcn, logistic
from training import graph_pooling, extract
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--batch_size', type=int, default=872, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--nhid', type=int, default=256, help='hidden size of MLP')
parser.add_argument('--pooling_ratio', type=float, default=0.005, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.01, help='dropout ratio')
parser.add_argument('--data_dir', type=str, default='./data', help='root of all the datasets')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--check_dir', type=str, default='./checkpoints', help='root of saved models')
parser.add_argument('--result_dir', type=str, default='./results', help='root of classification results')
parser.add_argument('--verbose', type=bool, default=True, help='print training details')

args = parser.parse_args()
# Set random seed
torch.manual_seed(args.seed)

# check if exists downsampled brain imaging data
dowmsample_file = os.path.join(args.data_dir, 'ABIDE_downsample',
                               'ABIDE_pool_{:.3f}_.txt'.format(args.pooling_ratio))
if not os.path.exists(dowmsample_file):
    print('Running graph pooling with pooling ratio = {:.3f}'.format(args.pooling_ratio))
    graph_pooling(args)

# load sparse brain networking
downsample = pd.read_csv(dowmsample_file, header=None, sep='\t').values
kfold_mlp(downsample, args)

# use the best MLP model to extract further learned features
# from pooling results
extract(downsample, args)

# run Logistic Regression as the classifier
logistic(args)

# check if population graph is constructed
adj_path = os.path.join(args.data_dir, 'population graph', 'ABIDE.adj')
attr_path = os.path.join(args.data_dir, 'population graph', 'ABIDE.attr')

if not os.path.exists(adj_path) or not os.path.exists(attr_path):
    population_graph(args)

# Load population graph
edge_index = pd.read_csv(adj_path, header=None).values
edge_attr = pd.read_csv(attr_path, header=None).values.reshape(-1)

# run GCN
kfold_gcn(edge_index, edge_attr, args)