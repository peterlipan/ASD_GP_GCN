import argparse
import os
import numpy as np
import pandas as pd
import torch
from GP_model import Model
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--pooling_ratio', type=float, default=0.20, help='pooling ratio')
fold_args = parser.parse_args()

check_dir = './checkpoints_pool{:.2f}'.format(fold_args.pooling_ratio)


for i in range(10):
    fold_dir = os.path.join(check_dir, 'fold_%d' % (i+1))
    files = os.listdir(fold_dir)
    fold_acc = []
    max_acc = 0
    max_file = None
        
    for f in files:
        if f.endswith('.pth') and f.startswith('num_'):
            acc = float(f.split('_')[3])
            if acc > max_acc:
                max_acc = acc
                max_file = f

    print('extracting information from {}'.format(fold_dir+'/' + max_file))

    checkpoint = torch.load(os.path.join(fold_dir, max_file))

    args = checkpoint['args']
    model = Model(args).to('cuda:0')
    args.batch_size = 256

    model.load_state_dict(checkpoint['net'])

    dataset = TUDataset('data', name=args.dataset, use_node_attr=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    feature_matrix = []
    correct = 0
    for data in data_loader:
        data = data.to(args.device)
        out, features = model(data)
        feature_matrix += features.cpu().detach().numpy().tolist()
        pred = (out > 0).long()
        correct += pred.eq(data.y).sum().item()

    fold_acc.append(correct*100/871)
    fold_feature_matrix = np.array(feature_matrix)

    print('Overall accuracy: {:.6f} on fold {:d}'.format(sum(fold_acc)/10, i+1))

    features = pd.DataFrame(fold_feature_matrix)
    graph_label = pd.read_csv('./data/ABIDE/raw/ABIDE_graph_labels.txt', header=None)
    graph_id = np.arange(1, 872)
    features['label'] = graph_label
    features.insert(0, 'id', graph_id)
    features.to_csv(os.path.join(fold_dir, 'ABIDE.content'), header=False, index=False, sep='\t')

print('Done!')
print('information saved to ABIDE.content')
