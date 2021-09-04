import os
import argparse
import pandas as pd
import numpy as np
import time
from utils import build_edges
import glob
import random
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data
from sklearn.manifold import TSNE
from GCN_model import GCN
from sklearn.model_selection import KFold

data_dir = './population_graph'

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--batch_size', type=int, default=872, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.01, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=10000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=3000, help='patience for early stopping')
parser.add_argument('--num_times', type=int, default=1, help='patience for early stopping')
parser.add_argument('--pooling_ratio', type=float, default=0.05, help='pooling ratio')

args = parser.parse_args()
args.num_classes = 2
args.num_features = 128
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


print(args)

adj_path = os.path.join(data_dir, 'ABIDE.adj')
att_path = os.path.join(data_dir, 'edge.att')

if not os.path.exists(adj_path) or not os.path.exists(att_path):
    edge_index, edge_att = build_edges(data_dir)
else:
    edge_index = pd.read_csv(adj_path, header=None).values
    edge_att = pd.read_csv(att_path, header=None).values.reshape(-1)


edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_att = torch.tensor(edge_att, dtype=torch.float)

criterion = nn.BCEWithLogitsLoss()


def visualize(out, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy().reshape(-1, 1))

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def train(dataloader, model, optimizer):
    min_loss = 1e10
    patience_cnt = 0
    loss_set = []
    acc_set = []
    best_epoch = 0
    num_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        num_epoch += 1
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, _ = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            pred = (out[data.train_mask] > 0).long()
            correct += pred.eq(data.y[data.train_mask]).sum().item()

        acc_train = correct / data.train_mask.sum().item()

        test_accu, test_losss, _ = compute_test(dataloader, model)

        print('\r', 'Epoch: {:06d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'acc_test: {:.6f}'.format(test_accu), 'test_loss: {:.6f}'.format(test_losss),
              'time: {:.6f}s'.format(time.time() - t), flush=True, end='')

        loss_set.append(loss_train)
        acc_set.append(acc_train)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if loss_set[-1] < min_loss:
            min_loss = loss_set[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('\nOptimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(dataloader, model):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    output = []
    for data in dataloader:
        data = data.to(args.device)
        out, _ = model(data.x, data.edge_index, data.edge_attr)
        output += out.cpu().detach().numpy().tolist()
        pred = (out[data.test_mask] > 0).long()
        correct += pred.eq(data.y[data.test_mask]).sum().item()
        loss_test += criterion(out[data.test_mask], data.y[data.test_mask].float()).item()
    return correct / data.test_mask.sum().item(), loss_test, output


if __name__ == '__main__':
    checkpoint_dir = './checkpoints_pool{:.2f}'.format(args.pooling_ratio)
    save_path = 'GCN_weights'
    indices = np.arange(871)

    kf = KFold(n_splits=10, shuffle=True, random_state=13)

    result_df = pd.DataFrame([])

    for times in range(args.num_times):
        print('%d out of %d times CV' % (times+1, args.num_times))
        test_result_acc = []
        test_result_loss = []
        for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
            fold_path = os.path.join(checkpoint_dir, 'fold_%d' % (i+1))
            print('Training on the %d fold' % (i+1))
            PL_model = GCN(args).to(args.device)
            PL_optimizer = torch.optim.Adam(PL_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            content_path = os.path.join(fold_path, 'ABIDE.content')
            content = pd.read_csv(content_path, header=None, sep='\t')
            x = content.iloc[:, 1:-1].values
            y = content.iloc[:, -1].values

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            abide_data = Data(x=x, edge_index=edge_index, edge_attr=edge_att, y=y)

            # form the mask from idx
            train_mask = np.zeros(871)
            test_mask = np.zeros(871)
            train_mask[train_idx] = 1
            test_mask[test_idx] = 1

            # set the mask for dataset
            abide_data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            abide_data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

            data_loader = DataLoader([abide_data], batch_size=args.batch_size)

            # Model training
            best_model = train(data_loader, PL_model, PL_optimizer)
            # Restore best model for test set
            PL_model.load_state_dict(torch.load('{}.pth'.format(best_model)))
            test_acc, test_loss, test_out = compute_test(data_loader, PL_model)
            result_df['time_%d_fold_%d' % (times+1, i+1)] = test_out
            result_df.to_csv('results.csv', index=False)
            test_result_acc.append(test_acc)
            test_result_loss.append(test_loss)

            print('{:0>2d} fold test set results, loss = {:.6f}, accuracy = {:.6f}'.format(i+1, test_loss, test_acc))

            state = {'net': PL_model.state_dict(), 'args': args}
            torch.save(state, os.path.join(save_path, 'fold_{:d}_test_{:.6f}_drop_{:.3f}_.pth'
                                           .format(i+1, test_acc, args.dropout_ratio)))
