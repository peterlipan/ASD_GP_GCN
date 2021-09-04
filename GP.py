import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np
from GP_model import Model
from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--nhid', type=int, default=256, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.13, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.01, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='ABIDE', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
parser.add_argument('--times', type=int, default=1, help='repeat times of 10-fold CV')
parser.add_argument('--least', type=int, default=50, help='smallest number of training epochs')

args = parser.parse_args()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset('./data', name=args.dataset, use_node_attr=True)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)


def train(model, train_loader, val_loader, optimizer, save_path):
    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    val_loss_values = []
    val_acc_values = []
    best_epoch = 0
    epoch_num = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, _ = model(data)
            loss_func = nn.BCEWithLogitsLoss()
            loss = loss_func(out, data.y.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = (out > 0).long()
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(model, val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        val_acc_values.append(acc_val)
        epoch_num += 1
        state = {'net':model.state_dict(), 'args':args}
        torch.save(state, os.path.join(save_path, '{}.pth'.format(epoch)))
        if epoch_num < args.least:
           continue
        if val_loss_values[-1] <= min_loss:
            min_loss = val_loss_values[-1]
            max_acc = val_acc_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
        for f in files:
            if f.startswith('num') or f.startswith('test'):
                continue
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(os.path.join(save_path, f))

    files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
    for f in files:
        if f.startswith('num') or f.startswith('test'):
            continue
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(os.path.join(save_path, f))
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch, max_acc, min_loss


def compute_test(model, loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out, _ = model(data)
        pred = (out > 0).long()
        correct += pred.eq(data.y).sum().item()
        loss_func = nn.BCEWithLogitsLoss()
        loss_test += loss_func(out, data.y.float()).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    check_dir = './checkpoints'
    # K-fold 
    kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
    val_kf = KFold(n_splits=10, shuffle=True)
    indices = np.arange(871)
    for repeat in range(args.times):
        print('%d times CV out of %d...' % (repeat+1, args.times))
        for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
            print('Learning on the %d fold' % (i + 1))
            for count, (train_id, val_id) in enumerate(val_kf.split(train_idx)):
                print('%d val set out of 10' % (count + 1))
                fold_dir = os.path.join(check_dir, 'fold_%d' % (i+1))
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)
                # set model
                PL_model = Model(args).to(args.device)
                opt = torch.optim.Adam(PL_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                # form the corresponding dataset
                train_set = Subset(dataset, train_id)
                val_set = Subset(dataset, val_id)
                # Stay away from the test set
                # test_set = Subset(dataset, test_idx)

                # form the dataloader
                training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
                validation_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
                # testing_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

                # Model training
                best_model, best_val_acc, best_val_loss = train(model=PL_model, train_loader=training_loader, val_loader=validation_loader, optimizer=opt, save_path=fold_dir)

                # Restore best model for test set
                checkpoint = torch.load(os.path.join(fold_dir, '{}.pth'.format(best_model)))
                PL_model.load_state_dict(checkpoint['net'])

                # Stay away from the test set
                # test_acc, test_loss = compute_test(PL_model, testing_loader)

                state = {'net': PL_model.state_dict(), 'args': args}
                torch.save(state, os.path.join(fold_dir, 'num_{:d}_valacc_{:.6f}_pool_{:.3f}_epoch_{:d}_.pth'
                                               .format(count+1, best_val_acc, args.pooling_ratio, best_model)))

