"""
10-fold cross-validation of MLP and GCN
"""
import os
import time
import torch
import pandas as pd
import numpy as np
from training import train_mlp, test_mlp, train_gcn, test_gcn
from models import GPModel, MultilayerPerceptron, GCN
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, Data
from sklearn.linear_model import LogisticRegression


def kfold_mlp(data, args):
    """
    10-fold cross-validation of MultilayerPerceptron.
    Note that another 10-fold is implemented on the train set.
    Remember not access the test set at this stage. Some parameters are locally set here.
    :param data: pooling results.
    :param args: args from main.py
    :return: None. Best model for each fold is saved to args.check_dir/MLP/fold_%d
    """
    # locally set some parameters
    args.times = 1  # repeat times of the second level 10-fold cross-validation
    args.least = 50  # smallest number of training epochs; avoid under-fitting
    args.patience = 50  # patience for early stopping
    args.epochs = 1000  # maximum number of epochs
    args.weight_decay = 0.1

    indices = np.arange(data.shape[0])
    kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
    # as state in the paper, we run another 10-fold cross-validation
    val_kf = KFold(n_splits=10, shuffle=True)

    x = data[:, :-1]
    y = data[:, -1]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    args.num_features = x.shape[1]

    # repeat args.times times
    # the larger this parameter, the more stable the performance
    for repeat in range(args.times):
        print('%d times CV out of %d on training MLP...' % (repeat + 1, args.times))
        for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
            print('Training MLP on the %d fold...' % (i + 1))
            for count, (train_id, val_id) in enumerate(val_kf.split(train_idx)):
                if args.verbose:
                    print('%d val set out of 10' % (count + 1))

                # create fold dir
                fold_dir = os.path.join(args.check_dir, 'MLP', 'fold_%d' % (i + 1))
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                # Make sure the test indices are the same
                # for every time you run the K fold
                # it's silly but I must make sure
                test_log = os.path.join(fold_dir, 'test_indices.txt')
                if not os.path.exists(test_log):
                    np.savetxt(test_log, test_idx, fmt='%d', delimiter=' ')
                else:
                    saved_indices = np.loadtxt(test_log, dtype=int)
                    assert np.array_equal(test_idx, saved_indices), \
                        'Something goes wrong with 10-fold cross-validation'

                # set model
                model = MultilayerPerceptron(args).to(args.device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                # Set training and validation set
                # Do not use the test set
                train_set = Subset(dataset, train_id)
                val_set = Subset(dataset, val_id)

                # Build the data loader
                training_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
                validation_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

                # Model training
                best_model, best_val_acc, best_val_loss = train_mlp(model=model, train_loader=training_loader,
                                                                    val_loader=validation_loader, optimizer=opt,
                                                                    save_path=fold_dir, args=args)

                # Load the best model for validation set
                checkpoint = torch.load(os.path.join(fold_dir, '{}.pth'.format(best_model)))
                model.load_state_dict(checkpoint['net'])

                # Save the best model
                state = {'net': model.state_dict(), 'args': args}
                torch.save(state, os.path.join(fold_dir, 'num_{:d}_valacc_{:.6f}_pool_{:.3f}_epoch_{:d}_.pth'
                                               .format(count + 1, best_val_acc, args.pooling_ratio, best_model)))


def kfold_gcn(edge_index, edge_attr, num_samples, args):
    """
    Training phase of GCN. Some parameters of args are locally set here.
    No validation is implemented in this section.
    :param num_samples: number of samples
    :param edge_index: adjacency matrix of population graph.
    :param edge_attr: edge weights, say cosine similarity values
    :param args: args from main.py
    :return:
    """
    # locally set parameters
    args.num_features = args.nhid // 2  # output feature size of MLP
    args.nhid = args.num_features // 2
    args.epochs = 100000  # maximum number of training epochs
    args.patience = 30000  # patience for early stop regarding the performance on train set
    args.weight_decay = 0.001
    args.least = 50000  # least number of training epochs

    # load population graph
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    indices = np.arange(num_samples)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    # store the predictions
    result_df = pd.DataFrame([])
    test_result_acc = []
    test_result_loss = []

    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        # Ready to read further learned features extracted by MLP on different folds
        fold_path = os.path.join(args.data_dir, 'Further_Learned_Features', 'fold_%d' % (i + 1))
        # working path of training gcn
        work_path = os.path.join(args.check_dir, 'GCN')

        np.random.shuffle(train_idx)
        val_idx = train_idx[:len(train_idx)//10]
        train_idx = train_idx[len(train_idx)//10:]

        if not os.path.exists(work_path):
            os.makedirs(work_path)

        print('Training GCN on the %d fold' % (i + 1))
        model = GCN(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        feature_path = os.path.join(fold_path, 'features.txt')
        assert os.path.exists(feature_path), \
            'No further learned features found!'
        content = pd.read_csv(feature_path, header=None, sep='\t')
        x = content.iloc[:, :-1].values
        y = content.iloc[:, -1].values

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # form the mask from idx
        train_mask = np.zeros(num_samples)
        test_mask = np.zeros(num_samples)
        val_mask = np.zeros(num_samples)
        train_mask[train_idx] = 1
        test_mask[test_idx] = 1
        val_mask[val_idx] = 1

        # set the mask for dataset
        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=bool)

        loader = DataLoader([data], batch_size=args.batch_size)

        # Model training
        best_model = train_gcn(loader, model, optimizer, work_path, args)
        # Restore best model for test set
        checkpoint = torch.load(os.path.join(work_path, '{}.pth'.format(best_model)))
        model.load_state_dict(checkpoint['net'])
        test_acc, test_loss, test_out = test_gcn(loader, model, args)
        result_df['fold_%d_' % (i + 1)] = test_out
        test_result_acc.append(test_acc)
        test_result_loss.append(test_loss)
        print('GCN {:0>2d} fold test set results, loss = {:.6f}, accuracy = {:.6f}'.format(i + 1, test_loss, test_acc))

        state = {'net': model.state_dict(), 'args': args}
        torch.save(state, os.path.join(work_path, 'fold_{:d}_test_{:.6f}_drop_{:.3f}_epoch_{:d}_.pth'
                                       .format(i + 1, test_acc, args.dropout_ratio, best_model)))

    # save the predictions to args.result_dir/Graph Convolutional Networks/GCN_pool_%.3f_seed_%d_.csv
    result_path = os.path.join(args.result_dir, 'Graph Convolutional Networks')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_df.to_csv(os.path.join(result_path,
                                  'GCN_pool_%.3f_seed_%d_time_%d_.csv' % (args.pooling_ratio, args.seed, times+1)),
                     index=False, header=True)

    print('Mean Accuracy: %f' % (sum(test_result_acc)/len(test_result_acc)))


def logistic(num_samples, args):
    kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
    indices = np.arange(num_samples)
    checkpoint_dir = os.path.join(args.data_dir, 'Further_Learned_Features')
    acc_set = []

    # store the output of logistic regression
    result_df = pd.DataFrame([])

    time_length = []
    training_iters = []
    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        fold_path = os.path.join(checkpoint_dir, 'fold_%d' % (i+1))
        feature_file = os.path.join(fold_path, 'features.txt')
        indices_file = os.path.join(fold_path, 'test_indices.txt')
        saved_indices = np.loadtxt(indices_file, dtype=int)

        # Again, make sure you are using the same test set
        assert np.array_equal(test_idx, saved_indices), \
            'Something wrong with the 10-fold Cross Validation'

        if args.verbose:
            print('Evaluate on the %d fold' % (i+1))
        features = pd.read_csv(feature_file, sep='\t', header=None)
        x = features.iloc[:, :-1].values
        y = features.iloc[:, -1].values
        trainx = x[train_idx]
        trainy = y[train_idx]
        testx = x[test_idx]
        testy = y[test_idx]

        start = time.time()
        clf = LogisticRegression(max_iter=10000)
        clf.fit(trainx, trainy)
        end = time.time()
        time_length.append(end-start)
        training_iters.append(clf.n_iter_)
        test_acc = clf.score(testx, testy)
        acc_set.append(test_acc)
        result_df['fold_%d_' % (i + 1)] = clf.predict_proba(x)[:, 1]
        if args.verbose:
            print('Test Accuracy: %f' % test_acc)

    print('\nAverage acc: %f' % (sum(acc_set)/len(acc_set)))
    if args.verbose:
        print('Average Training Overhead (s): %f' % (sum(time_length)/len(time_length)))
        print('Average Training Iterations: %d' % (sum(training_iters)/len(training_iters)))
    result_path = os.path.join(args.result_dir, 'Logistic Regression')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_df.to_csv(os.path.join(result_path,
                                  'LR_pool_%.3f_seed_%d_.csv'%(args.pooling_ratio, args.seed)),
                     index=False, header=True)
    print('Logistic Classification results saved to %s' % result_path)
