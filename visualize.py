import os
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from torch_geometric.data import Data, DataLoader
from models import GCN
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()

parser.add_argument('--roc', action='store_true', default=False, help='Visualize ROC curve and mean confusion matrix')
parser.add_argument('--embedding', action='store_true', default=False, help='Visualize the learned node embeddings')
parser.add_argument('--result_root', type=str, default='./results', help='Root for the results')
parser.add_argument('--model_root', type=str, default='./checkpoints', help='Root for the stored models')
parser.add_argument('--data_root', type=str, default='./data', help='Root for the data')
parser.add_argument('--seed', type=int, default=13, help='Random seed. To specify the test set for evaluation')
parser.add_argument('--pooling_ratio', type=int, default=0.005, help='pooling ratio.')
parser.add_argument('--classifier', type=str, default='lr', help='Name of the classifier, gcn or lr')
parser.add_argument('--group', type=str, default='gender', help='Phenotypic attribute to group subjects on')

args = parser.parse_args()


def draw_cv_roc_curve(cv, out, y, thre=0, title=''):
    """
    Draw a Cross Validated ROC Curve.
    Args:
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        out: Predictions
        y: Response Pandas Series
        thre: threshold
        title: title for the plot
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    cn_matrix = np.zeros((2, 2))
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(121)

    i = 0
    for train, test in cv.split(out, y):
        probas_ = out.iloc[test]
        preds = [int(item) for item in (probas_.iloc[:, i].values > thre)]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_.iloc[:, i])
        tprs.append(interp(mean_fpr, fpr, tpr))
        cn_matrix += confusion_matrix(y.iloc[test], preds)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        i += 1
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax1.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    plt.legend(loc="lower right")

    ax2 = plt.subplot(122)
    sum_of_rows = cn_matrix.sum(axis=1)
    normalized = cn_matrix / sum_of_rows[:, np.newaxis]
    disp = ConfusionMatrixDisplay(normalized, display_labels=['Control', 'ASD'])
    disp.plot(ax=ax2, cmap=plt.cm.Blues)
    ax2.set_title('Confusion Matrix')

    plt.suptitle(title)
    plt.show()


def view2D(out, color, axis, size=70, maximum=11, legend_title='', title=''):
    """
    Visualize node embeddings on 2D. Each node(subject) is colored according to its group(i.e. Male VS female).
    :param out: node embeddings
    :param color: group of each node, i.e. sites, genders, age groups.
    :param size: size of scatter
    :param maximum: maximum number of groups.
    :param legend_title:
    :param title:
    :return:
    """
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    if torch.is_tensor(color):
        color = label.detach().cpu().numpy()

    z = TSNE(n_components=2).fit_transform(out)

    color_set = set(color)

    for i, item in enumerate(color_set):
        selected = color == item
        if i > maximum-1:
            break
        axis.scatter(z[selected, 0], z[selected, 1], s=size, color=cm.Set3(i), label=item)

    axis.set_xlabel('Dimension 1')
    axis.set_ylabel('Dimension 2')
    axis.legend(title=legend_title)
    axis.set_title(title)


def feature2embedding(model, feature, edge_index, edge_attr, args):
    """
    Convert further learned features to node embeddings with pre-trained GCN model.
    :param model: an instance of GCN model. Remember to load the trained weights.
    :param feature: further learned features.
    :param edge_index: adjacency matrix of population graph
    :param edge_attr: edges weights of population graph
    :param args: model args
    :return: embeddings: node embeddings
    """
    x = feature.iloc[:, :-1].values
    y = feature.iloc[:, -1].values

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    dataloader = DataLoader([data], batch_size=args.batch_size, shuffle=False)

    embeddings = []
    for i, data in enumerate(dataloader):
        data = data.to(args.device)
        _, embedding = model(data.x, data.edge_index, data.edge_attr)
        embeddings += embedding.cpu().detach().numpy().tolist()

    return np.array(embeddings)


if __name__ == '__main__':
    # plot roc curve and mean confusion matrix
    if args.roc:
        classifier = {'lr': 'Logistic Regression',
                      'gcn': 'Graph Convolutional Networks'}
        # The threshold for two classifiers are different
        # because we have used BCELossWithDigits in GCN
        thresholds = {'lr': 0.5, 'gcn': 0}
        result_path = os.path.join(args.result_root, classifier[args.classifier])
        assert os.path.exists(result_path), \
            'No classification result found'
        file_name = [f for f in os.listdir(result_path)
                     if float(f.split('_')[2]) == args.pooling_ratio and
                     int(f.split('_')[-2]) == args.seed]
        assert len(file_name), \
            'No result match the requirements: ' \
            'pooling ratio {:.3f}, random seed: {:d}'.format(args.pooling_ratio, args.seed)
        file_name = file_name[0]
        # load the predictions
        pred = pd.read_csv(os.path.join(result_path, file_name))
        # initialize k-fold
        kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
        # load ground truth
        labels = pd.read_csv(os.path.join(args.data_root, 'phenotypic', 'log.csv'))['label']
        # plot
        draw_cv_roc_curve(kf, pred, labels, thre=thresholds[args.classifier],
                          title='{:s}, pooling ratio = {:.3f}, random seed = {:d}'.
                          format(classifier[args.classifier], args.pooling_ratio, args.seed))

    # visualize node embeddings
    if args.embedding:
        # load pre-trained GCN model to output learned embeddings
        check_path = os.path.join(args.model_root, 'GCN')
        models = [f for f in os.listdir(check_path)
                  if f.startswith('fold') and f.endswith('.pth')]
        assert len(models),\
            'No trained GCN model found.'

        # load phenotypic information for grouping
        logs = pd.read_csv(os.path.join(args.data_root, 'phenotypic', 'log.csv'))

        # assign each node to groups
        if args.group == 'gender':
            sex = ['Female', 'Male']
            tags = np.array([sex[2 - i] for i in logs['SEX'].values])
        elif args.group == 'site':
            tags = logs['SITE_ID'].values
        elif args.group == 'age':
            # hard coding...
            sample_ages = []
            for i in range(871):
                if logs['AGE_AT_SCAN'].values[i] <= 12:
                    sample_ages.append('0 <= age <= 12')
                elif logs['AGE_AT_SCAN'].values[i] <= 17:
                    sample_ages.append('13 <= age <= 17')
                else:
                    sample_ages.append('18 <= age <= 58')
            tags = np.array(sample_ages)
        else:
            raise AttributeError('No such group available: %s'%args.group)

        # find the model with highest test acc from 10 folds
        test_acc = [float(f.split('_')[3]) for f in models]
        best_index = np.argmax(test_acc)
        fold_num = int(models[best_index].split('_')[1])
        model_file = os.path.join(check_path, models[best_index])
        checkpoint = torch.load(model_file)
        # load args
        model_args = checkpoint['args']
        # load weights
        gcn_model = GCN(model_args).to(model_args.device)
        gcn_model.load_state_dict(checkpoint['net'])

        # load further learned features on the same fold
        features = pd.read_csv(os.path.join(args.data_root, 'Further_Learned_Features',
                                            'fold_%d' % fold_num, 'features.txt'), header=None, sep='\t')
        # load population graph
        edge_idx = pd.read_csv(os.path.join(args.data_root, 'population graph', 'ABIDE.adj'), header=None).values
        edge_attr = pd.read_csv(os.path.join(args.data_root,
                                             'population graph', 'ABIDE.attr'), header=None).values.reshape(-1)

        # convert features to embeddings using pre-trained gcn model
        node_embedding = feature2embedding(gcn_model, features, edge_idx, edge_attr, model_args)

        plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        # plot 2D view of features
        view2D(features, tags, axis=ax1, legend_title=args.group, title='Further Learned Features')

        # plot 2D view of embeddings
        view2D(node_embedding, tags, axis=ax2, legend_title=args.group, title='Node Embeddings')

        plt.suptitle('Features VS Node Embeddings on %s' % args.group)
        plt.show()

