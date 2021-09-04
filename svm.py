import argparse
import pandas as pd
import numpy as np
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('--pooling_ratio', type=float, default=0.20, help='pooling ratio')
fold_args = parser.parse_args()

kf = KFold(n_splits=10, random_state=13, shuffle=True)
indices = np.arange(871)
checkpoint_dir = './checkpoints_pool{:.2f}'.format(fold_args.pooling_ratio)
print('Pooling ratio = {:.2f}'.format(fold_args.pooling_ratio))
acc_set = []

for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
    fold_path = os.path.join(checkpoint_dir, 'fold_%d' % (i+1))
    feature_file = os.path.join(fold_path, 'ABIDE.content')
    print('Evaluate on the %d fold' % (i+1))
    features = pd.read_csv(feature_file, sep='\t',header=None)
    x = features.iloc[:,1:-1].values
    y = features.iloc[:, -1].values
    trainx = x[train_idx]
    trainy = y[train_idx]
    testx = x[test_idx]
    testy = y[test_idx]

    clf = LogisticRegression(max_iter=10000)
    clf.fit(trainx, trainy)
    test_acc = clf.score(testx, testy)
    acc_set.append(test_acc)

    print('Test Accuracy: %f' % test_acc)

'''    names = ['control', 'ASD']
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(clf, testx, testy, display_labels=names, cmap=plt.cm.Blues, normalize='true', ax=ax)
    plt.title('Confusion matrix')
    plt.show()
'''
print('\nAverage acc: %f' % (sum(acc_set)/len(acc_set)))
