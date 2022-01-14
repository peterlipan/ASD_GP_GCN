"""
Construct the graph representation of brain imaging and population graph
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def brain_graph(logs, atlas, path, data_folder):
    if not os.path.exists(path):
        os.makedirs(path)
    # the global mean is not included in ho_labels.csv
    atlas.loc[-1] = [3455, 'Background']
    print(atlas.shape)
    # label the regions as right/left/global mean
    label = []
    for e in atlas['area'].values:
        if e.startswith('Left'):
            label.append(0)
        elif e.startswith('Right'):
            label.append(1)
        else:
            label.append(-1)

    atlas['label'] = label
    atlas.sort_values('index', inplace=True)
    atlas = atlas.reset_index().drop('level_0', axis=1)

    ###################
    # Adjacent matrix #
    ###################
    print('Processing the adjacent matrix...')
    # now the index in [0, 110]
    adj = np.zeros([111, 111])
    not_right = [i for i in range(111) if atlas['label'][i] != 1]
    not_left = [i for i in range(111) if atlas['label'][i] != 0]
    not_gb = [i for i in range(111) if atlas['label'][i] != -1]

    # Build the bipartite brain graph
    for idx in range(111):
        if atlas['label'][idx] == 0:
            adj[idx, not_left] = 1
        elif atlas['label'][idx] == 1:
            adj[idx, not_right] = 1
        elif atlas['label'][idx] == -1:
            adj[idx, not_gb] = 1

    # now form the sparse adj matrix
    # node id:[1, 111*871]
    node_ids = np.array_split(np.arange(1, 111 * 871 + 1), 871)
    adj_matrix = []
    for i in range(871):
        node_id = node_ids[i]
        for j in range(111):
            for k in range(111):
                if adj[j, k]:
                    adj_matrix.append([node_id[j], node_id[k]])

    # save sparse adj matrix
    pd.DataFrame(adj_matrix).to_csv(os.path.join(path, 'ABIDE_A.txt'), index=False, header=False)
    print('Done!')

    ###################
    # Graph indicator #
    ###################
    print('processing the graph indicator...')
    indicator = np.repeat(np.arange(1, 872), 111)
    pd.DataFrame(indicator).to_csv(os.path.join(path, 'ABIDE_graph_indicator.txt'), index=False, header=False)
    print('Done!')

    ###################
    #   Graph labels  #
    ###################
    print('processing the graph labels...')
    graph_labels = logs[['label']]
    graph_labels.to_csv(os.path.join(path, 'ABIDE_graph_labels.txt'), index=False, header=False)
    print('Done!')

    ###################
    # Node Attributes #
    ###################
    print('processing the node attributes...')
    # follow the order in log.csv
    files = logs['file_name']
    node_att = pd.DataFrame([])
    for file in files:
        file_path = os.path.join(data_folder, file)
        # data collected from different site
        # may have different time length (rows in the data file)
        # Here I simply cut them off according to
        # the shortest one, 78.
        ho_rois = pd.read_csv(file_path, sep='\t').iloc[:78, :].T
        node_att = pd.concat([node_att, ho_rois])

    node_att.to_csv(os.path.join(path, 'ABIDE_node_attributes.txt'), index=False, header=False)

    print('The shape of node attributes is (%d, %d)' % node_att.shape)
    print('Done!')

    ###################
    #   Node labels   #
    ###################
    print('processing the node labels...')
    # Make sure all the downloaded files have the same column (brian regions) order
    cols = list(pd.read_csv(file_path, sep='\t').columns.values)
    for file in files:
        file_path = os.path.join(data_folder, file)
        temp_cols = list(pd.read_csv(file_path, sep='\t').columns.values)
        assert cols == temp_cols, 'Inconsistent order of brain regions in ABIDE pcp!'

    node_label = np.arange(111)
    node_labels = np.tile(node_label, 871)
    pd.DataFrame(node_labels).to_csv(os.path.join(path, 'ABIDE_node_labels.txt'), index=False, header=False)
    print('Done!')


def population_graph(args):
    """
    Build the population graph. The nodes are connected if their cosine similarity is above 0.5
    in terms of xhenotypic information: gender, site, age.
    :param args: args from main.py
    :return: adj, att: adjacency matrix and edge weights
    """
    # considering phenotypic information: gender, age and site
    cluster_att = ['SEX', 'SITE_ID']
    # get text information: sex, site
    logs = pd.read_csv(os.path.join(args.data_dir, 'phenotypic', 'log.csv'))
    text_info = logs[cluster_att].values
    enc = OneHotEncoder()
    enc.fit(text_info)
    text_feature = enc.transform(text_info).toarray()

    # take ages into consideration
    ages = logs['AGE_AT_SCAN'].values
    # Normalization
    ages = (ages - min(ages)) / (max(ages) - min(ages))

    cluster_features = text_feature

    adj = []
    att = []
    sim_matrix = cosine_similarity(cluster_features)
    for i in range(871):
        for j in range(871):
            if sim_matrix[i, j] > 0.5 and i > j:
                adj.append([i, j])
                att.append(sim_matrix[i, j])

    adj = np.array(adj).T
    att = np.array([att]).T

    if not os.path.exists(os.path.join(args.data_dir, 'population graph')):
        os.makedirs(os.path.join(args.data_dir, 'population graph'))

    pd.DataFrame(adj).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.adj'), index=False, header=False)
    pd.DataFrame(att).to_csv(os.path.join(args.data_dir, 'population graph', 'ABIDE.attr'), index=False, header=False)
