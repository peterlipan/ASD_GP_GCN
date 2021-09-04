import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def build_edges(data_dir):
    cluster_att = ['SEX', 'SITE_ID']
    # get text information: sex, site
    logs = pd.read_csv(os.path.join(data_dir, 'log.csv'))
    text_info = logs[cluster_att].values
    enc = OneHotEncoder()
    enc.fit(text_info)
    text_feature = enc.transform(text_info).toarray()

    # take ages into consideration
    ages = logs['AGE_AT_SCAN'].values
    # Normalization
    ages = (ages - min(ages)) / (max(ages) - min(ages))

    cluster_features = np.c_[text_feature, ages]

    adj = []
    att = []
    sim_matrix = cosine_similarity(cluster_features)
    for i in range(871):
        for j in range(871):
            if i != j:
                if sim_matrix[i, j] > 0.5 and i > j:
                    adj.append([i, j])
                    att.append(sim_matrix[i, j])

    adj = np.array(adj).T
    att = np.array([att]).T

    pd.DataFrame(adj).to_csv(os.path.join(data_dir, 'ABIDE.adj'), index=False, header=False)
    pd.DataFrame(att).to_csv(os.path.join(data_dir, 'edge.att'), index=False, header=False)

    return adj, att
