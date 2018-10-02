import numpy as np
import datasets, random
import sklearn.cluster

def get_clustered_data(batch_x, batch_y, k):
    cluster_algo = sklearn.cluster.KMeans(n_clusters=k)
    cluster_algo.fit(batch_y)
    labels = cluster_algo.labels_
    batch_size = len(batch_x)

    mapper = dict()
    
    for i in range(batch_size):
        if mapper.get(labels[i]) is None:
            mapper[labels[i]] = [], []
        mapper[labels[i]][0].append(batch_x[i])
        mapper[labels[i]][1].append(batch_y[i])

    for i in range(k):
        clustered_batch_x = np.mat(mapper[i][0])
        clustered_batch_y = np.mat(mapper[i][1])
        np.save('clustered_data/clustered-x%d.npy' % i, clustered_batch_x)
        np.save('clustered_data/clustered_y%d.npy' % i, clustered_batch_y)

# batch_x, batch_y = datapoints_x, datapoints_y = datasets.read_data()
# get_clustered_data(batch_x, batch_y)
