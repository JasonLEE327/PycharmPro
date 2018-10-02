import regressionmodel
import datasets, translate_to_image
import numpy as np, random
import cluster_datasets

model = regressionmodel.NNModel()

clusters_count = 256

batch_x, batch_y = x, y = datasets.read_data()
batch_size = len(x)

batches_x = []
batches_y = []

#print('clustering...')
#cluster_datasets.get_clustered_data(x, y, clusters_count)

for i in range(clusters_count):
    batches_x.append(np.load('clustered_data/clustered-x%d.npy' % i))
    batches_y.append(np.load('clustered_data/clustered_y%d.npy' % i))

print('cluster process finished')
    
def generate_batch_list(batch_number = 10):
    batches = []
    for i in range(batch_number):
        batches.append(datasets.next_batch(x, y, int(.2 * batch_size)))
    return batches

def generate_clustered_batch(count_per_batch=200):
    batch_x, batch_y = [], []
    for i in range(clusters_count):
        current_batch_x = batches_x[i]
        current_batch_y = batches_y[i]
        for j in range(count_per_batch):
            r = random.randint(0, len(current_batch_x) - 1)
            batch_x.append(current_batch_x[r])
            batch_y.append(current_batch_y[r])
    return batch_x, batch_y

batch_train = True

for i in range(100000):
    if batch_train:
        batch_x, batch_y = generate_clustered_batch()
        model.train_step(50, batch_x, batch_y)
    else:
        model.train_step(1000, x, y)
    print('current loss = %g' % model.loss(x, y))
    translate_to_image.save_image('image/output-%d.bmp' % i, model.predict(datasets.read_input()))
    model.store()

'''
batch_x, batch_y = datasets.next_batch(x, y, 10)
for i in range(1000):
    model.train_step(1000, batch_x, batch_y)
    print('current loss = %g' % model.loss(batch_x, batch_y))
    print(np.mat(batch_x))
    print(np.mat(batch_y))
    print(model.predict(batch_x))
    
'''
