import datasets, numpy as np
import nn_model, remote_message

batch_size = 4096 * 8

model = nn_model.NNModel()
datapoints = (x, y) = datasets.read_data()
remote_message.send('model initialized, start training...')
remote_message.send('current loss = %g' % model.evaluate_loss(x, y))

def get_train_set(train_count = 100):
    train_set = []
    for i in range(train_count):
        train_set.append(datasets.next_batch(x, y, batch_size))
    return train_set


for i in range(10000):
    train_set = get_train_set()
    for train_data in train_set:
        batch_x, batch_y = train_data
        neckpoint = np.argmax(np.mean(np.square(model.batch_predict(x) - y)))
        batch_x.append(x[neckpoint])
        batch_y.append(y[neckpoint])
        neckpoint = np.argmax(np.max(np.max(np.abs(model.batch_predict(x) - y))))
        batch_x.append(x[neckpoint])
        batch_y.append(y[neckpoint])
        model.train(10, np.mat(batch_x), np.mat(batch_y))
    # model.train(10, x, y)
    print('Extreme case:')
    print('Expect = ')
    print(y[np.argmax(np.max(np.abs(model.batch_predict(x) - y)))])
    print('Predict = ')
    print(model.batch_predict([x[np.argmax(np.max(np.abs(model.batch_predict(x) - y)))]]))
    remote_message.send('current loss = %g' % model.evaluate_loss(x, y))
    model.random_test(x, y, 3)
    model.store()

