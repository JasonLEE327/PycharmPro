import numpy as np
import tensorflow as tf

import datasets, random
import remote_message

class NNModel:
    
    INPUT_X = 10
    INPUT_Y = 3
    
    LAYER1_UNITS = 32
    LAYER2_UNITS = 64
    LAYER3_UNITS = 32

    SAVER_MODEL_NAME = 'NN-Model-(%d-%d-%d)' % (LAYER1_UNITS, LAYER2_UNITS, LAYER3_UNITS)

    w1 = tf.Variable(tf.random_normal([INPUT_X, LAYER1_UNITS]))
    w2 = tf.Variable(tf.random_normal([LAYER1_UNITS, LAYER2_UNITS]))
    w3 = tf.Variable(tf.random_normal([LAYER2_UNITS, LAYER3_UNITS]))

    w4 = tf.Variable(tf.random_normal([LAYER3_UNITS, INPUT_Y]))

    b1 = tf.Variable(tf.random_normal([LAYER1_UNITS]))
    b2 = tf.Variable(tf.random_normal([LAYER2_UNITS]))
    b3 = tf.Variable(tf.random_normal([LAYER3_UNITS]))
    b4 = tf.Variable(tf.random_normal([INPUT_Y]))


    placeholder_x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_X])
    placeholder_y = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_Y])

    r1 = tf.sigmoid(tf.matmul(placeholder_x, w1) + b1)
    r2 = tf.sigmoid(tf.matmul(r1, w2) + b2)
    r3 = tf.sigmoid(tf.matmul(r2, w3) + b3)
    r4 = tf.sigmoid(tf.matmul(r3, w4)) * 300 + b4
    output = r4

    loss = tf.reduce_sum(tf.reduce_mean(tf.square((output - placeholder_y))))
    minimize_step = tf.train.AdamOptimizer().minimize(loss)
    sess = tf.Session()

    def store(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.SAVER_MODEL_NAME + '/model.ckpt')

    def batch_predict(self, batch_x):
        x_size = len(batch_x)
        return self.sess.run(self.output, feed_dict={self.placeholder_x: batch_x,
                                                 self.placeholder_y: np.zeros([x_size, self.INPUT_Y])})
    def train(self, iteration_times, batch_x, batch_y):
        for i in range(iteration_times):
            self.sess.run(self.minimize_step,
                          feed_dict={self.placeholder_x: batch_x,
                                     self.placeholder_y: batch_y})

    def evaluate_loss(self, batch_x, batch_y):
        return self.sess.run(self.loss,
                             feed_dict={self.placeholder_x: batch_x,
                                        self.placeholder_y: batch_y})

    def predict(self, x):
        batch_x = [x]
        batch_y = tf.zeros([1, self.INPUT_Y])
        return self.sess.run(self.output,
                             feed_dict={self.placeholder_x: batch_x,
                                        self.placeholder_y: batch_y})

    def __init__(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('\n')
        print('*' * 32)
        ckpt = tf.train.get_checkpoint_state(self.SAVER_MODEL_NAME)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('saved model loaded')


    def random_test(self, batch_x, batch_y, test_case):
        test_x = []
        test_y = []
        for i in range(test_case):
            r = random.randint(0, len(batch_x) - 1)
            test_x.append(batch_x[r])
            test_y.append(batch_y[r])
        
        print('')
        print('*' * 32)
        print('Test case\nExpect =')
        print(np.mat(test_y))
        print('Predict = ')
        print(self.batch_predict(test_x))
        print('*' * 32)


