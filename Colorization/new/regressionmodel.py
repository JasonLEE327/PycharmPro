import numpy as np
import remote_message, os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    sig = sigmoid(x)
    return np.multiply(sig, 1 - sig)

def normalize(shape):
    return np.mat(np.random.random(shape), dtype=np.float32) * 2 - 1


class NNModel:

    INPUT_SIZE = (INPUT_X, INPUT_Y) = (9, 3)
    LAYER1_UNITS = LAYER2_UNITS = 9

    w1 = normalize([INPUT_X, LAYER1_UNITS])
    w2 = normalize([LAYER1_UNITS, LAYER2_UNITS])
    w3 = normalize([LAYER2_UNITS, INPUT_Y])

    b1 = normalize([1, LAYER1_UNITS])
    b2 = normalize([1, LAYER2_UNITS])
    b3 = normalize([1, INPUT_Y])

    STORE_FILENAME = ['w1.npy', 'w2.npy', 'w3.npy',
                      'b1.npy', 'b2.npy', 'b3.npy']
    STORE_FOLDER = 'model_data'

    LEARNING_RATE = 5e-2

    USE_DROP_OUT = True
    USE_ADAM = False

    last_w1 = np.zeros(np.shape(w1))
    last_w2 = np.zeros(np.shape(w2))
    last_w3 = np.zeros(np.shape(w3))
    
    last_b1 = np.zeros(np.shape(b1))
    last_b2 = np.zeros(np.shape(b2))
    last_b3 = np.zeros(np.shape(b3))

    def _normalize_x(self, x):
        x = np.mat(x, dtype=np.float32)
        return x
        mean_x = np.mean(x)
        min_x = np.min(x)
        max_x = np.max(x)
        return (x - mean_x) / (max_x - min_x)
    

    def predict(self, batch_x):
        x = self._normalize_x(batch_x)
        r1 = sigmoid(np.matmul(batch_x, self.w1) + self.b1)
        r2 = sigmoid(np.matmul(r1, self.w2) + self.b2)
        r3 = sigmoid(np.matmul(r2, self.w3) + self.b3) * 255
        return r3

    def train_step(self, iteration_times, batch_x, batch_y):
        x = self._normalize_x(batch_x)
        y = np.mat(batch_y, dtype=np.float32)

        batch_size = len(batch_x)

        b_input = np.ones([batch_size])

        for i in range(iteration_times):
            i1 = np.matmul(x, self.w1) + self.b1
            o1 = sigmoid(i1)
            d1 = np.multiply(o1, 1 - o1)
            i2 = np.matmul(o1, self.w2) + self.b2
            o2 = sigmoid(i2)
            d2 = np.multiply(o2, 1 - o2)
            i3 = np.matmul(o2, self.w3) + self.b3
            o3 = sigmoid(i3)
            d3 = np.multiply(o3, 1 - o3)

            predict_y = o3 * 255
            l3_loss = np.multiply((predict_y - y), d3) * self.LEARNING_RATE / batch_size
            # l3_loss = l3_loss / (np.max(l3_loss) - np.min(l3_loss)) * .002
            w3_gradient = np.matmul(o2.T, (l3_loss))
            b3_gradient = np.matmul(b_input.T, (l3_loss))
            
            l2_loss = np.multiply(np.matmul(l3_loss, self.w3.T) + np.matmul(l3_loss, self.b3.T), d2)
            l2_loss *= self.LEARNING_RATE
            w2_gradient = np.matmul(o1.T, (l2_loss))
            b2_gradient = np.matmul(b_input.T, (l2_loss))

            l1_loss = np.multiply(np.matmul(l2_loss, self.w2.T) + np.matmul(l2_loss, self.b2.T), d1)
            l1_loss *= self.LEARNING_RATE
            w1_gradient = np.matmul(x.T, (l1_loss))
            b1_gradient = np.matmul(b_input.T, (l1_loss))

            def dropout_layer(probability, mat):
                dropout = np.random.random(np.shape(mat))
                mat[dropout < probability] = 0

            if self.USE_DROP_OUT:
                dropout_layer(.1, w3_gradient)
                dropout_layer(.1, b3_gradient)

                dropout_layer(.1, w2_gradient)
                dropout_layer(.1, b2_gradient)

                dropout_layer(.1, w1_gradient)
                dropout_layer(.1, b1_gradient)

            if self.USE_ADAM:
                w3_gradient = .9 * w3_gradient + .1 * self.last_w3
                w2_gradient = .9 * w2_gradient + .1 * self.last_w2
                w1_gradient = .9 * w1_gradient + .1 * self.last_w1

                b3_gradient = .9 * b3_gradient + .1 * self.last_b3
                b2_gradient = .9 * b2_gradient + .1 * self.last_b2
                b1_gradient = .9 * b1_gradient + .1 * self.last_b1

                self.last_w3 = w3_gradient.copy()
                self.last_w2 = w2_gradient.copy()
                self.last_w1 = w1_gradient.copy()

                self.last_b3 = b3_gradient.copy()
                self.last_b2 = b2_gradient.copy()
                self.last_b1 = b1_gradient.copy()


            self.w3 -= w3_gradient
            self.w2 -= w2_gradient
            self.w1 -= w1_gradient
            self.b3 -= b3_gradient
            self.b2 -= b2_gradient
            self.b1 -= b1_gradient
        # print(o1)

    def loss(self, batch_x, batch_y):
        predict_y = self.predict(batch_x)
        expect_y = np.mat(batch_y)
        return np.mean(np.square(predict_y - expect_y))

        

    def __init__(self):
        model_exists = True
        for filename in self.STORE_FILENAME:
            model_exists = model_exists and os.path.exists(self.STORE_FOLDER + '/' + filename)
        if model_exists:
            self.w1 = np.load(self.STORE_FOLDER + '/w1.npy')
            self.w2 = np.load(self.STORE_FOLDER + '/w2.npy')
            self.w3 = np.load(self.STORE_FOLDER + '/w3.npy')
            self.b1 = np.load(self.STORE_FOLDER + '/b1.npy')
            self.b2 = np.load(self.STORE_FOLDER + '/b2.npy')
            self.b3 = np.load(self.STORE_FOLDER + '/b3.npy')
            print('Pre-trained model loaded.')
        print(self.b1)


    def store(self):
        np.save(self.STORE_FOLDER + '/w1.npy', self.w1)
        np.save(self.STORE_FOLDER + '/w2.npy', self.w2)
        np.save(self.STORE_FOLDER + '/w3.npy', self.w3)
        np.save(self.STORE_FOLDER + '/b1.npy', self.b1)
        np.save(self.STORE_FOLDER + '/b2.npy', self.b2)
        np.save(self.STORE_FOLDER + '/b3.npy', self.b3)
    

    
