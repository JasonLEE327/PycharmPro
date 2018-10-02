import random


class Classfication:

    def readData(self, dataname):
        with open('class' + dataname + '.txt') as f:
            lines = f.readlines()

        newlines = []
        for line in lines:
            data = line.split()
            if data:
                newlines.append(data)

        newdata = []
        for i in range(5):
            d = []
            for j in range(5):
                d = d + (newlines[5 * i + j])
            if 'A' == dataname:
                d.append('1')
            else:
                d.append('-1')
            newdata.append(d)
            newdata = [[int(j) for j in i] for i in newdata]
        return newdata

    def readMysteryData(self):
        with open('Mystery.txt') as f:
            lines = f.readlines()

        newlines = []
        for line in lines:
            data = line.split()
            if data:
                newlines.append(data)
        newdata = []
        for i in range(5):
            d = []
            for j in range(5):
                d = d + (newlines[5 * i + j])
            newdata.append(d)
            newdata = [[int(j) for j in i] for i in newdata]
        return newdata

    def sign(self, v):
        if v > 0:
            return 1
        else:
            return -1

    def training(self):
        train_data1 = self.readData('A')
        train_data2 = self.readData('B')
        train_datas = train_data1 + train_data2

        weight = [0 for i in range(25)]
        bias = 0
        learning_rate = 0.5

        for i in range(1000):
            train = random.choice(train_datas)
            mul = 0
            for i in range(25):
                mul += weight[i] * train[0]
            predict = self.sign(mul + bias)  # input
            print("predict: %d" % predict)
            if train[25] * predict <= 0:
                for i in range(25):
                    weight[i] = weight[i] + learning_rate * train[25] * train[i]
                bias = bias + learning_rate * train[25]  # update bias

        print("stop training")

        return weight, bias

    # test
    def test(self):
        weight, bias = self.training()
        data = self.readMysteryData()
        for i in range(5):
            test_data = data[i]
            mul = 0
            for j in range(25):
                mul += weight[j] * test_data[0]
            predict = self.sign(mul + bias)
            # predict = self.sign(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
            print("predict %d ==> %d" % (i+1, predict))


c = Classfication()
c.test()