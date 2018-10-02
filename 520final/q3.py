import numpy as np

def readData(dataname):
    try:
        with open('class' + dataname + '.txt') as f:
            lines = f.readlines()
    except:
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
            d = d + (newlines[5*i+j])
        if 'A' == dataname:
            d += '0'
        elif 'B' == dataname:
            d += '1'

        newdata.append(d)
    return newdata


def str2int(lis):
    list_new = []
    for i in range(len(lis)):
        list_new.append(int(lis[i]))
    return list_new

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

class classify():
    def __init__(self):
        self.Data_A = readData('A')
        self.Data_B = readData('B')
        self.Data_Mystery = readData('C')

        self.Weight = [1 for i in range(25)]
        self.alf  = 0.1

    def logisfunc(self,data):
        data_int = str2int(data)
        f = sigmoid(sum(np.multiply(data_int[:25], self.Weight)))

        for i in range(25):
            self.Weight[i] = self.Weight[i] - (self.alf * (f - data_int[25]) * data_int[i])


    def train(self):
        for k in range(10000):
            for i in range(5):
                self.logisfunc(self.Data_A[i])
            for i in range(5):
                self.logisfunc(self.Data_B[i])

    def run(self):
        self.train()
        result = []
        for mystery in self.Data_Mystery:
            mystery_int = str2int(mystery)
            r = sigmoid(sum(np.multiply(mystery_int, self.Weight)))
            result.append(int(round(r)))
        print result

a = classify()
a.run()




