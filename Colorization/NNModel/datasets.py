import csv, random
import numpy as np

data_filename = 'data.csv'
input_filename = 'input.csv'
color_filename = 'color.csv'

def read_data():

    vector_x = []
    vector_y = []

    with open('color.csv', 'r') as color_file:
        with open('input.csv', 'r') as input_file:
            input_data = csv.reader(input_file)
            color_data = csv.reader(color_file)
            for input_row, color_row in zip(input_data, color_data):
                x = [int(x) for x in input_row]
                x.append(1)
                vector_x.append(x)
                vector_y.append([float(v) for v in color_row])
    return vector_x, vector_y


def next_batch(datasets_x, datasets_y, batch_size):
    vector_x = []
    vector_y = []
    for i in range(batch_size):
        rand_index = random.randint(0, len(datasets_x) - 1)
        vector_x.append(datasets_x[rand_index])
        vector_y.append(datasets_y[rand_index])
    return (vector_x), (vector_y)
