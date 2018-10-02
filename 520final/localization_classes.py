# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from collections import deque


class LocalizationMaze(object):
    """Basic Graph Class
    1, use dim and rate to create a maze
    2, then generate a graph by this maze
    3, draw the maze with or without path
    """

    def num_of_v(self):  # return the num of
        return self.v

    def adj_of_v(self, v):  # all the adjunct point
        return self.adj[v]

    def to_string(self, v):  # use to print
        return (int(v // self.dim), int(v % self.dim))

    def get_maze(self):  # return the maze matrix
        return self.maze

    def get_start(self):
        x = random.randint(1, self.dim)
        y = random.randint(1, self.dim)
        while self.maze[x][y] is not '0':
            x = random.randint(1, self.dim)
            y = random.randint(1, self.dim)
        return (x * self.dim + y)

    def __init__(self):  # init with dim and rate
        # generate maze
        self.target = 0
        self.start = 0
        loc_matrix = []
        with open('Localization.txt') as locs:
            r = locs.read()
            lines = r.split('\n')
            for line in lines:
                if(len(line) > 0):
                    ls = line.split('\t')
#                    lss = [int(x) for x in ls]
                    loc_matrix.append(ls)
        self.maze = loc_matrix
        self.dim = len(loc_matrix)
        self.adj = []
        rows = np.shape(self.maze)[0]
        columns = np.shape(self.maze)[1]
        self.v = rows * columns
        # init all adj_lists
        for i in range(self.v):
            self.adj.append([])

        self.spaces = 0
        # init all edges
        for row, row_list in enumerate(self.maze):
            for column, item in enumerate(row_list):
                if item == 'G':
                    self.target = row * self.dim + column
                if item == '0' or item == 'G':
                    self.spaces += 1
                    index = row * rows + column
                    if row > 0 and self.maze[row-1][column] == '0' or self.maze[row-1][column] == 'G':
                        self.adj[index].append(index - columns)
                    if row < rows - 1 and self.maze[row+1][column] == '0' or self.maze[row-1][column] == 'G':
                        self.adj[index].append(index + columns)
                    if column > 0 and self.maze[row][column-1] == '0' or self.maze[row-1][column] == 'G':
                        self.adj[index].append(index - 1)
                    if column < columns - 1 and self.maze[row][column+1] == '0' or self.maze[row-1][column] == 'G':
                        self.adj[index].append(index + 1)
        self.spaces += 1
        self.start = self.get_start()
#        self.start = 266 # 42 # 1111
        self.compute_surrounding_blocks()


    def compute_surrounding_blocks(self):
        self.hints = {}
        for row in range(self.dim):
            for col in range(self.dim):
                if self.maze[row][col] == '0' or self.maze[row][col] == 'G':
                    surround = 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            ty_index = col + i
                            tx_index = row + j
                            if ty_index >= 0 and ty_index < self.dim and tx_index >= 0 and tx_index < self.dim:
                                if self.maze[tx_index][ty_index] == '1':
                                    surround += 1
                    if surround in self.hints:
                        self.hints[surround].append(row*self.dim + col)
                    else:
                        self.hints[surround] = [row*self.dim + col]


    def find_five_surrounds(self):
        fives = self.hints[5]
        self.maybein = []
        for i in range(len(fives)-2):
            if (fives[i]+1) in fives and (fives[i]+2) in fives:
                self.maybein.append(fives[i])
        self.print_all_fives()

    def print_all_fives(self):
        for may in self.maybein:
            print(self.to_string(may))

    def find_most_probabily(self, obs, acts):
        begin_points = self.hints[obs[0]]
        for i in range(1, len(obs)):
            act = acts[i]
            if act == 'up':
                tp = -self.dim
            elif act == 'down':
                tp = self.dim
            elif act == 'left':
                tp = -1
            elif act == 'right':
                tp = 1

            for point in begin_points:
                next_point = point + tp
                if next_point not in self.hints[obs[i]]:
                    begin_points.remove(point)
            for i in range(len(begin_points)):
                begin_points[i] = begin_points[i] + tp

            for p in begin_points:
                if p < 0 or p > self.dim**2:
                    begin_points.remove(p)
        self.final_points = begin_points
        for p in self.final_points:
            print(self.to_string(p))
#        print(self.final_points)
#    def find_point_with_sournd




    def move(self, d):
        x = self.start // self.dim
        y = self.start % self.dim
        if d == 'left':
            y -= 1
            if self.maze[x][y] == 'G':
                print('reach target')
                self.start = self.target
            elif self.maze[x][y] == '0':
                self.start = x * self.dim + y
        elif d == 'right':
            y += 1
            if self.maze[x][y] == 'G':
                print('reach target')
                self.start = self.target
            elif self.maze[x][y] == '0':
                self.start = x * self.dim + y
        elif d == 'up':
            x -= 1
            if self.maze[x][y] == 'G':
                print('reach target')
                self.start = self.target
            elif self.maze[x][y] == '0':
                self.start = x * self.dim + y
        elif d == 'down':
            x += 1
            if self.maze[x][y] == 'G':
                print('reach target')
                self.start = self.target
            elif self.maze[x][y] == '0':
                self.start = x * self.dim + y

    def brute_search(self):
        # move the the outline of each square
        for i in range(7):
            self.move('down')
        for i in range(7):
            self.move('left')
        for i in range(7):
            self.move('down')

        for i in range(1):
            self.move('right')
        for i in range(2):
            self.move('down')
        for i in range(3):
            self.move('left')

        for i in range(6):
            self.move('up')
        for i in range(3):
            self.move('right')
        for i in range(2):
            self.move('up')
        for i in range(1):
            self.move('right')

        # move the left_bottom of each square
        for i in range(40):
            self.move('down')
        for i in range(40):
            self.move('left')
        for i in range(40):
            self.move('down')
        # afte the 3 steps above we can get the left_bottom of each erea

        # move to bottom
        for i in range(24):
            self.move('right')
        for i in range(24):
            self.move('down')
        # afte the 2 steps above we can get the 3 position of bottom

        # move toe left_up point in the left_bottom square
        for i in range(10):
            self.move('up')
        for i in range(24):
            self.move('right')
        # afte the 2 steps above we can get the only point in the maze

        # then begin to search
        for i in range(5):
            self.move('left')
        for i in range(2):
            self.move('down')
        for i in range(3):
            self.move('right')
        for i in range(6):
            self.move('down')
        for i in range(3):
            self.move('left')
        for i in range(3):
            self.move('up')


        self.draw_maze()

    # draw the maze, if path is not none, draw path together
    def draw_maze(self, title='520 project 1', size=10, path=[]):
        fig, ax = plt.subplots()
        draw_path = False
        path_len = len(path)
        if path_len > 0:
            draw_path = True
        for i in range(np.shape(self.maze)[0]):
            for j in range(np.shape(self.maze)[1]):
                x = (j*10 + 1) / 10
                y = (i*10 + 1) / 10
                h = 9 / 10
                w = 9 / 10
                index = i * self.dim + j
                cl = '#cc6600' if self.maze[i][j] == '1' else '#EEEEEE'
                if self.maze[i][j] == 'G':
                    cl = '#FF9999'
                    ax.text(x+0.1, y+0.6, 'G', fontsize=12, color='#000000')
                elif (i * self.dim + j) == self.start:
                    cl = '#00F0F0'
                    ax.text(x+0.1, y+0.6, 'S', fontsize=12, color='#000000')
                elif draw_path:
                    if index in path:
                        cl = '#00F000'
                        ax.text(x+0.1, y+0.6, path_len-path.index(index)-2,
                                fontsize=8, color='#666666')
                rect = mpatches.Rectangle([x, y], w, h, color=cl)
                ax.add_patch(rect)

        plt.axis([-0.1, self.dim + 0.2, -0.1, self.dim + 0.2])
        plt.gca().invert_yaxis()
        fig.set_size_inches(size, size)
        plt.title(title, fontsize=15)
        plt.show()

    def marked(self, w):
        return self.marked[w]

    def has_path_to(self, v):
        return self.marked[v]

    def path(self):
        if self.has_path_to(self.target) == 0:
            print('No Path')
            return None
        path = [self.target]
        x = self.target
        while x != self.start:
            x = self.edge_to[int(x)]
            path.append(x)
        path.append(self.start)
        return path

    def total_expanded(self):
        return self.marked.sum()
