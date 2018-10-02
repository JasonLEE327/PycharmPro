import numpy as np
import random

class localization():
    def __init__(self):
        loc_matrix = []
        with open('Localization.txt') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            if (len(line) > 0):
                loc_matrix.append(line)

        self.maze = loc_matrix
        self.width = len(self.maze)

        for i in range(self.width):
            for j in range(self.width):
                if 'G' == self.maze[i][j]:
                    self.target = (i,j)

        self.pos = self.start()


    def start(self):
        row = random.randint(0,self.width-1)
        column = random.randint(0,self.width-1)
        while self.maze[row][column] != '0':
            row = random.randint(0, self.width-1)
            column = random.randint(0, self.width-1)
        row,column = (1,35)
        print "start at point (%s,%s) & our target is at %s " % (row,column,str(self.target))
        return row,column


    def move(self,direction):
        x = self.pos[0]
        y = self.pos[1]
        if 'up' == direction:
            x -= 1
            if '0' == self.maze[x][y] or 'G' == self.maze[x][y]:
                self.pos = (x,y)

        elif 'down' == direction:
            x += 1
            if '0' == self.maze[x][y] or 'G' == self.maze[x][y]:
                self.pos = (x,y)
        elif 'left' == direction:
            y -= 1
            if '0' == self.maze[x][y] or 'G' == self.maze[x][y]:
                self.pos = (x,y)
        elif 'right' == direction:
            y += 1
            if '0' == self.maze[x][y] or 'G' == self.maze[x][y]:
                self.pos = (x,y)


    def brutal_search(self):
        # move the left_bottom of each square
        for i in range(35):
            self.move('down')
        for i in range(35):
            self.move('left')
        for i in range(35):
            self.move('down')
        # afte the 3 steps above we can get the left_bottom of each erea
        print self.pos

        # move to bottom
        for i in range(24):
            self.move('right')
        for i in range(24):
            self.move('down')
        # afte the 2 steps above we can get the 3 position of bottom
        print self.pos

        # move toe left_up point in the left_bottom square
        for i in range(10):
            self.move('up')
        for i in range(24):
            self.move('right')
        # afte the 2 steps above we can get the only point in the maze
        print self.pos

        # then begin to search
        for i in range(5):
            self.move('left')
        for i in range(2):
            self.move('down')
        for i in range(3):
            self.move('left')
        for i in range(6):
            self.move('down')
        for i in range(3):
            self.move('right')
        for i in range(3):
            self.move('up')

        print self.pos



a = localization()
a.brutal_search()
