import numpy as np
import random


FLAT_PRO = 0.2
HILL_PRO = 0.3
FOREST_PRO = 0.3
CAVE_PRO = 0.2


class landscape(object):

    def __init__(self,width):
        self.width = width
        self.data = np.zeros([self.width,self.width],dtype = np.int)
        self.goal = (0,0)

    def random_remove(self,lis):
        point = random.choice(lis)
        lis.remove(point)
        return point

    def set_goal(self,lis):
        self.goal = random.choice(lis)


    def build_map(self):
        coordinates = []
        for x in range(self.width):
            for y in range(self.width):
                coordinates.append((x,y))

        self.set_goal(coordinates)

        for i in range(int(self.width**2 * FLAT_PRO)):
            self.data[self.random_remove(coordinates)] = 1
        for i in range(int(self.width**2 * HILL_PRO)):
            self.data[self.random_remove(coordinates)] = 2
        for i in range(int(self.width**2 * FOREST_PRO)):
            self.data[self.random_remove(coordinates)] = 3
        for i in range(int(self.width**2 * CAVE_PRO)):
            self.data[self.random_remove(coordinates)] = 4

        #print self.goal
        #print self.data

#a = landscape(WIDTH)
#a.build_map()