import numpy as np
from copy import copy as cp
import random
from landscape import landscape

WIDTH = 10

#Prior_belief =

IN_Nofound = {
    1 : 0.1,     #flat
    2 : 0.3,     #hill
    3 : 0.7,     #forest
    4 : 0.9      #cave
}
Type = {
    1: 'flat',
    2:'hill',
    3:'forest',
    4:'cave'
}

class ProSearch():

    def __init__(self,width):
        self.width = width
        self.point = (0,0)
        self.path = [str(self.point)]

        SearchMap = landscape(self.width)
        SearchMap.build_map()

        self.data = SearchMap.data
        self.goal = SearchMap.goal
        self.belief = np.zeros([self.width, self.width])
        self.surveillance = []

        self.coordinates = []
        for x in range(self.width):
            for y in range(self.width):
                self.coordinates.append((x, y))

        for c in self.coordinates:
            self.belief[c] = 1.0/(self.width*self.width)

    def update_belief(self,cell_t_1):   #cell_t_1 is last point explored
        # update last cell's belief
        belief_t_1 = self.belief[cell_t_1]
        self.belief[cell_t_1] =  \
            (belief_t_1 * IN_Nofound[self.data[cell_t_1]])/ \
            (belief_t_1 * IN_Nofound[self.data[cell_t_1]] + (1 - belief_t_1))

        coordinates = cp(self.coordinates)
        coordinates.remove(cell_t_1)
        #
        for c in coordinates:
            self.belief[c] = self.belief[c]/  \
                (belief_t_1 * IN_Nofound[self.data[cell_t_1]] + (1 - belief_t_1))


    def target_move(self):
        type1 = self.data[self.goal]
        x,y = self.goal
        self.goal = random.choice([x for x in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)] if self.check_point_vaild(x)])
        type2 = self.data[self.goal]
        self.surveillance = [type1,type2]


    def check_pattern_with_surveillance(self,current_pattern):
        return 1.0 if sorted(current_pattern) == sorted(self.surveillance) else 0.0

    def check_point_vaild(self,pos):
        x,y = pos
        return True if x >= 0 and y >= 0 and x < self.width and y < self.width else False

    def update_after_move(self,cellpos,Belief):
        x,y = cellpos
        p = 0
        for a,b in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            if self.check_point_vaild((a,b)):
                pro = self.check_pattern_with_surveillance([self.data[(a, b)], self.data[cellpos]]) * \
                    Belief[(a,b)]
                if 0 == pro:
                    p += 0
                else:
                    sum = 0
                    for c,d in [(a-1,b),(a+1,b),(a,b-1),(a,b+1)]:
                        if self.check_point_vaild((c,d)):
                            sum += self.check_pattern_with_surveillance([self.data[(c,d)],self.data[a,b]])
                    p += pro/sum
        return p

    def update_predict(self):
        belief_copy = cp(self.belief)
        predict = np.zeros([self.width, self.width])
        for c in self.coordinates:
            predict[c] = self.update_after_move(c,belief_copy)
        self.belief = predict

        scale = np.sum(self.belief)
        self.belief = np.divide(self.belief, scale)


    def get_biggest_belief(self):
        max_pos = np.argmax(self.belief)
        x,y = divmod(max_pos, self.width)
        return (x,y)

    def run(self):
        #goal_pro = []
        while True:
            #if self.width == 50:
            #    goal_pro.append(self.belief[self.goal])

            self.update_belief(self.point)
            self.point = self.get_biggest_belief()

            self.path.append(str(self.point))
            if self.point == self.goal and (1-IN_Nofound[self.data[self.point]])*100 > random.randint(1,100):
            #if self.point == self.goal:
                break

            #when failed,move target,update belief
            self.target_move()
            #print self.goal
            self.update_predict()
            #print self.belief

        #print "Target type: %s, the step cost: %s" % (Type[self.data[self.point]],len(self.path))
        #if goal_pro:
            #print goal_pro
        return len(self.path)
        #print (" => ").join(self.path)


if __name__ == "__main__":
    for case in range(5):
        re = []
        for i in range(100):
            Search = ProSearch(WIDTH)
            cost = Search.run()
            re.append(cost)
            #print '%sth finished and cost is %s' %(i+1,cost)

        print "Landscape width is %s, average cost:%s" % (WIDTH,float(sum(re)/len(re)))

        WIDTH += 10
        #print re








