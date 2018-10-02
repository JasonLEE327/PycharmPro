#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: li
'''


class Position(object):
    def __init__(self):
        self.level1_1_to_2 = None
        self.level1_2_to_1 = None
        self.level2_1_to_2 = None
        self.level2_2_to_1 = None
        self.team1_current = None
        self.team2_current = None
        self.pre_score1_2 = None
        self.pre_score2_1 = None
        self.current_score1_2 = None
        self.current_score2_1 = None

    def set_value(self,team1_current, team2_current):
        self.team1_current = team1_current
        self.team2_current = team2_current

    def set_map(self, level112, level212, level121, level221):
        self.level1_1_to_2 = level112
        self.level1_2_to_1 = level121
        self.level2_1_to_2 = level212
        self.level2_2_to_1 = level221

    def set_pre_score(self,pre_score1_2, pre_score2_1):
        self.pre_score1_2 = pre_score1_2
        self.pre_score2_1 = pre_score2_1

    def set_current_score(self,current_score1_2, current_score2_1):
        self.current_score1_2 = current_score1_2
        self.current_score2_1 = current_score2_1


class Formation(object):
    def __init__(self, back1, mid1, front1, back2, mid2, front2):
        self.back1 = back1
        self.mid1 = mid1
        self.front1 = front1
        self.back2 = back2
        self.mid2 = mid2
        self.front2 = front2
        # network
        self.T1 = Position()
        self.BL = Position()
        self.B = Position()
        self.BR = Position()
        self.ML = Position()
        self.M = Position()
        self.MR = Position()
        self.FL = Position()
        self.F = Position()
        self.FR = Position()
        self.T2 = Position()

        self.T1.set_map(None, None, (self.BL, self.B, self.BR), (self.ML, self.M, self.MR))

        self.BL.set_map((self.T1,), None,(self.ML, self.M), (self.FL, self.F))

        self.B.set_map((self.T1,), None, (self.ML, self.M, self.MR), (self.FL, self.F, self.FR))

        self.BR.set_map((self.T1,), None, (self.M, self.MR), (self.F, self.FR))

        self.ML.set_map((self.BL, self.B), None, (self.FL, self.F), None)

        self.M.set_map((self.BL, self.B, self.BR), None, (self.FL, self.F, self.FR), None)

        self.MR.set_map((self.B, self.BR), None, (self.F, self.FR), None)

        self.FL.set_map((self.ML, self.M), (self.BL, self.B), (self.T2,), None)

        self.F.set_map((self.ML, self.M, self.MR), (self.BL, self.B, self.BR), (self.T2,), None)

        self.FR.set_map((self.M, self.MR), (self.B, self.BR), (self.T2,), None)

        self.T2.set_map((self.FL, self.F, self.FR), (self.ML, self.M, self.MR), None, None)



    def set_player(self, team1, team2):
        # 0 - 10 F, M, B, Goal keeper
        F_player_T1 = team1[0:self.front1]
        M_player_T1 = team1[self.front1:self.front1+self.mid1]
        B_player_T1 = team1[self.front1+self.mid1:self.front1+self.mid1+self.back1]
        G_player_T1 = team1[len(team1) - 1:len(team1)]

        F_player_T2 = team2[0:self.front2]
        M_player_T2 = team2[self.front2:self.front2 + self.mid2]
        B_player_T2 = team2[self.front2 + self.mid2:self.front2 + self.mid2 + self.back2]
        G_player_T2 = team2[len(team2) - 1:len(team2)]

        # get FL, F ,FR
        F1 = self.divide_abiity_number(F_player_T1)
        B2 = self.divide_abiity_number(B_player_T2)
        self.FL.set_value(F1[0], B2[0])
        self.F.set_value(F1[1], B2[1])
        self.FR.set_value(F1[2], B2[2])

        # get ML, M ,MR
        M1 = self.divide_abiity_number(M_player_T1)
        M2 = self.divide_abiity_number(M_player_T2)
        self.ML.set_value(M1[0], M2[0])
        self.M.set_value(M1[1], M2[1])
        self.MR.set_value(M1[2], M2[2])

        # get BL, B ,BR
        B1 = self.divide_abiity_number(B_player_T1)
        F2 = self.divide_abiity_number(F_player_T2)
        self.BL.set_value(B1[0], F2[0])
        self.B.set_value(B1[1], F2[1])
        self.BR.set_value(B1[2], F2[2])

        # set goal keeper
        self.T1.set_value(100-team1[10], team1[10])
        self.T1.set_current_score(0.5, None)
        self.T2.set_value(team2[10], 100-team2[10])
        self.T2.set_current_score(None, 0.5)

    def divide_abiity_number(self, players):
        number_means = float(len(players))/3
        result = [0, 0, 0]
        i = 0
        count = number_means
        left = float(0)
        position = 0
        while i < len(players):
            if count >= 0.99999999:
                result[position] += players[i]
                i += 1
                count = count - 1
                if count == 0:
                    position += 1
                    count = number_means
            else:
                result[position] += players[i] * count
                position += 1
                if number_means + 0.000001 >= (1 - count):
                    result[position] += players[i] * (1 - count)
                    i += 1
                    count = number_means - (1 - count)
                else:
                    result[position] += players[i] * count
                    position += 1
                    result[position] += players[i] * count
                    position += 1
                    count = 0
                    i += 1

        print result, players
        return result

    def get_Team1(self):
        result = 0
        self.get_current(self.BL)
        self.get_current(self.B)
        self.get_current(self.BR)
        self.get_current(self.ML)
        self.get_current(self.M)
        self.get_current(self.MR)
        self.get_current(self.FL)
        self.get_current(self.F)
        self.get_current(self.FR)
        self.get_current(self.T2)
        print "self.T2.current_score1_2", self.T2.current_score1_2
        result = self.T2.current_score1_2
        return result

    def get_Team2(self):
        result = 0
        self.get_current(self.FL,False)
        self.get_current(self.F,False)
        self.get_current(self.FR,False)
        self.get_current(self.ML,False)
        self.get_current(self.M,False)
        self.get_current(self.MR,False)
        self.get_current(self.BL,False)
        self.get_current(self.B,False)
        self.get_current(self.BR,False)
        self.get_current(self.T1,False)
        print "self.T1.current_score2_1", self.T1.current_score2_1
        result = self.T1.current_score2_1
        return result

    def get_current(self, current_position, direction1_2 = True):
        level1_score = float(0)
        level2_score = float(0)
        level2to1_constant = 0.4
        if direction1_2:
            count_ways = 0
            if current_position.level1_1_to_2 is not None:
                for pre in current_position.level1_1_to_2:
                    level1_score += 1 * pre.current_score1_2
                    count_ways += 1
                level1_score = level1_score / count_ways  # could use max here TODO
                count_ways = 0
            if current_position.level2_1_to_2 is not None:
                for pre in current_position.level2_1_to_2:
                    level2_score += 1 * pre.current_score1_2
                    count_ways += 1
                level2_score = level2_score / count_ways  # could use max here TODO
            print (level1_score+level2to1_constant*level2_score) * \
                current_position.team1_current/(current_position.team1_current+current_position.team2_current)
            current_position.current_score1_2 = (level1_score+level2to1_constant*level2_score) * \
                current_position.team1_current/(current_position.team1_current+current_position.team2_current)
        else:
            count_ways = 0
            if current_position.level1_2_to_1 is not None:
                for pre in current_position.level1_2_to_1:
                    level1_score += 1 * pre.current_score2_1
                    count_ways += 1
                level1_score = level1_score / count_ways  # could use max here TODO
                count_ways = 0
            if current_position.level2_2_to_1 is not None:
                for pre in current_position.level2_2_to_1:
                    level2_score += 1 * pre.current_score2_1
                    count_ways += 1
                level2_score = level2_score / count_ways  # could use max here TODO
            print (level1_score+level2to1_constant*level2_score) * \
                current_position.team2_current/(current_position.team1_current+current_position.team2_current)
            current_position.current_score2_1 = (level1_score+level2to1_constant*level2_score) * \
                current_position.team2_current/(current_position.team1_current+current_position.team2_current)


if __name__ == "__main__":
    f = Formation(1, 5, 4, 1, 5, 4)
    f.set_player([90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90])
    f.get_Team1()
    f.get_Team2()

    f = Formation(4, 4, 2, 4, 4, 2)
    f.set_player([90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], [85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85])
    f.get_Team1()
    f.get_Team2()

