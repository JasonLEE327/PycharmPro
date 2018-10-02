#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:52:21 2017

@author: xubo
"""
import copy


def get_utility(state, discount, err):
    us_new = []
    #until = err * (1 - discount) / discount
    #until = (1-discount) /discount
    #print until
    for i in range(state):
        us_new.append([0, 0])
#    us = us_new.copy()
    max_charge = 100
    u = 0
    r = 0
    while max_charge >= err:
        print max_charge
#        print('while')
        us = copy.deepcopy(us_new)
        max_charge = 0
        for i in range(state):
            if i == 0:
                us_new[i][0] = 100 + discount * us[i + 1][0]
                us_new[i][1] = 0
            elif i == state - 1:
                us_new[i][0] = discount * us[0][0] - 250
                us_new[i][1] = 1
            else:
                u = 100 - 10 * i + discount * (0.1 * i * us[i + 1][0] + (1 - 0.1 * i) * us[i][0])
                r = discount * us[0][0] - 250
#                print('u, r:', u, r)
                if u > r:
                    us_new[i][0] = u
                    us_new[i][1] = 0
                else:
                    us_new[i][0] = r
                    us_new[i][1] = 1
#            print(us_new[i][0], us[i][0])
            if abs(us_new[i][0] - us[i][0]) > max_charge:
                max_charge = abs(us_new[i][0] - us[i][0])
#        print(max_charge)
    return us


def get_cost(state, discount, err):
    cost = 0
#    us_new = []
#    for i in range(state):
#        us_new.append([0, 0])
#    us = copy.deepcopy(us_new)
    for i in range(250, 0, -1):
        until = err * (1 - discount) / discount
        us_new = []
        for n in range(state):
            us_new.append([0, 0])
        max_charge = 100
        u = 0
        r = 0
        ru = 0
        print(i)
        while max_charge > until:
#            if i == 170:
#                print(max_charge - until)
            us = copy.deepcopy(us_new)
            max_charge = 0
            for j in range(state):
                if j == 0:
                    us_new[j][0] = 100 + discount * us[j+1][0]
                    us_new[j][1] = 0
                elif j == state - 1:
                    ru = discount * (0.5 * us[1][0] + 0.5 * us[2][0]) - i
                    r = discount * us[0][0] - 250
                    if ru > r:
                        us_new[j][0] = ru
                        us_new[j][1] = 2
                    else:
                        us_new[j][0] = r
                        us_new[j][1] = 1
                else:
                    u = 100 - 10 * j + discount * (0.1 * j * us[j+1][0] + (1 - 0.1 * j) * us[j][0])
                    r = discount * us[0][0] - 250
                    ru = discount * (0.5 * us[1][0] + 0.5 * us[2][0]) - i
                    if u > r:
                        if ru > u:
                            us_new[j][0] = ru
                            us_new[j][1] = 2
                        else:
                            us_new[j][0] = u
                            us_new[j][1] = 0
                    else:
                        if ru > r:
                            us_new[j][0] = ru
                            us_new[j][0] = 2
                        else:
                            us_new[j][0] = r
                            us_new[j][1] = 1
                if abs(us_new[j][0] - us[j][0]) > max_charge:
                    max_charge = abs(us_new[j][0] - us[j][0])
        for k in range(state):
            if us[k][1] == 2:
                return i
    return cost


if __name__ == '__main__':
   u = get_utility(10, 0.9, 0.1)
   print(u)
#     cost = get_cost(10, 0.9, 0.1)
#     print(cost)
