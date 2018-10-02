#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:35:26 2017

@author: xubo
"""
from localization_classes import LocalizationMaze


if __name__ == '__main__':
    print('520_p1 main function')
    maze = LocalizationMaze()
    print('total space :', maze.spaces)
    print('probability at G : 1/' + str(maze.spaces))
#    maze.draw_maze(size=7)
    print('start at point', maze.to_string(maze.start))
#    path = maze.bfs_search()
#    maze.draw_maze(path = path)
    maze.brute_search()
    ob = [5, 5, 5, 6, 4, 5, 6]
    ac = ['left', 'left', 'up', 'up', 'left', 'right', 'left']
    maze.find_most_probabily(ob, ac)


