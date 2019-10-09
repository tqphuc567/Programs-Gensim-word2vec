# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 02:55:05 2019

@author: Phuc123
"""

# Program to multiply two matrices using nested loops

# 4x3 matrix
X = [[1,0,1],
    [1 ,0,0],
    [1 ,0,0],
    [1,1,0]]
# 3x3 matrix
Y = [[5,5,5],
    [5,5,5],
    [5,5,5]]
# result is 4x3
result = [[0,0,0],
         [0,0,0],
         [0,0,0],
         [0,0,0]]

# iterate through rows of X
for i in range(len(X)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]

for r in result:
   print(r)