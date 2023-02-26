import os
from scipy.io import loadmat
import numpy as np
import copy
import sys
import math
import cmath


def batch_generator(arr, batch_size, input_sizes, output_sizes):
    arr = copy.copy(arr)
    all_sizes = input_sizes+output_sizes
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr_target = np.zeros((batch_size, all_sizes * n_batches))
    for i in range(len(arr)):
        arr_target[i%batch_size, int(i/batch_size)*all_sizes:int(i/batch_size)*all_sizes+all_sizes] = arr[i]
    arr_out = []
    for n in range(0, arr_target.shape[1]//all_sizes):
        x_con = []
        y_con = []
        x = arr_target[:, n*all_sizes : n*all_sizes+input_sizes]
        y = arr_target[:, n*all_sizes+input_sizes: (n+1)*all_sizes]
        x_con.append(x)
        y_con.append(y)
        x_con = np.array(x_con)
        y_con = np.array(y_con)
        x_con = x_con.reshape((batch_size, input_sizes))
        y_con = y_con.reshape((batch_size, output_sizes))
        arr_out.append((x_con, y_con))
    return arr_out