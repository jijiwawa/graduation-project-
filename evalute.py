import scipy.sparse as sp
import numpy as np
import math
import time
import re
import os
import random

def Evaluate_MAE(preditmatrix, testmatrix):
    matrix_sub = sp.dok_matrix.copy(testmatrix)
    num_users,num_items = preditmatrix.shape
    userid1 = -1
    m = 0
    n = np.zeros(num_items)
    for (userid, itemid) in testmatrix.keys():
        matrix_sub[userid, itemid] = math.fabs(matrix_sub[userid, itemid] - preditmatrix[userid][itemid])
        if userid != userid1:
            m += 1
        n[userid] += 1
        userid1 = userid
    sum = 0
    sum_each_row = np.sum(matrix_sub.toarray(), axis=1)
    for i in range(0, num_items):
        if n[i] != 0:
            sum += sum_each_row[i] / n[i]
    MAE = sum / m
    return MAE


if __name__ == '__main__':
    # ml-100k
    preditmatrix = np.load(os.getcwd() + '\\out_file\\A.npy')
    testmatrix = np.load(os.getcwd() + '\\out_file\\A.npy')
    Evaluate_MAE()
