import scipy.sparse as sp
import numpy as np
import prepare_datasets
import multiprocessing
import timeit
import operator
import os
import re
from numpy.random import RandomState

# 打开文件
# 分割数据集
#
'''
验证分割的数据集是否正确
调用算法进行预训练，跑起来
'''


def test_split():
    s = '123,234,,422:123  412 : 1::::1241'
    s1 = '1::1193::5::978300760'
    s3 = '195,241,3,881250949'
    pattern = r'[,|\s|:]+'
    result = re.split(pattern, s3)
    print(result)


def test_rand():
    rdm = RandomState(1)
    dataset_size = 5
    X = rdm.rand(dataset_size, 2)
    print(X is list)
    print(X is tuple)
    print(X is list)
    print(type(X))
    print(X)


def test_sp():
    a = sp.dok_matrix((5, 5), dtype=np.float32)
    a[0, 0] = 1
    a[3, 2] = 5
    a[1, 2] = 3
    print(a)
    print(sp.dok_matrix.toarray(a))
    print(sp.dok_matrix.toarray(a)[0])
    b = sp.dok_matrix.transpose(a).toarray()
    print(b)


def test_path():
    print(os.getcwd() + '\\out_file')
    A = [[1, 2], [3, 4]]
    np.save(os.getcwd() + '\\out_file\\A[%d].npy' % (1), A)
    B = np.load(os.getcwd() + '\\out_file\\A.npy')
    for i in range(0, 5):
        C = np.load(os.getcwd() + '\\out_file\\userSimilarity[%d].npy' % (i))
        print(C)
    D = np.load(os.getcwd() + '\\out_file\\itemSimialrityMatrix_Sitem.npy')
    print(D)
    print(B)


def mult():
    for userId_i in range(0, 5):
        # userId_i = [x1 for x1 in range(0, 5)]
        userId_i1 = [userId_i for x in range(0, 5)]
        userId_j = [x2 for x2 in range(0, 5)]
        zip_args = list(zip(userId_i1, userId_j))
        print(zip_args)


def do_something(x):
    v = pow(x, 2)
    return v


if __name__ == '__main__':
    # a =[]
    # start = timeit.default_timer()
    # for i in range(1, 10000000):
    #     a.append(do_something(i))
    #
    # end = timeit.default_timer()
    # print('single processing time:', str(end-start), 's')
    # print(a[1:10])
    #
    # # revise to parallel
    # items = [x for x in range(1, 10000000)]
    # p = multiprocessing.Pool(4)
    # start = timeit.default_timer()
    # b = p.map(do_something, items)
    # p.close()
    # p.join()
    # end = timeit.default_timer()
    # print('multi processing time:', str(end-start),'s')
    # print(b[1:10])
    # print('Return values are all equal ?:', operator.eq(a, b))
    # test_sp()
    # mult()
    # test_path()
    # test_rand()
    # test_split()
    # datafile = 'E:\\0学业\\毕设\\useful_dataset\\m-100k\\m1-100k.csv'
    # print(os.path.basename(datafile))
    # print(os.getcwd())
    # print(os.getcwd()+'\\prepare_datasets\\'+ os.path.basename(datafile)+'_train.csv')
    for t in range(5):
        print(t)
