import scipy.sparse as sp
import numpy as np
import prepare_datasets
import multiprocessing
import timeit
import operator
import tensorflow as tf
import os
import re
import math
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

def test_asarray():
    A= [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
    embedding = a = np.asarray(A)
    print(embedding)
    print(a)
    print(A)
    a[1] =1
    print(embedding)
    print(a)
    print(np.transpose(a))
    print(A)

def get_dataset_path(num_dataset):
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'
    Hybird_ = [Hybird,Hybird_train,Hybird_test]
    # test_99_400
    test = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'
    test_ = [test,test_train,test_test]
    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'
    ml_100k_ =[ml_100k,ml_100k_train,ml_100k_test]
    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'
    ml_1m_ = [ml_1m,ml_1m_train,ml_1m_test]
    numbers = {
        0:Hybird_,
        1:test_,
        2:ml_100k_,
        3:ml_1m_
    }
    return numbers.get(num_dataset)

def do_something(x):
    v = pow(x, 2)
    return v
# def get_train_instances(train, neg_ratio):
#     user_input, item_input, labels = [], [], []
#     num_users = train.shape[0]
#     num_items = train.shape[1]
#     for (u, i) in train.keys():
#         # positive instance
#         user_vector_u = sp.dok_matrix.toarray(train)[u]
#         user_vector_i = sp.dok_matrix.transpose(train).toarray()[i]
#         user_input.append(list(user_vector_u))
#         item_input.append(list(user_vector_i))
#         labels.append(train[u, i])
#         # negative instances
#         for t in range(neg_ratio):
#             j = np.random.randint(num_items)
#             while (u, j) in train.keys():
#                 j = np.random.randint(num_items)
#             user_input.append(list(user_vector_u))
#             user_vector_j = sp.dok_matrix.transpose(train).toarray()[j]
#             item_input.append(list(user_vector_j))
#             labels.append(1.0e-6)
#     return user_input, item_input, labels
def get_train_instances(train, neg_ratio):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    print(num_users)
    print(num_items)
    # count=1
    for (u, i) in train.keys():
        # positive instance
        # user_vector_u = sp.dok_matrix.toarray(train)[u]
        # user_vector_i = sp.dok_matrix.transpose(train).toarray()[i]
        # user_input.append(list(user_vector_u))
        # item_input.append(list(user_vector_i))
        user_input.append([u])
        item_input.append([i])
        labels.append([train[u, i]])
        # negative instances
        for t in range(neg_ratio):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            # user_input.append(list(user_vector_u))
            # user_vector_j = sp.dok_matrix.transpose(train).toarray()[j]
            # item_input.append(list(user_vector_j))
            user_input.append([j])
            item_input.append([u])
            labels.append([1.0e-6])
            # count+=1
            # print(count)
    print('加载数据完成！')
    return user_input, item_input, labels

def test_S2():
    a_list = [2,4,5,6]
    b_list = [1,2,3,5]
    ret_list = list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
    s2_result = 1 / (1 + math.exp(-(len(ret_list)) / len(a_list)))
    print(a_list)
    print(b_list)
    print(ret_list)
    print(s2_result)

def test_juzhenchengfa():
    a= np.array([[1,2]])
    b = np.array([[2,1]])
    c=np.array([[3,3]])
    print(a)
    d = a*b*c
    return d
import  heapq
if __name__ == '__main__':
    a=[0,0,0,1,1,2,2,3,3,[1],[1],[0]]

    # each_user_topK_item[userid] = list(map(user_u_vertor.index, heapq.nlargest(TOP_K, user_u_vertor)))
    user_u_vertor=[2,3,4,12,33,5,2,5,6,34,12,3,2,1,2]
    topKlist = list(map(user_u_vertor.index, heapq.nlargest(5, user_u_vertor)))
    nums = [1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]
    max_num = map(nums.index, heapq.nlargest(5, nums))
    print(list(max_num))
    print(topKlist)
    print(a.count(5))
    print(a.count(0))
    print(a.count(1))
    print(a.count([0]))
    print(a.count([1]))
    print(a[0:2])
    print(a[2:4])


    # print(get_dataset_path(1)[0])
