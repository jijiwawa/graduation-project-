import scipy.sparse as sp
import numpy as np
import math
import time
import re
from Recommendation import Recommendation
import os
import heapq
import random


def Evaluate_HR(preditmatrix, testmatrix, top_k):
    # 每个用户在测试集上商品个数
    num_users, num_items = testmatrix.shape
    # 生成每个用户的推荐列表TOP-5 { user_id:{item1,item2...},...}
    each_user_topK_item = dict()
    TOP_K = top_k  # 5，10

    for userid in range(0, num_users):
        user_u_vertor = list(preditmatrix[userid])
        if userid not in each_user_topK_item.keys():
            each_user_topK_item[userid] = list(map(user_u_vertor.index, heapq.nlargest(TOP_K, user_u_vertor)))

    # 判断testmatrix中的元素是否在each_user_topK_item中出现
    num_testsample = sp.dok_matrix.count_nonzero(testmatrix)
    num_predict_in_testsample = np.zeros(num_users)
    print(preditmatrix)
    print(each_user_topK_item)
    count = 0
    for (userid, itemid) in testmatrix.keys():
        if userid in each_user_topK_item.keys():
            if itemid in each_user_topK_item[userid]:
                count += 1
    HR = count / num_testsample
    return HR

def HR_Generate_result( path, path_train, path_test,top_k):
    preditmatrix_bingxing = np.load(
        os.getcwd() + '\\out_file\\predictMatrix_{}_'.format(8) + os.path.basename(
            path_train) + '_bingxing.npy')
    rec = Recommendation(path, path_train, path_test)

    return Evaluate_HR(preditmatrix_bingxing,rec.testMatrix,top_k)

def Evaluate_MAE_AND_NMAE(preditmatrix, testmatrix):
    matrix_sub = sp.dok_matrix.copy(testmatrix)
    num_users, num_items = preditmatrix.shape
    userid1 = -1
    m = 0
    n = np.zeros(num_users)
    for (userid, itemid) in testmatrix.keys():
        matrix_sub[userid, itemid] = math.fabs(matrix_sub[userid, itemid] - preditmatrix[userid][itemid])
        if userid != userid1:
            m += 1
        n[userid] += 1
        userid1 = userid
    sum = 0
    sum_each_row = np.sum(matrix_sub.toarray(), axis=1)
    for i in range(0, num_users):
        if n[i] != 0:
            sum += sum_each_row[i] / n[i]
    MAE = sum / m
    NMAE = sum / (m * 4)
    # NMAE = sum / (m*5)
    return MAE, NMAE


def MAE_Generate_resultFile(K_start, K_end, K_step, path, path_train, path_test):
    rec = Recommendation(path, path_train, path_test)
    K = K_start
    result_file = os.getcwd() + '\\result\\resultOfMAE_' + os.path.basename(path) + '.csv'
    with open(result_file, 'w') as result_f:
        result_f.write('A Hybrid User Similarity Model for Collaborative Filtering\n')
        result_f.write('num_user:%d\nnum_items:%d\nranting:%d\nSparsity level:%.3f\n' % (
            rec.num_users, rec.num_items, rec.num_rating, rec.num_rating / (rec.num_items * rec.num_users)))
        result_f.write('{}'.format('K') + '\t' + 'MAE' + '\t' + 'NMAE' + '\n')
        while K <= K_end:
            preditmatrix_bingxing = np.load(
                os.getcwd() + '\\out_file\\predictMatrix_{}_'.format(K) + os.path.basename(
                    path_train) + '_bingxing.npy')
            MAE_result, NMAE_result = Evaluate_MAE_AND_NMAE(preditmatrix_bingxing, rec.testMatrix)
            # "{} {}".format("hello", "world")
            line = '{}'.format(K) + '\t' + str(MAE_result) + '\t' + str(NMAE_result) + '\n'
            result_f.write(line)
            K += K_step


if __name__ == '__main__':
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'

    # test_99_400
    test = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'

    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'

    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'

    # MAE_Generate_resultFile(4, 20, 4, test, test_train, test_test)

    print(HR_Generate_result(test, test_train, test_test,5))
    print(HR_Generate_result(test, test_train, test_test,10))

    # MAE_Generate_resultFile(20, 20, 20, ml_100k, ml_100k_train, ml_100k_test)
