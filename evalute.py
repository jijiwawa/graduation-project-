import scipy.sparse as sp
import numpy as np
import math
import time
import re
from Recommendation import Recommendation, get_dataset_path
import os
import random
import pandas as pd


def Evaluate_HR(preditmatrix, rec, top_k):
    testmatrix = rec.testMatrix
    num_users, num_items = testmatrix.shape
    # 生成每个用户的推荐列表TOP-K { user_id:{item1,item2...},...}
    each_user_topK_item = dict()
    TOP_K = top_k  # 5，10

    for userid in range(0, num_users):
        user_u_vertor = list(preditmatrix[userid])
        if userid not in each_user_topK_item.keys():
            # each_user_topK_item[userid] = list(map(user_u_vertor.index, heapq.nlargest(TOP_K, user_u_vertor)))
            each_user_topK_item[userid] = np.argsort(user_u_vertor)[-TOP_K:]

    # 判断testmatrix中的元素是否在each_user_topK_item中出现
    num_testsample = sp.dok_matrix.count_nonzero(testmatrix)
    # print(preditmatrix)
    # print(each_user_topK_item)
    count = 0
    for (userid, itemid) in testmatrix.keys():
        if testmatrix[userid, itemid] >= rec.user_ave_rating_dict[userid]:
            if userid in each_user_topK_item.keys():
                if itemid in each_user_topK_item[userid]:
                    count += 1
        else:
            num_testsample -= 1
    HR = count / num_testsample
    return HR


def Generate_HR_resultfile(K_start, K_end, K_step, path, path_train, path_test, top_k, dataname):
    rec = Recommendation(path, path_train, path_test)
    K = K_start
    result_file = os.getcwd() + '\\result\\' + dataname + '\\HR_TOP' + str(top_k) + '_' + os.path.basename(
        path) + '.csv'
    with open(result_file, 'w') as result_f:
        if dataname == 'PCC':
            result_f.write('PCC Model for Collaborative Filtering\n')
            filename = 'PCC_predictMatrix'
        if dataname == 'Hybird':
            result_f.write('A Hybrid User Similarity Model for Collaborative Filtering\n')
            filename = 'predictMatrix'
        result_f.write('num_user:%d\nnum_items:%d\nranting:%d\nSparsity level:%.3f\n' % (
            rec.num_users, rec.num_items, rec.num_rating, rec.num_rating / (rec.num_items * rec.num_users)))
        result_f.write("%6.6s\t%6.6s\n" % ('K', 'HR'))
        while K <= K_end:
            preditmatrix_bingxing = np.load(
                os.getcwd() + '\\out_file\\' + dataname + '\\' + filename + '_{}_'.format(K) + os.path.basename(
                    path_train) + '_bingxing.npy')
            hr_result = Evaluate_HR(preditmatrix_bingxing, rec, top_k)
            line = "%6.6s\t%6.6s\n" % (K, str(hr_result))
            result_f.write(line)
            K += K_step

# preditmatrix 满的矩阵
# testmatrix 测试稀疏矩阵
def Evaluate_MAE_AND_NMAE(preditmatrix, testmatrix):
    matrix_sub = sp.dok_matrix.copy(testmatrix)
    num_users, num_items = preditmatrix.shape
    user_add = []
    m = 0
    n = np.zeros(num_users)
    for (userid, itemid) in testmatrix.keys():
        matrix_sub[userid, itemid] = math.fabs(matrix_sub[userid, itemid] - preditmatrix[userid][itemid])
        if userid not in user_add:
            user_add.append(userid)
            m += 1
        n[userid] += 1
    sum = 0
    sum_each_row = np.sum(matrix_sub.toarray(), axis=1)
    for i in range(0, num_users):
        if n[i] != 0:
            sum += sum_each_row[i] / n[i]
    MAE = sum / m
    NMAE = sum / (m * 4)
    # NMAE = sum / (m*5)
    return MAE, NMAE



def MAE_Generate_resultFile(K_start, K_end, K_step, path, path_train, path_test, algorithmname):
    rec = Recommendation(path, path_train, path_test)
    K = K_start
    mae_list = []
    nmae_list = []
    if algorithmname == 'PCC':
        result_file = os.getcwd() + '\\result\\PCC\\MAE_' + os.path.basename(path) + '.csv'
    if algorithmname == 'Hybird':
        result_file = os.getcwd() + '\\result\\Hybird\\resultOfMAE_' + os.path.basename(path) + '.csv'

    with open(result_file, 'w') as result_f:
        if algorithmname == 'PCC':
            result_f.write('PCC Model for Collaborative Filtering\n')
        if algorithmname == 'Hybird':
            result_f.write('A Hybrid User Similarity Model for Collaborative Filtering\n')
        result_f.write('num_user:%d\nnum_items:%d\nranting:%d\nSparsity level:%.3f\n' % (
            rec.num_users, rec.num_items, rec.num_rating, rec.num_rating / (rec.num_items * rec.num_users)))
        result_f.write("%6.6s\t%6.6s\t%6.6s\n" % ('K', 'MAE', 'NMAE'))

        while K <= K_end:
            # pcc
            if algorithmname == 'PCC':
                preditmatrix_bingxing = np.load(
                    os.getcwd() + '\\out_file\\PCC\\PCC_predictMatrix_{}_'.format(K) + os.path.basename(
                        path_train) + '_bingxing.npy')
            # Hybird
            if algorithmname == 'Hybird':
                preditmatrix_bingxing = np.load(
                    os.getcwd() + '\\out_file\\Hybird\\predictMatrix_{}_'.format(K) + os.path.basename(
                        path_train) + '_bingxing.npy')
            MAE_result, NMAE_result = Evaluate_MAE_AND_NMAE(preditmatrix_bingxing, rec.testMatrix)
            mae_list.append(MAE_result)
            nmae_list.append(NMAE_result)
            # "{} {}".format("hello", "world")
            line = "%6.6s\t%6.6s\t%6.6s\n" % (K, str(MAE_result), str(NMAE_result))
            result_f.write(line)
            K += K_step
    return mae_list, nmae_list


if __name__ == '__main__':
    #         0: Hybird_,
    #         1: ml-100-400_,
    #         2: ml_100k_,
    #         3: ml_1m_,
    #         4: pcc_data,
    #         5:ml_200_1000_
    #         6:test99400
    list_dataset = get_dataset_path(2)

    '''
    # MAE NMAE
    name = ['4','8','12','16','20']
    
    ## PCC
    # mae_pcc, nmae_pcc = MAE_Generate_resultFile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 'PCC')
    # Pcc_MAE = pd.DataFrame(columns=name,data=[mae_pcc,nmae_pcc])
    # Pcc_MAE.to_csv(os.getcwd()+'\\result\\PCC\\Pcc_MAE_'+ os.path.basename(list_dataset[0])+'.csv')

    ## Hybird
    mae_hybird, nmae_hybird = MAE_Generate_resultFile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 'Hybird')
    Hybird_MAE = pd.DataFrame(columns=name,data=[mae_hybird,nmae_hybird])
    Hybird_MAE.to_csv(os.getcwd()+'\\result\\Hybird\\Hybird_MAE_'+ os.path.basename(list_dataset[0])+'.csv')
    '''
    ## PMF
    # predictmatrix_pmf= np.load(os.getcwd()+'\\out_file\\PMF\\predictMatrix_m1-200-1000.csv_train.csv_PMF.npy')
    predictmatrix_pmf = np.load(os.getcwd() + '\\out_file\\PMF\\predictMatrix_'+os.path.basename(list_dataset[1])+'_PMF.npy')
    Pmf_mae,Pmf_nmae=Evaluate_MAE_AND_NMAE(predictmatrix_pmf,Recommendation(list_dataset[0], list_dataset[1], list_dataset[2]).testMatrix)
    Pmf_MAE = pd.DataFrame(columns=['pmf_MAE_NMAE'],data=[Pmf_mae,Pmf_nmae])
    Pmf_MAE.to_csv(os.getcwd()+'\\result\\PMF\\Pmf_MAE'+ os.path.basename(list_dataset[0])+'.csv')
    '''
    # HR
    ## PCC
    # Generate_HR_resultfile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 5, 'PCC')
    # Generate_HR_resultfile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 10, 'PCC')
    ## Hybird
    Generate_HR_resultfile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 5, 'Hybird')
    Generate_HR_resultfile(4, 20, 4, list_dataset[0], list_dataset[1], list_dataset[2], 10, 'Hybird')
    '''
    ## Pmf
    rec = Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    predictmatrix_pmf_hr = np.load(
        os.getcwd() + '\\out_file\\PMF\\predictMatrix_' + os.path.basename(list_dataset[1]) + '_PMF.npy')
    print(predictmatrix_pmf_hr)
    for (i,j) in rec.trainMatrix.keys():
         predictmatrix_pmf_hr[i,j] = -1
    print(predictmatrix_pmf_hr)
    Pmf_HR_5 = Evaluate_HR(predictmatrix_pmf_hr, rec, 5)
    Pmf_HR_10 = Evaluate_HR(predictmatrix_pmf_hr, rec, 10)
    list_ = [[str(Pmf_HR_5),str(Pmf_HR_10)]]
    name=['5','10']
    Pcc_MAE = pd.DataFrame(columns=name, data=list_)
    Pcc_MAE.to_csv(os.getcwd() + '\\result\\PMF\\Pmf_HR_' + os.path.basename(list_dataset[0]) + '.csv')
    # '''