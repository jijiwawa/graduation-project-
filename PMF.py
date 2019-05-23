# -*- coding: utf-8 -*-
from __future__ import print_function

import scipy.sparse as sp
import scipy.stats as stats
import heapq
import numpy as np
from numpy.random import RandomState
import math
import time
import os
import re
import copy
from Recommendation import Recommendation, get_dataset_path


class PMF(Recommendation):
    def __init__(self, file, train_file, test_file, latent_size):
        super(PMF, self).__init__(file, train_file, test_file)
        # 一些参数
        self.latent_size = latent_size
        self.lambda_alpha = 0.01
        self.lambda_beta = 0.01
        self.iterations = 2000
        # self.lr = 3e-5
        self.lr = 7e-6
        # self.lr = 5e-3

        self.momentum = 0.9

        self.random_state = RandomState(1)
        self.I = copy.deepcopy(self.trainMatrix)
        self.I[self.I != 0] = 1
        #  Initialize factor parameters U0 ,V0
        # self.U = np.array([[1,0,0],
        #                    [0,1,0],
        #                    [0,0,1]])
        # self.V = np.array([[1,1,1],
        #                    [2,2,2],
        #                    [3,3,3],
        #                    [4,4,4],
        #                    [5,5,5]])
        self.U = 0.1 * self.random_state.rand(np.size(self.trainMatrix, 0), self.latent_size)
        self.V = 0.1 * self.random_state.rand(np.size(self.trainMatrix, 1), self.latent_size)
        # self.U = stats.norm(sp.(self.trainMatrix, 0), self.latent_size)
        # self.V = stats.norm(np.size(self.trainMatrix, 1), self.latent_size)
        self.train()
        # 预测评分矩阵
        print('预测评分矩阵：')
        self.predict_matrix = np.dot(self.U, self.V.T)
        np.save(os.getcwd() + '\\out_file\\PMF\\predictMatrix_' + os.path.basename(train_file) + '_PMF.npy',
                self.predict_matrix)
        self.predict_matrix = np.load(
            os.getcwd() + '\\out_file\\PMF\\predictMatrix_' + os.path.basename(train_file) + '_PMF.npy')
        print(self.predict_matrix)

    def loss(self):
        # the loss function of the model
        loss = np.sum(self.I.toarray() * (
                self.trainMatrix.toarray() - np.dot(self.U, self.V.T)) ** 2) + self.lambda_alpha * np.sum(
            np.square(self.U)) + self.lambda_beta * np.sum(np.square(self.V))
        return loss

    def predict_test(self):
        index_data = np.array(list(self.testMatrix.keys()))
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        predict_value = np.sum(u_features * v_features, 1)
        '''
        print(predict_value)
        for i in range(len(predict_value)):
            predict_value[i] = 1 / (1+math.exp(predict_value[i]))
        print(predict_value)

        '''

        return predict_value

    def train(self):

        train_loss_list = []
        test_rmse_list = []
        last_test_rmse = None

        # momentum
        momentum_u = np.zeros(self.U.shape)
        momentum_v = np.zeros(self.V.shape)

        for it in range(self.iterations):
            # derivate of Vi
            grads_u = np.dot(self.I.toarray() * (self.trainMatrix.toarray() - np.dot(self.U, self.V.T)),
                             -self.V) + self.lambda_alpha * self.U

            # derivate of Tj
            grads_v = np.dot((self.I.toarray() * (self.trainMatrix.toarray() - np.dot(self.U, self.V.T))).T,
                             -self.U) + self.lambda_beta * self.V

            # update the parameters
            momentum_u = (self.momentum * momentum_u) + self.lr * grads_u
            momentum_v = (self.momentum * momentum_v) + self.lr * grads_v
            self.U = self.U - momentum_u
            self.V = self.V - momentum_v

            # training evaluation
            train_loss = self.loss()
            train_loss_list.append(train_loss)

            test_preds = self.predict_test()
            test_rmse = self.RMSE(test_preds, np.array(list(self.testMatrix.values())))
            test_rmse_list.append(test_rmse)

            print('traning iteration:{: d} ,loss:{: f}, test_rmse:{: f}'.format(it, train_loss, test_rmse))

            if last_test_rmse and (last_test_rmse - test_rmse) <= 0:
                print('convergence at iterations:{: d}'.format(it))
                break
            else:
                last_test_rmse = test_rmse

        return self.U, self.V, train_loss_list, test_rmse_list

    def RMSE(self, preds, truth):
        # return np.sqrt(np.mean(np.square(preds - truth)))
        return np.mean(np.abs(preds - truth))

        # tf.reduce_mean(tf.abs(y - y_))

    def Evaluate_MAE_AND_NMAE(self):
        matrix_sub = sp.dok_matrix.copy(self.testMatrix)
        user_add = []
        m = 0  # 用户数
        num_item_each_user = np.zeros(self.num_users)
        for (userid, itemid) in self.testMatrix.keys():
            matrix_sub[userid, itemid] = math.fabs(matrix_sub[userid, itemid] - self.predict_matrix[userid][itemid])
            if userid not in user_add:
                user_add.append(userid)
                m += 1
            num_item_each_user[userid] += 1
        sum = 0
        sum_each_row = np.sum(matrix_sub.toarray(), axis=1)
        for i in range(0, self.num_users):
            if num_item_each_user[i] != 0:
                sum += sum_each_row[i] / num_item_each_user[i]
        MAE = sum / m
        NMAE = sum / (4 * m)
        return MAE, NMAE


if __name__ == '__main__':
    #         0: Hybird_,
    #         1: test_,
    #         2: ml_100k_,
    #         3: ml_1m_,
    #         4: pcc_data,
    #         5:ml_200_1000_
    list_dataset = get_dataset_path(2)
    recommemdation = PMF(list_dataset[0], list_dataset[1], list_dataset[2], 128)
    # count=1
    # mae,nmae=recommemdation.Evaluate_MAE_AND_NMAE()
    # print(mae)
    # print(nmae)
    # for i in range(99):
    #     for j in range(400):
    #         if matrix[i,j] >=5.0:
    #             print(matrix[i,j])
    #             count+=1
    #             break
    # print(count)
