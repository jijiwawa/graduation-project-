import math
import time
import DMF as ntf
from Recommendation import Recommendation
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.sparse as sp
import numpy as np
import os
import pandas as pd


def Evaluate_MAE(preditmatrix, testmatrix):
    matrix_sub = sp.dok_matrix.copy(testmatrix)
    num_users, num_items = preditmatrix.shape
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


# 获取测试样例，评测模型准确性
def get_test_instances(test):
    user_input, item_input, labels = [], [], []
    num_users = test.shape[0]
    num_items = test.shape[1]
    print(num_users)
    print(num_items)
    for (u, i) in test.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(test[u, i])
    print(len(user_input))
    print('加载数据完成！')
    return user_input, item_input, labels


if __name__ == '__main__':
    list_dataset = ntf.get_dataset_path(1)
    recommemdation = Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    modelsavepath = os.getcwd() + '\\model\\' + os.path.basename(list_dataset[0]) + '_model.ckpt'
    num_users, num_items = recommemdation.num_users, recommemdation.num_items
    # 定义输入输出
    # 用的就是训练矩阵预测测试的值
    embedding = a = np.asarray(recommemdation.trainMatrix.toarray())
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('input_vector_user'):
        user_index = tf.placeholder(dtype='int32', shape=(None), name='user_index-input')
        user_vector = tf.nn.embedding_lookup(embedding, user_index)
    with tf.name_scope('input_vector_item'):
        item_index = tf.placeholder(dtype='int32', shape=(None), name='item_index-input')
        item_vector = tf.nn.embedding_lookup(np.transpose(embedding), item_index)
    with tf.name_scope('y_labels'):
        y_ = tf.placeholder(tf.float32, shape=(None), name='y-input')
    regularizer = slim.l2_regularizer(0.0001)
    y = ntf.get_inference(user_vector, item_vector, recommemdation, regularizer, keep_prob)

    accuracy = tf.reduce_mean(tf.abs(y - y_))

    user_input_test, item_input_test, labels_test = get_test_instances(recommemdation.testMatrix)

    # MAE
    # '''
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(modelsavepath + '.meta')
        saver = tf.train.Saver()
        saver.restore(sess, modelsavepath)
        # saver.restore(sess, tf.train.latest_checkpoint('./'))

        start = 0
        sum = 0
        count_user = 0
        while start < len(user_input_test):
            userID = user_input_test[start]
            print(userID)
            num_userID = user_input_test.count(userID)
            print('batch_size',num_userID)
            end = start + num_userID
            print(end)
            y_predict = sess.run(y, feed_dict={user_index: user_input_test[start:end],
                                               item_index: item_input_test[start:end],keep_prob:1})
            print(y_predict)
            print(labels_test[start:end])
            accuracy_srocesOf_userI = sess.run(accuracy, feed_dict={user_index: user_input_test[start:end],
                                                                    item_index: item_input_test[start:end],
                                                                    y_: labels_test[start:end],keep_prob:1})
            print('用户%s的均方误差%g' % (userID, accuracy_srocesOf_userI))
            sum += accuracy_srocesOf_userI
            print(sum)
            count_user += 1
            print(count_user)
            start = end

        MAE = sum / count_user
        NMAE =MAE/4
        Dmf_mae, Dmf_nmae = MAE,NMAE
        Dmf_MAE = pd.DataFrame(columns=['Dmf_MAE_NMAE'], data=[[Dmf_mae],[Dmf_nmae]])
        Dmf_MAE.to_csv(os.getcwd() + '\\result\\DMF\\Dmf_MAE' + os.path.basename(list_dataset[0]) + '.csv')
    # '''
    # HR 生成预测评分矩阵
    # '''
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph(modelsavepath + '.meta')
        saver = tf.train.Saver()
        saver.restore(sess, modelsavepath)
        predict_matrix = np.full((num_users, num_items), -1,dtype=np.float32)
        print(predict_matrix)
        train_matrix = recommemdation.trainMatrix

        for userid in range(num_users):
            for itemid in range(num_items):
                print('当前计算（%d,%d)'%(userid,itemid))
                if train_matrix[userid, itemid] == 0 or train_matrix[userid, itemid] == None:
                    predict_matrix[userid][itemid] = sess.run(y, feed_dict={user_index: np.array([userid]),
                                                                            item_index: np.array([itemid]),
                                                                            keep_prob: 1})

        print(predict_matrix)
        np.save(os.getcwd() + '\\out_file\\DMF\\predictMatrix_' + os.path.basename(list_dataset[1]) + '_DMF.npy',
                predict_matrix)
    # '''