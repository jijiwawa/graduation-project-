import math
import time
import NurtualBaseRecommendation_tf as ntf
from Recommendation import Recommendation
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.sparse as sp
import numpy as np
import os


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
        user_input.append([u])
        item_input.append([i])
        labels.append([test[u, i]])
    print(len(user_input))
    print('加载数据完成！')
    return user_input, item_input, labels


if __name__ == '__main__':
    list_dataset = ntf.get_dataset_path(1)
    recommemdation = Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    modelsavepath = os.getcwd() + '\\model\\' + os.path.basename(list_dataset[0]) + '_model.ckpt'

    # 定义输入输出
    user_index = tf.placeholder(dtype='int32', shape=(None, 1), name='user_index-input')
    item_index = tf.placeholder(dtype='int32', shape=(None, 1), name='item_index-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
    embedding = a = np.asarray(recommemdation.trainMatrix.toarray())
    user_vector = tf.nn.embedding_lookup(embedding, user_index)
    item_vector = tf.nn.embedding_lookup(np.transpose(embedding), item_index)
    user_vector = slim.flatten(user_vector)
    item_vector = slim.flatten(item_vector)
    regularizer = slim.l2_regularizer(0.0001)

    y = ntf.get_inference(user_vector, item_vector, recommemdation,regularizer)

    accuracy = tf.reduce_mean(tf.abs(y - y_))

    user_input_test, item_input_test, labels_test = get_test_instances(recommemdation.testMatrix)
    # user_input_test = [[0]]
    # item_input_test = [[0]]
    # labels_test = [[5]]

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
            print(num_userID)
            end = start + num_userID
            print(end)
            y_predict = sess.run(y, feed_dict={user_index: user_input_test[start:end],
                                               item_index: item_input_test[start:end]})
            print(y_predict)
            accuracy_srocesOf_userI = sess.run(accuracy, feed_dict={user_index: user_input_test[start:end],
                                                                    item_index: item_input_test[start:end],
                                                                    y_: labels_test[start:end]})
            print('用户%s的均方误差%g' % (userID, accuracy_srocesOf_userI))
            sum += accuracy_srocesOf_userI
            print(sum)
            count_user += 1
            print(count_user)
            start = end

        MAE = sum / count_user
        print(MAE)
