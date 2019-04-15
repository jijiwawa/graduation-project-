import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import os
from Recommendation import Recommendation


# 输入位训练的评分矩阵，以及负样本率。负样本个数=正样本数*neg_ratio
def get_train_instances(train, neg_ratio):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_vector_u = sp.dok_matrix.toarray(train)[u]
        user_vector_i = sp.dok_matrix.transpose(train).toarray()[i]
        user_input.append(user_vector_u)
        item_input.append(user_vector_i)
        labels.append(train[u, i])
        # negative instances
        for t in range(neg_ratio):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(user_vector_u)
            user_vector_j = sp.dok_matrix.transpose(train).toarray()[j]
            item_input.append(user_vector_j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    csv_path = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    # m1-1m
    csv_path_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    # m1-100k
    csv_path_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    # 给出每个隐藏层输出的维度，即降维之后的横坐标
    recommemdation = Recommendation(csv_path_100k)

    layer_dimension = [128, 64]  # 每层隐藏层的输出维度
    batch_size = 256  # batch大小
    learning_rate = 0.0001  # 学习率
    negative_radio = 7  # 一个正样本，7个负样本

    # 隐藏层层数N
    N = len(layer_dimension)

    # 定义权重参数Wh_user,Wh_item
    Wh_user = tf.Variable()
    Wh_item = tf.Variable()

    # 定义输入输出
    user_vertor = tf.placeholder(tf.float32, shape=(batch_size, recommemdation.num_users), name='user_vector-input')
    item_vertor = tf.placeholder(tf.float32, shape=(batch_size, recommemdation.num_items), name='item_vertor-input')
    layer1_user = tf.matmul(user_vertor, W1_u)
    layer1_item = tf.matmul(item_vertor, W1_v)
    for i in range(1, N):
        Wh_user = tf.Variable()
        Wh_item = tf.Variable()

    y = tf.matmul(user_vertor, item_vertor) / (layer_dimension[-1] * layer_dimension[-1])

    # 定义深度神经网络前向传播过程

    # 定义损失函数和反向传播算法
    cross_entropy =
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # 加载训练集,获取输入输出
    user_input, item_input, labels = get_train_instances(recommemdation.trainMatrix, negative_radio)

    # 训练神经网络
    dataset_size = len(user_input)
    with tf.Session() as sess:
        # 参数初始化。
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 迭代更新参数
        STEPS = 1000  # 设定训练轮次。
        for i in range(STEPS):
            # 每次选取batch_size个样本进行训练
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={x: [user_input[start:end], item_input[start:end]], y_: labels[start, end]})
            if i % 100 == 0:
                # 每隔一段时间计算在所有数据上的交叉熵并输出。
                total_cross_entropy = sess.run(cross_entropy,
                                               feed_dict={(user_vertor,item_vertor): (user_input[start:end], item_input[start:end]),
                                                          y: labels[start, end]})
                print("After %d training step(s),cross entropy on all data is %g" % (i,total_cross_entropy))
