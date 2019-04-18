import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.sparse as sp
import numpy as np
import os
from Recommendation import Recommendation


# 获取训练样例
# 输入为训练的评分矩阵，以及负样本率。负样本个数=正样本数*neg_ratio
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
            user_input.append([u])
            labels.append([1.0e-6])
            # count+=1
            # print(count)
    print('加载数据完成！')
    return user_input, item_input, labels


if __name__ == '__main__':
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'

    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'

    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'

    # recommemdation = Recommendation(Hybird, Hybird, Hybird_test)
    recommemdation = Recommendation(ml_100k, ml_100k_train, ml_100k_test)
    # recommemdation = Recommendation(ml_1m,ml_1m_train,ml_1m_test)

    # 给出每个隐藏层输出的维度，即降维之后的横坐标
    num_users = recommemdation.num_users
    num_items = recommemdation.num_items
    layer_dimension = [128, 64]  # 每层隐藏层的输出维度
    batch_size = 128  # batch大小
    learning_rate = 0.0001  # 学习率
    maxR = recommemdation.ratingMax  # 最大评分
    negative_radio = 5  # 一个正样本，7个负样本

    # 隐藏层层数N
    N = len(layer_dimension)

    # 定义模型参数Wh_user,Wh_item
    W1_user = tf.Variable(tf.random_normal((num_items, layer_dimension[0]), mean=0, stddev=0.01))
    W1_item = tf.Variable(tf.random_normal((num_users, layer_dimension[0]), mean=0, stddev=0.01))
    W2_user = tf.Variable(tf.random_normal((layer_dimension[0], layer_dimension[1]), mean=0, stddev=0.01))
    W2_item = tf.Variable(tf.random_normal((layer_dimension[0], layer_dimension[1]), mean=0, stddev=0.01))
    biases1_user = tf.Variable(tf.constant(0.1, shape=[layer_dimension[1]]))
    biases1_item = tf.Variable(tf.constant(0.1, shape=[layer_dimension[1]]))

    # 定义输入输出
    user_index = tf.placeholder(dtype='int32', shape=(None, 1), name='user_index-input')
    item_index = tf.placeholder(dtype='int32', shape=(None, 1), name='item_index-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
    embedding = a = np.asarray(recommemdation.trainMatrix.toarray())
    user_vector = tf.nn.embedding_lookup(embedding, user_index)
    item_vector = tf.nn.embedding_lookup(np.transpose(embedding), item_index)
    user_vector = slim.flatten(user_vector)
    item_vector = slim.flatten(item_vector)

    # 定义深度神经网络前向传播过程
    layer1_user = tf.nn.relu(tf.matmul(user_vector, W1_user))
    layer1_item = tf.nn.relu(tf.matmul(item_vector, W1_item))
    layer2_user = tf.nn.relu(tf.matmul(layer1_user, W2_user) + biases1_user)
    layer2_item = tf.nn.relu(tf.matmul(layer1_item, W2_item) + biases1_item)

    # 结果是一个矩阵？
    y = tf.matmul(layer2_user, tf.transpose(layer2_item)) / (layer_dimension[-1] * layer_dimension[-1])

    # 定义损失函数和反向传播算法 y_ 标准答案 y 预测答案
    # cross_entropy = -tf.reduce_sum((y_ / maxR) * tf.log(y) + (1 - y_ / maxR) * tf.log(1 - y))
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    print('ssss')

    # 加载训练集,获取输入输出
    user_input, item_input, labels = get_train_instances(recommemdation.trainMatrix, negative_radio)

    # 训练神经网络
    dataset_size = len(user_input)
    print(dataset_size)
    with tf.Session() as sess:
        # 参数初始化。
        print('gogog1')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print(sess.run(W1_user))
        print(sess.run(biases1_user))

        # sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]})
        # 迭代更新参数
        STEPS = 10000  # 设定训练轮次。
        for i in range(STEPS):
            # 每次选取batch_size个样本进行训练
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={user_index: np.array(user_input)[start:end],
                                            item_index: np.array(item_input)[start:end],
                                            y_: labels[start:end]})
            print(i)

            print(sess.run(W1_user))
            print(sess.run(biases1_user))
            # sess.run(train_step, feed_dict={user_index: np.array(user_input), item_index: np.array(item_input),
            #                                 y_: np.array(labels)})
            if (i + 1) % 100== 0:
                print('gogogog!!!')
                # 每隔一段时间计算在所有数据上的交叉熵并输出。
                total_cross_entropy = sess.run(cross_entropy,feed_dict={user_index: np.array(user_input)[start:end],
                                            item_index: np.array(item_input)[start:end],
                                            y_: labels[start:end]})
                # total_cross_entropy = sess.run(cross_entropy, feed_dict={user_vector: np.array(user_input),
                #                                                          item_vector: np.array(item_input),
                #                                                          y_: np.array(labels)})
                print(sess.run(W1_user))
                print(sess.run(biases1_user))
                print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
