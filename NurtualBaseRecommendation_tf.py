import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.sparse as sp
import numpy as np
import os
import time
import evalute_tf
from Recommendation import Recommendation

config = tf.ConfigProto(allow_soft_placement=True)


def get_dataset_path(num_dataset):
    # test数据集合
    Hybird = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv'
    Hybird_train = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_train.csv'
    Hybird_test = os.getcwd() + '\\prepare_datasets\\Hybird_data.csv_test.csv'
    Hybird_ = [Hybird, Hybird_train, Hybird_test]
    # test_99_400
    test = os.getcwd() + '\\prepare_datasets\\test_99_400.base'
    test_train = os.getcwd() + '\\prepare_datasets\\test_99_400.base_train.csv'
    test_test = os.getcwd() + '\\prepare_datasets\\test_99_400.base_test.csv'
    test_ = [test, test_train, test_test]
    # m1-100k
    ml_100k = os.getcwd() + '\\prepare_datasets\\m1-100k.csv'
    ml_100k_train = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_train.csv'
    ml_100k_test = os.getcwd() + '\\prepare_datasets\\m1-100k.csv_test.csv'
    ml_100k_ = [ml_100k, ml_100k_train, ml_100k_test]
    # m1-1m
    ml_1m = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_train = os.getcwd() + '\\prepare_datasets\\ml-1m.train.rating'
    ml_1m_test = os.getcwd() + '\\prepare_datasets\\ml-1m.test.rating'
    ml_1m_ = [ml_1m, ml_1m_train, ml_1m_test]
    numbers = {
        0: Hybird_,
        1: test_,
        2: ml_100k_,
        3: ml_1m_
    }
    return numbers.get(num_dataset)


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
            user_input.append([u])
            item_input.append([j])
            labels.append([0])
            # count+=1
            # print(count)
    print(len(user_input))
    print('加载数据完成！')
    return user_input, item_input, labels


def get_Userweight_variable(shape, regularizer):
    weights_user = tf.get_variable("weights_user", shape,
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights_user))
    return weights_user


def get_Itemweight_variable(shape, regularizer):
    weights_item = tf.get_variable("weights_item", shape,
                                   initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights_item))
    return weights_item


# 定义神经网络模型
def get_inference(user_vector, item_vector, rc, regularizer):
    layer_dimension = [128, 64]  # 每层隐藏层的输出维度
    # 定义模型参数Wh_user,Wh_item,biases
    # W1_user = tf.Variable(tf.random_normal((rc.num_items, layer_dimension[0]), mean=0, stddev=0.01))
    # W1_item = tf.Variable(tf.random_normal((rc.num_users, layer_dimension[0]), mean=0, stddev=0.01))
    # W2_user = tf.Variable(tf.random_normal((layer_dimension[0], layer_dimension[1]), mean=0, stddev=0.01))
    # W2_item = tf.Variable(tf.random_normal((layer_dimension[0], layer_dimension[1]), mean=0, stddev=0.01))
    # biases1_user = tf.Variable(tf.constant(0.0, shape=[layer_dimension[1]]))
    # biases1_item = tf.Variable(tf.constant(0.0, shape=[layer_dimension[1]]))

    # 定义深度神经网络前向传播过程
    with tf.variable_scope('layer1'):
        W1_user = get_Userweight_variable([rc.num_items, layer_dimension[0]], regularizer)
        W1_item = get_Itemweight_variable([rc.num_users, layer_dimension[0]], regularizer)
        layer1_user = tf.nn.relu(tf.matmul(user_vector, W1_user))
        layer1_item = tf.nn.relu(tf.matmul(item_vector, W1_item))
    with tf.variable_scope('layer2'):
        W2_user = get_Userweight_variable([layer_dimension[0], layer_dimension[1]], regularizer)
        W2_item = get_Itemweight_variable([layer_dimension[0], layer_dimension[1]], regularizer)
        layer2_user = tf.nn.relu(tf.matmul(layer1_user, W2_user))
        layer2_item = tf.nn.relu(tf.matmul(layer1_item, W2_item))
    # layer2_user = tf.nn.relu(tf.matmul(layer1_user, W2_user) + biases1_user)
    # layer2_item = tf.nn.relu(tf.matmul(layer1_item, W2_item) + biases1_item)

    # 结果
    y1 = tf.matmul(layer2_user, tf.transpose(layer2_item)) / (layer_dimension[-1] * layer_dimension[-1])
    y = tf.clip_by_value(y1, 1.0e-6, 5.0)
    return y


if __name__ == '__main__':
    # 获取数据集路径，0-Hybird，1-test99_400,2-ml_100k,3-ml_1m
    list_dataset = get_dataset_path(1)
    recommemdation = Recommendation(list_dataset[0], list_dataset[1], list_dataset[2])
    modelsavepath = os.getcwd() + '\\model\\' \
                    + os.path.basename(list_dataset[0]) + '_model.ckpt'

    # 给出每个隐藏层输出的维度，即降维之后的横坐标
    num_users = recommemdation.num_users
    num_items = recommemdation.num_items

    batch_size = 256  # batch大小
    learning_rate = 0.0001  # 学习率
    maxR = recommemdation.ratingMax  # 最大评分
    negative_radio =5  # 一个正样本，7个负样本
    layer_dimension = [128, 64]  # 每层隐藏层的输出维度

    # 隐藏层层数N
    N = len(layer_dimension)

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
    y = get_inference(user_vector, item_vector, recommemdation, regularizer)

    # 定义损失函数和反向传播算法 y_ 标准答案 y 预测答案
    # cross_entropy = -tf.reduce_sum((y_ / maxR) * tf.log(y) + (1 - y_ / maxR) * tf.log(1 - y))
    # ce
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))+ tf.add_n(
        tf.get_collection('losses'))
    # 论文提出的损失函数 nce
    n_cross_entropy = -tf.reduce_sum(y_ / maxR * tf.log(y) + (1 - y_ / maxR) * tf.log(1 - y)) + tf.add_n(
        tf.get_collection('losses'))

    accuracy = tf.reduce_mean(tf.abs(y - y_))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    print('ssss')

    # 加载训练集,获取输入输出
    user_input, item_input, labels = get_train_instances(recommemdation.trainMatrix, negative_radio)
    user_input_test, item_input_test, labels_test = evalute_tf.get_test_instances(recommemdation.testMatrix)

    # 训练神经网络
    dataset_size = len(user_input)
    print(dataset_size)
    with tf.Session(config=config) as sess:
        # 参数初始化。
        print('gogog1')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 声明Saver类用于保存模型
        saver = tf.train.Saver()
        # sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]})
        # 迭代更新参数
        STEPS = 200  # 设定训练轮次。
        for i in range(STEPS):
            k = 0
            end = 0
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            while end < dataset_size:
                # 每次选取batch_size个样本进行训练
                start = (k * batch_size) % dataset_size
                end = min(start + batch_size, dataset_size)
                # print(str(start) + ',' + str(end))
                sess.run(train_step, feed_dict={user_index: user_input[start:end],
                                                item_index: item_input[start:end],
                                                y_: labels[start:end]})
                k += 1

            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            # sess.run(train_step, feed_dict={user_index: np.array(user_input), item_index: np.array(item_input),
            #                                 y_: np.array(labels)})
            # 每隔一段时间计算在所有数据上的交叉熵并输出。
            total_cross_entropy = sess.run(cross_entropy, feed_dict={user_index: user_input[start:end],
                                                                       item_index: item_input[start:end],
                                                                       y_: labels[start:end]})
            accuracy_sroce = sess.run(accuracy, feed_dict={user_index: user_input[start:end],
                                                                       item_index: item_input[start:end],
                                                                       y_: labels[start:end]})
            '''
            {user_index: user_input_test,
             item_index: item_input_test,
             y_: labels_test}
            '''
            print("After %d training step(s),cross entropy on all data is %g\naccuracy_sroce is %g" % (
                i, total_cross_entropy, accuracy_sroce))
        saver.save(sess, modelsavepath)

        '''
        print("评估模型!!!")
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state函数会通过checkpoint文件
            # 自动找到目录中最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(modelsavepath)
            saver.restore(sess, modelsavepath)
            accuracy_score = sess.run(accuracy, feed_dict={user_index: user_input_test,
                                                           item_index: item_input_test,
                                                           y_: labels_test})
            y_prediction = sess.run(y, feed_dict={user_index: user_input_test,
                                                  item_index: item_input_test,
                                                  y_: labels_test})
            print(y_prediction.shape)
            print(len(y_prediction))
            print(labels_test.shape)
            print("After %s training step(s), validation accuracy = %g" % (1, accuracy_score))
'''
