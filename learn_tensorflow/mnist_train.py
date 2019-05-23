# -*- coding:utf-8 -*-
import os
import tensorflow.contrib.slim as slim

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数。
from learn_tensorflow import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = os.getcwd()
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出placeholder。
    # 将处理输入数据的计算都放在名字为“input”的命名空间下。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = slim.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程。
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 和5.2.1节样例中类似地定义损失函数、学习率、滑动平均操作以及训练过程。
    # 将处理滑动平均相关的计算都放在名为moving_average的命名空间下。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 将计算损失函数相关的计算都放在名为loss_function的命名空间下。
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型数据上的表现，验证和测试的过程将会有一个独
        # 立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 每1000轮保存一次模型。
            if i % 1000 == 0:
                # 配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata= tf.RunMetadata()
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys},options=run_options,run_metadata=run_metadata)

                writer = tf.summary.FileWriter("tensorflow_log/path1/log", tf.get_default_graph())
                writer.close()
                print("After %d training step(s),loss on training batch is %g." % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被
                # 保存模型的文件名末尾加上训练的轮次，比如"model.ckpt-1000"表示
                # 训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})


def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/mnist_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
