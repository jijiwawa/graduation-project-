import tensorflow as tf

# 在CPU上运行tf.Variable
a_cpu = tf.Variable(0, name="a_cpu")

with tf.device('/gpu:0'):
    # 将tf.Variable强制放在GPU上。
    a_gpu = tf.Variable(0, name="a_gpu")

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables)
