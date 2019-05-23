import tensorflow as tf

# 通过tf.device 将运算指定到指定的设备上。
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='q')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

with tf.device('/gpu:1'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
