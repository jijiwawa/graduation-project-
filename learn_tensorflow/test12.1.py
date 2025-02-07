import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='q')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')

c = a + b
# 通过log_device_placement参数来输出运行没一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
