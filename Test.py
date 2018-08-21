import tensorflow as tf
from PyQt5 import QtGui

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

print('Hello World')
