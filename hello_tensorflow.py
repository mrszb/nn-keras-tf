import tensorflow as tf
# https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))
