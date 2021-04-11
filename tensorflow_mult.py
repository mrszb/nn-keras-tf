import tensorflow as tf
from tensorflow.python.client import device_lib

tf.compat.v1.disable_eager_execution()


def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']


print("tf version = ", tf.__version__)
print (device_lib.list_local_devices())

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

gpus = get_available_gpus()
print (gpus)

 
with tf.device('/CPU:0'):
#with tf.device('/XLA_CPU:0'):
#with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)
    

with tf.compat.v1.Session() as sess:
	print (sess.run(c))

# result should be 
# [[22. 28.]
# [49. 64.]]