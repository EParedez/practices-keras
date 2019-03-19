import sys
import keras as K
import tensorflow as tf

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

print("Python version: " + str(py_ver))
print("Keras version: " + str(k_ver))
print("Tensorflow version: " + str(tf_ver))