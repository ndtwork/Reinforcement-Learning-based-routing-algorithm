import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)
print("Available devices:")
print(device_lib.list_local_devices())
