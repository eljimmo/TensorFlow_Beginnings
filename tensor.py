import tensorflow as tf


if __name__=='__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
