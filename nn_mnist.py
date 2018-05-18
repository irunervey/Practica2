import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
entrenar_x=train_x[ 0:35000, ]
validar_x=train_x[ 35000:42500,]
test_x=train_x[ 42500:50000,]
entrenar_y=train_y[ 0:35000,]
validar_y=train_y[ 35000:42500,]
test_y=train_y[ 42500:50000,]

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
entrenar_y=one_hot(entrenar_y, 10);
validar_y=one_hot(validar_y, 10);
test_y=one_hot(test_y, 10);
print (train_y[0])


# TODO: the neural net!!

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 40)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(40)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(40, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")
validationerror=[]
batch_size = 40
error=50
error1=45
epoch=1
validar_xs = validar_x [0:40]
validar_ys = validar_y [0:40]
while (abs(error - error1)>0.001):
    for jj in range(int(len(entrenar_x) / batch_size)):
        batch_xs = entrenar_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = entrenar_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error=error1
    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: validar_xs, y_: validar_ys}))
    epoch=epoch+1
    error1=sess.run(loss, feed_dict={x: validar_xs, y_: validar_ys})
    validationerror.append(error1)
result = sess.run(y, feed_dict={x: test_x})
errorcito=sess.run(loss, feed_dict={x: test_x, y_: test_y})
print("Error",errorcito)
exito = 100 - (error * 100 / 10000)
print("Exito ",exito,"%")
errorTest =0;
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        errorTest += 1
print(b, "-->", r, "Cantidad de errores en el test: ", errorTest)
plt.ylabel('Errors')
plt.xlabel('Epochs')
training_line, = plt.plot(validationerror)
plt.savefig('grafica.png')


"""

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print train_y[57]


# TODO: the neural net!!


"""


