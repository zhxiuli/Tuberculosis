
##tensorflow，keras

import pandas as pd
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


df = pd.read_csv("feature.txt")
y = df.ix[:,-1].values
x = df.ix[:,:99].as_matrix()

x_train_all, x_test, y_train_all, y_test = train_test_split(x, y)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_vaild_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, activation = 'relu', input_shape = x_train.shape[1:]),
     tf.keras.layers.Dense(30, activation = 'relu'),
     tf.keras.layers.Dense(30, activation = 'relu'),
	 tf.keras.layers.Dense(30, activation = 'relu'),
     tf.keras.layers.Dense(30, activation = 'relu'),
     tf.keras.layers.Dense(3, activation = 'softmax'),
)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 200
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_valid, y_valid), verbose=1, EarlyStopping(monitor='loss', patience=3)
#防止过拟合

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'acc')
plot_graphs(history, 'loss')



#tensorflow related methods
 
# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
 
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
 
sess = tf.InteractiveSession()
 
 
# Create the model
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
 
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
 
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
 
 
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
##train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
 
 
# Train
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x:batch_xs, y_:batch_ys, keep_prob:0.75})
 
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


model.compile(optimizer = 'adam', loss = 'mse')
callbacks = [keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-3)]

history = model.fit(x_train_scaled, y_train, validation_data = (x_vaild_scaled, y_valid), epochs = 200, callbacks = callbacks)

def plot_learning_curver(history):
    pd.DataFrame(history.history).plot(figsize = (10, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

plot_learning_curver(history)

model.evaluate(x_test_scaled, y_test, verbose=0)




#theano

import numpy as np
import theano
import theano.tensor as T

def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

rng = np.random

N = 400             # training 数据个数
feats = 784         # input 的 feature 数

# 生成随机数: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

x = T.dmatrix("x")
y = T.dvector("y")

W = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

p_1 = T.nnet.sigmoid(T.dot(x, W) + b)   # sigmoid 激励函数
prediction = p_1 > 0.5                  
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # 交叉熵

# xent 同效
# xent = T.nnet.binary_crossentropy(p_1, y) 

cost = xent.mean() + 0.01 * (W ** 2).sum()  # l2 正则化
gW, gb = T.grad(cost, [W, b])     

learning_rate = 0.1
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent.mean()],
          updates=((W, W - learning_rate * gW), (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Training
for i in range(500):
    pred, err = train(D[0], D[1])
    if i % 50 == 0:
        print('cost:', err)
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))
		
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

"""
cost: 11.677533008109728
accuracy: 0.52
cost: 6.1946164642562636
accuracy: 0.6175
cost: 3.012375762498935
accuracy: 0.725
cost: 1.3340537876600198
accuracy: 0.8275
cost: 0.4690120202455575
accuracy: 0.9075
...


target values for D:
[1 1 0 1 0 1 0 1 1 1 1 1 .....]

prediction on D:
[1 1 0 1 0 1 0 1 1 1 1 1 .....]
"""



#MLPClassifier

#coding=utf-8


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
import gzip

# 加载数据
# mnist = fetch_mldata("MNIST original")
with gzip.open("D:\\xxx\\mnist.pkl.gz") as fp:
    training_data,valid_data,test_data = pickle.load(fp)
x_training_data,y_training_data = training_data
x_valid_data,y_valid_data = valid_data
x_test_data,y_test_data = test_data
classes = np.unique(y_test_data)


x_training_data_final = np.vstack((x_training_data,x_valid_data))
y_training_data_final = np.append(y_training_data,y_valid_data)


mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
# 使用solver='lbfgs',准确率为79%，比较适合少于几千的数据集

mlp.fit(x_training_data_final, y_training_data_final) 

print mlp.score(x_test_data,y_test_data)
print mlp.n_layers_
print mlp.n_iter_
print mlp.loss_
print mlp.out_activation_

