"""
Artificial Neural Networks - Classification (Diabetes)
"""
from network import *
from numpy import genfromtxt, savetxt
from sys import argv

# instantiate the network
num_of_features = 8
net = Network([num_of_features, 100, 10, 1])

# Prepare the training data
train = genfromtxt('data/diabetes/train.csv', delimiter=",")
test = genfromtxt('data/diabetes/test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

x_test = test[:, 0:num_of_features]
y_test = test[:, [num_of_features]]

training_data = zip(x_train, y_train)
test_data = zip(x_test, y_test)

if len(argv) > 1:
	lr = float(argv[1])
else:
	lr = 0.03

# Train the neural network
weights, biases = net.train(training_data, epochs=1000, momentum_factor=0.3, mini_batch_size=10, lr=lr, check=100, test_data=test_data)

for i in xrange(len(weights)):
	savetxt("params/weights/weight_{0}.txt".format(i), weights[i])
	savetxt("params/biases/bias_{0}.txt".format(i), biases[i])
