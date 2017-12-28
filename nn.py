"""
Artificial Neural Networks - Classification (Diabetes)
"""
from network import *
from numpy import genfromtxt, savetxt
from sys import argv

# instantiate the network
num_of_features = 6

# net = Network([num_of_features, 50, 100, 150, 100, 50, 1])
net = Network([num_of_features, 10, 150, 10, 1])
# net = Network([num_of_features, 50, 1])

# Prepare the training data
train = genfromtxt('data/nn_data/train.csv', delimiter=",")
test = genfromtxt('data/nn_data/test.csv', delimiter=",")

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
weights, biases = net.train(training_data, epochs=6000, momentum_factor=0.9, mini_batch_size=10, lr=lr, check=100, test_data=test_data)

for i in xrange(len(weights)):
	savetxt("params/nn_data/weights/weight_{0}.txt".format(i), weights[i])
	savetxt("params/nn_data/biases/bias_{0}.txt".format(i), biases[i])
