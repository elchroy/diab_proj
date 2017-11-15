"""
Artificial Neural Networks - Classification (Appendicitis)
"""
from network import *
from numpy import *#genfromtxt, savetxt
from sys import argv

# instantiate the network
num_of_features = 7
net = Network([num_of_features, 3000, 30, 1])

# Prepare the training data
train = genfromtxt('appendicitis_train.csv', delimiter=",")
test = genfromtxt('appendicitis_test.csv', delimiter=",")

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
