from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

def import_data(filepath, app_length) :
	mdata = MNIST(filepath)

	data, label_temp = mdata.load_training()

	data = np.array(data[0:app_length])
	label_vector = np.array(label_temp[0:app_length])

	output = []

	for i in range(app_length) :
		index = label_vector[i]
		arr = np.zeros(10)
		arr[index] = 1
		output.append(arr)
	
	label = np.array(output, dtype=np.int)

	return data, label, label_vector

def backprop(batch_index) :

	global w1, w2, w3, b1, b2, b3

	# Feedforeward

	start = batch_index * batch_size
	end = (batch_index + 1) * batch_size

	data_batch = data[start:end, :]
	label_batch = label[start:end, :]

	z1 = data_batch.dot(w1) + b1
	a1 = sigmoid(z1)

	z2 = a1.dot(w2) + b2 
	a2 = sigmoid(z2)

	z3 = a2.dot(w3) + b3
	a3 = sigmoid(z3)

	# Backpropagation

	delta = (a3 - label_batch) * sigmoid_prime(z3)
	nabla_b3 = delta.sum(axis=0) / batch_size
	nabla_w3 = np.dot(a2.T, delta)

	delta = np.dot(delta, w3.T) * sigmoid_prime(z2)
	nabla_b2 = delta.sum(axis=0) / batch_size
	nabla_w2 = np.dot(delta.T, a1)

	delta = np.dot(delta, w2.T) * sigmoid_prime(z1)
	nabla_b1 = delta.sum(axis=0) / batch_size
	nabla_w1 = np.dot(delta.T, data_batch).T

	# Ajust weights and biases

	w1 -= step_size * nabla_w1
	b1 -= step_size * nabla_b1

	w2 -= step_size * nabla_w2
	b2 -= step_size * nabla_b2

	w3 -= step_size * nabla_w3
	b3 -= step_size * nabla_b3

def sigmoid(x) :
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x) :
	return sigmoid(x) * (1 - sigmoid(x))

def log_precision(i) :
	z1 = data.dot(w1) + b1
	a1 = sigmoid(z1)

	z2 = a1.dot(w2) + b2 
	a2 = sigmoid(z2)

	z3 = a2.dot(w3) + b3
	a3 = sigmoid(z3)

	pred_label = a3

	pred_vector = np.argmax(pred_label, axis=1)
	result = pred_vector - label_vector
	count = np.count_nonzero(result==0)

	loss = np.square(pred_label - label).sum() / (10 * data_size)
	percent = round(100 * count / data_size, 2)

	learning_rate.append(percent)

	print(str(i) + " - " + str(percent) + "% - " + str(round(loss, 5)))

def plot_graph() :
	plt.plot(learning_rate)
	plt.ylabel('Learning rate')
	plt.ylim((0, 100))
	plt.show()

def random_matrix(rows, cols) :
	return 2 * np.random.random((rows, cols)) - 1

''' Script '''

input_size = 784
hidden_1_size = 16
hidden_2_size = 16
output_size = 10

backprop_steps = 100000
data_size = 60000
batch_count = 400
batch_size = int(data_size / batch_count)
step_size = 0.0005

# Init

data, label, label_vector = import_data("MNIST-data", data_size)

w1 = random_matrix(input_size, hidden_1_size)
b1 = random_matrix(1, hidden_1_size)

w2 = random_matrix(hidden_1_size, hidden_2_size)
b2 = random_matrix(1, hidden_2_size)

w3 = random_matrix(hidden_2_size, output_size)
b3 = random_matrix(1, output_size)

learning_rate = []

# Backprop

for i in range(backprop_steps) :

	backprop(i % batch_count)

	if (i % 1000 == 0) :
		log_precision(i)

log_precision(i)
plot_graph()

