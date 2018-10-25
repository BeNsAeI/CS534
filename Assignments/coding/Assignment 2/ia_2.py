import os
import argparse
import numpy as np

def main():
	x_train, y_train = get_data("pa2_train.csv")
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--plot", action='store_true', help="if set, script will show plots")
	parser.add_argument("-s", "--save", action='store_true', help="if set, a copy of kernels will be stored")
	args = parser.parse_args()

	x_valid, y_valid = get_data("pa2_valid.csv")
	x_test, _ = get_data("pa2_test_no_label.csv", test=True)

	iters = 15
	first_part(x_train, y_train, x_valid, y_valid, x_test, iters, args.plot)
	second_part(x_train, y_train, x_valid, y_valid, iters, args.plot)
	third_part(x_train, y_train, x_valid, y_valid, x_test, iters, args.plot, args.save)

def get_data(filename, test=False):
	x = np.genfromtxt(filename, delimiter=',', dtype=float)
	y = None

	x = np.insert(x, [1], 1, axis=1)
	if not test:
		y = x[:, 0]
		y[y == 5] = -1
		y[y == 3] = 1

		x = np.delete(x, [0], 1) # remove first column with y values
	return x, y

def first_part(x_train, y_train, x_valid, y_valid, x_test, iters, plot):

	# train
	print "-------------------------------"
	print "Part 1. Training"
	print "-------------------------------"
	all_weights, train_accuracies = online_perceptron(x_train, y_train, iters)
	#np.savetxt('my_out.txt', weights)

	# validate
	print "-------------------------------"
	print "Part 1. Validation"
	print "-------------------------------"
	valid_accuracies = perceptron_validate(all_weights, x_valid, y_valid)

	# predict
	print "-------------------------------"
	print "Part 1. Prediction"
	weights = all_weights[-2] # choose 14th iteration's weights
	filename = 'oplabel.csv'
	predictions = perceptron_predict(x_test, weights)
	np.savetxt(filename, predictions)
	print "Saved to %s" % filename
	print "-------------------------------"

	if plot:
		plot_accuracies(train_accuracies, valid_accuracies,iter='1')

def second_part(x_train, y_train, x_valid, y_valid, iters, plot):

	# train
	print "-------------------------------"
	print "Part 2. Training"
	print "-------------------------------"
	all_weights, train_accuracies = average_perceptron(x_train, y_train, iters)
	#np.savetxt('my_out.txt', weights)

	# validate
	print "-------------------------------"
	print "Part 2. Validation"
	print "-------------------------------"
	valid_accuracies = perceptron_validate(all_weights, x_valid, y_valid)

	if plot:
		plot_accuracies(train_accuracies, valid_accuracies,iter='2')

def third_part(x_train, y_train, x_valid, y_valid, x_test, iters, plot, save):
	all_p = [1, 2, 3, 7, 15]
	all_alphas = {} # alphas for all p values and all iterations

	for p in all_p:
		# compute kernel or upload existing from file, if exists
		print "-------------------------------"
		print "Part 3. Preparing kernel matrix for training with p=%d" % p
		print "-------------------------------"
		filename = 'kernels/kernel_train_p%d.csv' % p
		kernel = get_kernel(filename, x_train, x_train, p,save)

		# train
		print "-------------------------------"
		print "Part 3. Training with p=%d" % p
		print "-------------------------------"
		alphas, train_accuracies = kernel_perceptron(kernel, y_train, iters)
		all_alphas[p] = alphas
		#np.savetxt('my_out.txt', weights)

		print "-------------------------------"
		print "Part 3. Preparing kernel matrix for validation with p=%d" % p
		print "-------------------------------"
		filename = 'kernels/kernel_valid_p%d.csv' % p
		kernel = get_kernel(filename, x_valid, x_train, p, save)

		# validate
		print "-------------------------------"
		print "Part 3. Validation with p=%d" % p
		print "-------------------------------"
		valid_accuracies = kernel_validate(kernel, y_valid, p, alphas, y_train)

		if plot:
			plot_accuracies(train_accuracies, valid_accuracies,iter=('3 - '+ str(p)))

	# predict
	p = 3
	iterations = 6
	print "-------------------------------"
	print "Part 3. Prdiction with p=%d and %d iterations" % (p, iterations)
	print "Part 3. Preparing kernel matrix for predicion with p=%d" % p
	print "-------------------------------"
	filename = 'kernels/kernel_test_p%d.csv' % p
	kernel = get_kernel(filename, x_test, x_train, p, save)

	alpha = all_alphas[p][iterations-1]
	predictions = kernel_predict(kernel, y_train, alpha, p)

	filename = 'kplabel.csv'
	np.savetxt(filename, predictions)
	print "Saved to %s" % filename
	print "-------------------------------"

def get_kernel(filename, x_1, x_2, p, save):
	kernel = None

	if os.access(filename, os.R_OK):
		print "Processing file %s" % filename
		kernel = np.loadtxt(filename)
	else:
		print "File %s not found. Will create new one." % filename
		kernel = compute_kernel(x_1, x_2, p)
		if(save):
			np.savetxt(filename, kernel)

	return kernel

def plot_accuracies(train_accuracies, valid_accuracies,iter):
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
	plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies)
	plt.gca().legend(('Training error','Validation error'))
	#plt.show()
	plt.savefig('figure '+ iter + '.png');

def kernel_function(x, y, p):
	k = (1 + np.dot(x.T, y))**p
	return k

def compute_kernel(x_1, x_2, p):
	m = x_1.shape[0]
	n = x_2.shape[0]
	kernel = np.zeros((m, n))
	for i in range(m):
		for j in range(n):
			kernel[i, j] = kernel_function(x_1[i, :], x_2[j, :], p)
	return kernel

def kernel_perceptron(kernel, y, iters):
	m = kernel.shape[0]

	alphas = []
	accuracies = []

	alpha = np.zeros(m)
	for iter_ in range(iters):
		correct_predictions = 0
		for i in range(m):
			u = np.dot(kernel[:, i], alpha * y)
			if y[i] * u <= 0:
				alpha[i] += 1
			else:
				correct_predictions += 1

		accuracy = correct_predictions / float(m)
		print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
		accuracies.append(accuracy)
		alphas.append(alpha.copy())
	return alphas, accuracies

def kernel_validate(kernel, y, p, alphas, y_train):
	accuracies = []
	for i, alpha in enumerate(alphas):
		predictions = kernel_predict(kernel, y_train, alpha, p)
		u = predictions * y
		correct_predictions = u[u > 0].shape[0]
		accuracy = correct_predictions / float(y.shape[0])
		print "Iteration %s, accuracy %s" % (i+1, accuracy)
		accuracies.append(accuracy)
	return accuracies

def kernel_predict(kernel, y_train, alpha, p):
	m = kernel.shape[0]
	n = kernel.shape[1]
	y_pred = np.zeros(m)
	for i in range(m):
		s = 0
		for j in range(n):
			if alpha[j] > 0:
				s += alpha[j] * y_train[j] * kernel[i, j]

		y_pred[i] = np.sign(s)
	return y_pred

def online_perceptron(x, y, iters):
	all_weights = []
	m = x.shape[0]
	accuracies = []
	weights = np.zeros(x.shape[1])

	for iter_ in range(iters):
		correct_predictions = 0
		for i in range(m):
			u = np.sign(y[i] * np.dot(x[i], weights))
			#loss += max(0, -1*u)
			if u <= 0:
				weights += y[i] * x[i]
			else:
				correct_predictions += 1

		accuracy = correct_predictions / float(m)
		print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
		accuracies.append(accuracy)

		all_weights.append(weights.copy())

	return all_weights, accuracies

def average_perceptron(x, y, iters):
	all_weights = []
	m = x.shape[0]
	accuracies = []
	weights = np.zeros(x.shape[1])
	avg_w = np.zeros(x.shape[1])
	s = 0
	c = 0

	for iter_ in range(iters):
		correct_predictions = 0
		for i in range(m):
			u = np.sign(y[i] * np.dot(x[i], weights))
			#loss += max(0, -1*u)
			if u <= 0:
				if s + c > 0:
					avg_w = ((s * avg_w) + (c * weights)) / float(s + c)
				s = s + c
				weights += y[i] * x[i]
				c = 0
			else:
				correct_predictions += 1
				c += 1

		if c > 0:
			avg_w = ((s * avg_w) + (c * weights)) / (s + c)

		accuracy = correct_predictions / float(m)
		print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
		accuracies.append(accuracy)

		all_weights.append(avg_w.copy())
	return all_weights, accuracies

def perceptron_validate(all_weights, x, y):
	m = x.shape[0]
	accuracies = []
	for iter_, weights in enumerate(all_weights):
		correct_predictions = 0
		for i in range(m):
			u = np.sign(y[i] * np.dot(x[i], weights))
			#loss += max(0, -1*u)
			if u > 0:
				correct_predictions += 1

		accuracy = correct_predictions / float(m)
		print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
		accuracies.append(accuracy)
	return accuracies

def perceptron_predict(x, weights):
	predictions = np.sign(np.dot(x, np.matrix(weights).T))

	return predictions

if __name__ == "__main__":
	main()
