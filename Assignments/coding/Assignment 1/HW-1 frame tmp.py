import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
from Helper import Helper
	
def part_2_a():
	my_helper = Helper("resources/PA1_train.csv","resources/PA1_test.csv","resources/PA1_dev.csv")
	landas=[0,1e-3,1e-2,1e-1,1,10,100]
	plt.figure(5)
	for l in landas:
		results = my_helper.solve_lrn(x_train, y_train, alpha=1,landa=l,n_epoch=10000)
		plt.xlabel('Iterations')
		plt.ylabel('SSE')
		plt.plot(results[1],results[2])
	plt.legend(['Landa= {}'.format(x) for x in landas], loc='upper right')
	plt.show()


	training_sse=list()
	dev_sse=list()
	for l in landas:
		results = my_helper.solve_lrn(x_train, y_train, alpha=1,landa=l,n_epoch=10000)
		w=results[0]
		sse=test(w,x_cross,y_cross)
		dev_sse.append(sse)
		training_sse.append(results[2][-1])
	print('training sse for all the landa values are:\n {}:\n'.format(training_sse))
	print('dev sse for all the landa values are:\n {}:\n'.format(dev_sse))
	plt.figure(6)
	plt.xlabel('landa')
	plt.ylabel('SSE')
	plt.plot(landas,dev_sse)
	plt.plot(landas,training_sse)
	plt.legend(['Validation SSE','Training SSE'], loc='upper right')
	plt.show()
	a=np.arange(5)
	print('if a is an ndarray')
	print('shape of a is {}'.format(a.shape))
	print('a is {}'.format(a))
	print('np.dot(a.T,a) is {}'.format(np.dot(a,a.T)))
	print('a.T*a is {}'.format(a*a.T))
	a=np.asmatrix(a)
	print('\nif a is matrix')
	print('shape of a is {}'.format(a.shape))
	print('a is {}'.format(a))
	print('np.dot(a.T,a) is {}'.format(np.dot(a,a.T)))
	print('a.T*a is {}'.format(a*a.T))
def main():
	part_2_a()
main()
