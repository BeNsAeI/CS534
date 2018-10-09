# Import neccasary libraries
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os

class Helper:
	def __init__(self,file1,file2,file3):
		self.training_set = pd.read_csv(file1)
		self.training_set=self.training_set.drop('dummy',1)
		self.training_set=self.training_set.drop('id',1)

		self.test_set=pd.read_csv(file2)
		self.test_set=self.test_set.drop('dummy',1)
		self.test_set=self.test_set.drop('id',1)

		self.dev_set=pd.read_csv(file3)
		self.dev_set=self.dev_set.drop('dummy',1)
		self.dev_set=self.dev_set.drop('id',1)

	def gen_training_data(self,file,normalization):
		day=list()
		month=list()
		year=list()
		for item in self.training_set['date']:
			day.append(item.split('/')[1])
			month.append(item.split('/')[0])
			year.append(item.split('/')[2])
		day=np.asarray(day).astype(int)
		month=np.asarray(month).astype(int)
		year=np.asarray(year).astype(int)
		self.training_set.insert(loc=0, column='year', value=year)
		self.training_set.insert(loc=0, column='month', value=month)
		self.training_set.insert(loc=0, column='day', value=day)
		self.training_set=self.training_set.drop('date',1)
		self.training_set.head()
		x_train = self.training_set.drop('price', 1)
		y_train = self.training_set['price']
		#Normlization
		if normalization==True:
			x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
		else:
			action='Do nothing'
		# add the bias column
		ones = np.ones(x_train.shape[0])
		x_train.insert(loc=0, column='Intercept', value=ones)
		return x_train,y_train

	def gen_dev_data(self,file,normalization):
		day=list()
		month=list()
		year=list()
		for item in self.dev_set['date']:
			day.append(item.split('/')[1])
			month.append(item.split('/')[0])
			year.append(item.split('/')[2])
		day=np.asarray(day).astype(int)
		month=np.asarray(month).astype(int)
		year=np.asarray(year).astype(int)
		self.dev_set.insert(loc=0, column='year', value=year)
		self.dev_set.insert(loc=0, column='month', value=month)
		self.dev_set.insert(loc=0, column='day', value=day)
		self.dev_set=self.dev_set.drop('date',1)
		self.dev_set.head()
		x_train = self.dev_set.drop('price', 1)
		y_train = self.dev_set['price']
		#Normlization
		if normalization==True:
			x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
		else:
			action='Do nothing'
		# add the bias column
		ones = np.ones(x_train.shape[0])
		x_train.insert(loc=0, column='Intercept', value=ones)
		return x_train,y_train

	def gen_test_data(self,file,normalization):
		day=list()
		month=list()
		year=list()
		for item in self.test_set['date']:
			day.append(item.split('/')[1])
			month.append(item.split('/')[0])
			year.append(item.split('/')[2])
		day=np.asarray(day).astype(int)
		month=np.asarray(month).astype(int)
		year=np.asarray(year).astype(int)
		self.test_set.insert(loc=0, column='year', value=year)
		self.test_set.insert(loc=0, column='month', value=month)
		self.test_set.insert(loc=0, column='day', value=day)
		self.test_set=self.test_set.drop('date',1)
		self.test_set.head()
		x_train = self.test_set
		#Normlization
		if normalization==True:
			x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
		else:
			action='Do nothing'
		# add the bias column
		ones = np.ones(x_train.shape[0])
		x_train.insert(loc=0, column='Intercept', value=ones)
		return x_train

	def solve_lr(self,x_train,y_train,alpha,n_epoch):
		# Option 1 --> set w as random values between 0 and 1
		w=np.random.rand(x_train.shape[1])
		# Option 2 --> set w as zero
		#w=np.zeros(x_train.shape[1])
		w=np.matrix(w).T
		X=np.matrix(x_train)
		y=np.matrix(y_train)
		grad_norm=10000
		counters = list()
		sse_s = list()
		counter=0
		while grad_norm>0.01:
			e=X*w-y.T
			grad=X.T*e/X.shape[0]
			# case 1 separate w0 and other w terms
			#w[0] = w[0] - alpha * e[0]
			#w[1:] = w[1:] - alpha * grad[1:]
			# case to follow the same implementation for all w terms
			w = w - alpha * grad
			e=X*w-y.T
			sse=np.dot(e.T,e)[0,0]/X.shape[0]
			grad_norm=np.square(grad.T*grad)[0,0]
			#print(0.5*sse)
			counter+=1
			counters.append(counter)
			sse_s.append(0.5*sse)
			if counter >= n_epoch:
				print('maximum iteration limit reached!')
				break
		return w,counters,sse_s
	def solve_lrn(self,x_train,y_train,alpha,landa,n_epoch):
		# Option 1 --> set w as random values between 0 and 1
		w=np.random.rand(x_train.shape[1])
		# Option 2 --> set w as zero
		#w=np.zeros(x_train.shape[1])

		w=np.matrix(w).T
		X=np.matrix(x_train)
		y=np.matrix(y_train)
		grad_norm=1000
		counters = list()
		sse_s = list()
		counter=0
		while grad_norm>0.01:
			e=X*w-y.T
			grad=X.T*e/X.shape[0]
			# case 1 separate w0 and other w terms
			w[0] = w[0] - alpha * grad[0]
			w[1:] = w[1:] - alpha * grad[1:]+landa/X.shape[0]*w[1:]
			e=X*w-y.T
			sse=np.dot(e.T,e)[0,0]/X.shape[0]
			grad_norm=np.square(grad.T*grad)[0,0]
			#print(0.5*sse)
			counter+=1
			counters.append(counter)
			sse_s.append(0.5*sse)
			if counter  >= n_epoch:
				print('maximum iteration limit reached!')
				break
		return w,counters,sse_s
	def test(self,w,x_train,y_train):
		X=np.matrix(x_train)
		y=np.matrix(y_train)
		e=X*w-y.T
		sse=np.dot(e.T,e)[0,0]/X.shape[0]
		return sse
	def predict(self,w,x_test):
		X=np.matrix(x_test)
		y=X*w
		return y
