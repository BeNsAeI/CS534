#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
from time import sleep

plot_mode = False
n_epoch_scale = 1
if(len(sys.argv) > 1):
	if sys.argv[1][0] == 'p':
		print("executing: "+sys.argv[1]+" -> "+str(len(sys.argv[1]))+" Plot mode enabled")
		plot_mode = True
	if sys.argv[1][0] == 'h':
		print("executing: h -> "+str(len(sys.argv[1]))+" High performance mode enabled")
		n_epoch_scale = 10
	sleep(5)
if(len(sys.argv) > 2):
	if sys.argv[1][0] == 'p' or sys.argv[2][0] == 'p':
		print("executing: "+sys.argv[1]+" -> "+str(len(sys.argv[1]))+" Plot mode enabled")
		plot_mode = True
	if sys.argv[1][0] == 'h' or sys.argv[2][0] == 'h':
		print("executing: h -> "+str(len(sys.argv[1]))+" High performance mode enabled")
		n_epoch_scale = 10
# Import neccasary libraries
import numpy as np
import pandas as pd
import os
if plot_mode:
	import matplotlib.pyplot as plt

#get_ipython().magic(u'matplotlib inline')
np.random.seed(0)

def gen_data(file,normalization):
	train_data=pd.read_csv(file)
	train_data=train_data.drop('dummy',1)
	train_data=train_data.drop('id',1)
	day=list()
	month=list()
	year=list()
	for item in train_data['date']:
		day.append(item.split('/')[1])
		month.append(item.split('/')[0])
		year.append(item.split('/')[2])
	day=np.asarray(day).astype(int)
	month=np.asarray(month).astype(int)
	year=np.asarray(year).astype(int)
	train_data.insert(loc=0, column='year', value=year)
	train_data.insert(loc=0, column='month', value=month)
	train_data.insert(loc=0, column='day', value=day)
	train_data=train_data.drop('date',1)
	train_data.head()
	x_train = train_data.drop('price', 1)
	y_train = train_data['price']
	#Normlization
	if normalization==True:
		x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
	else:
		action='Do nothing'
	# add the bias column
	ones = np.ones(x_train.shape[0])
	x_train.insert(loc=0, column='Intercept', value=ones)
	return x_train,y_train

def gen_test_data(file,normalization):
	train_data=pd.read_csv(file)
	train_data=train_data.drop('dummy',1)
	train_data=train_data.drop('id',1)
	day=list()
	month=list()
	year=list()
	for item in train_data['date']:
		day.append(item.split('/')[1])
		month.append(item.split('/')[0])
		year.append(item.split('/')[2])
	day=np.asarray(day).astype(int)
	month=np.asarray(month).astype(int)
	year=np.asarray(year).astype(int)
	train_data.insert(loc=0, column='year', value=year)
	train_data.insert(loc=0, column='month', value=month)
	train_data.insert(loc=0, column='day', value=day)
	train_data=train_data.drop('date',1)
	train_data.head()
	x_train = train_data
	#Normlization
	if normalization==True:
		x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
	else:
		action='Do nothing'
	# add the bias column
	ones = np.ones(x_train.shape[0])
	x_train.insert(loc=0, column='Intercept', value=ones)
	return x_train

def solve_lr(x_train,y_train,alpha,n_epoch):
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
	while grad_norm>0.1:
		e=X*w-y.T
		grad=np.multiply(e,X,dtype=np.float64) #float128
		grad=np.sum(grad,axis=0)/X.shape[0]
		# case 1 separate w0 and other w terms
		#w[0] = w[0] - alpha * e[0]
		#w[1:] = w[1:] - alpha * grad[1:]
		# case to follow the same implementation for all w terms
		w = w - alpha * grad
		sse=np.dot(e.T,e)[0,0]/X.shape[0]
		grad_norm=np.square(grad.T*grad,dtype=np.float64)[0,0] #float128
		#print(0.5*sse)
		counter+=1
		counters.append(counter)
		sse_s.append(0.5*sse)
		if counter >= (n_epoch*n_epoch_scale):
			print('maximum iteration limit reached!')
			break
	return w,counters,sse_s
def solve_lrn(x_train,y_train,alpha,landa,n_epoch):
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
		#print (e)
		#print (X)
		grad=np.multiply(e,X,dtype=np.float64) #float128
		grad=np.sum(grad,axis=0)/X.shape[0]
		w = np.squeeze(np.asarray(w))
		grad = np.squeeze(np.asarray(grad, dtype=np.float64))
		#print(grad)
		# case 1 separate w0 and other w terms
		#print('shape of w is {}'.format(w.shape))
		#print('w[0]: {}'.format(w[0]))
		#print('w[1:]: {}'.format(w[1:]))
		#print('grad[0]: {}'.format(grad[0]))
		#print('grad[1:: {}'.format(grad[1:]))
		#print('operation w[0] - alpha * grad[0,0] is \n{}'.format(w[0] - alpha * grad[0]))
		#print('w[0,1:] = w[0,1:] - alpha * grad[0,1:]+landa/X.shape[0]*w[0,1:] is \n {}'.format(w[1:] - alpha * grad[1:]+landa/X.shape[0]*w[1:]))
		w[0] = w[0] - alpha * grad[0]
		w[1:] = w[1:] - alpha * grad[1:]-landa/X.shape[0]*w[1:]
		w=np.asmatrix(w).T
		grad=np.asmatrix(grad)
		e=X*w-y.T
		# print(sum([x**2 for x in e])) cross-check for sse calculation
		sse=np.dot(e.T,e)[0,0]/X.shape[0]
		#print(np.dot(e.T,e))
		#print (sse)
		#sleep(5)
		#print(sum([x**2 for x in grad.T])) #cross-check for grad_norm calculation
		grad_norm=np.sum(np.multiply(grad,grad,dtype=np.float64), dtype=np.float64) #float128
		grad_norm=np.sqrt(grad_norm)
		#print(grad_norm)
		#print(0.5*sse)
		counter+=1
		counters.append(counter)
		sse_s.append(0.5*sse)
		if counter >= (n_epoch*n_epoch_scale):
			print('maximum iteration limit reached!')
			break
	return w,counters,sse_s

def test(w,x_train,y_train):
	X=np.matrix(x_train)
	y=np.matrix(y_train)
	e=X*w-y.T
	sse=np.dot(e.T,e)[0,0]/X.shape[0]
	return sse
def predict(w,x_test):
	X=np.matrix(x_test)
	y=X*w
	return y


# In[3]:


# ========== Part 0.(a) ================
train_data=pd.read_csv("resources/PA1_train.csv")
train_data=train_data.drop('dummy',1)
train_data=train_data.drop('id',1)


# In[4]:


# ========== Part 0.(b) ================
day=list()
month=list()
year=list()
for item in train_data['date']:
	day.append(item.split('/')[1])
	month.append(item.split('/')[0])
	year.append(item.split('/')[2])
day=np.asarray(day).astype(int)
month=np.asarray(month).astype(int)
year=np.asarray(year).astype(int)
train_data.insert(loc=0, column='year', value=year)
train_data.insert(loc=0, column='month', value=month)
train_data.insert(loc=0, column='day', value=day)
train_data=train_data.drop('date',1)
train_data.head()


# In[5]:


# ========== Part 0.(c) ================
print("Category proportions for categorical columns \n")
categs=['waterfront','view','condition','grade']
for item in categs:
	print(train_data.groupby(item).agg({'price':'count'})/train_data.shape[0]*100)
print("\nStandard deviation for numerical columns \n")
print(train_data.std().drop(categs,0))
print("\nMean for numerical columns \n")
print(train_data.mean().drop(categs,0))
range_col=train_data.max()-train_data.min()
print("\nRange for numerical columns \n")
print(range_col.astype(float).drop(categs,0))


# In[6]:


# ========== Part 0.(d) ================
if plot_mode:
	plt.figure(0)
	plt.plot(train_data['sqft_living15'],train_data['price'],'ro')
	plt.xlabel('square footage')
	plt.ylabel('price')
	plt.figure(1)
	plt.plot(train_data['bedrooms'],train_data['price'],'ro')
	plt.xlabel('bedrooms')
	plt.ylabel('price')


# In[7]:


# ========== Part 0.(e) ================
x_train=train_data.drop('price',1)
y_train=train_data['price']
x_train=(x_train - x_train.mean()) / (x_train.max() - x_train.min())
ones=np.ones(x_train.shape[0])
x_train.insert(loc=0, column='Intercept', value=ones)
x_train.head()
w=np.random.rand(x_train.shape[1])


# In[9]:


print('Part 1 -------------------')
# ============= Part 1.a =======================s
data=gen_data("resources/PA1_dev.csv",normalization=True)
x_cross=data[0]
y_cross=data[1]
#alphas=[3,2,1.99,1.5,1.2,1.1,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
alphas=[1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]

if plot_mode:
	plt.figure(3)

dev_sse=list()
for a in alphas:
	results = solve_lr(x_train, y_train, alpha=a,n_epoch=1000)
	if plot_mode:
		plt.xlabel('Iterations')
		plt.ylabel('SSE')
		plt.plot(results[1],results[2])
if plot_mode:
	plt.legend(['alpha= {}'.format(x) for x in alphas], loc='upper right')
	plt.show()


# In[10]:


# ============= Part 1.b =======================s
training_sse=list()
dev_sse=list()
for a in alphas:
	results = solve_lr(x_train, y_train, alpha=a,n_epoch=1000)
	w=results[0]
	sse=test(w,x_cross,y_cross)
	dev_sse.append(sse)
	training_sse.append(results[2][-1])
print('training sse for all the alpha values are:\n {}:\n'.format(training_sse))
print('dev sse for all the alpha values are:\n {}:\n'.format(dev_sse))
if plot_mode:
	plt.figure(5)
	plt.xlabel('alpha')
	plt.ylabel('SSE')
	plt.plot(alphas,dev_sse)
	plt.plot(alphas,training_sse)
	plt.legend(['Validation SSE','Training SSE'], loc='upper right')
	plt.show()


# In[11]:


# ============= Part 1.c =======================s
features=list(train_data._info_axis[:-1])
results = solve_lr(x_train, y_train, alpha=0.001,n_epoch=1000)	
w=results[0]
features.insert(0,'Intercept')
for i in range(len(w)):
	print('weight of {} is {}'.format(features[i],w[0,i]))



# In[12]:


# ============= Part 2.a =======================s
from math import log

landas=[1e-5,1e-3,1e-2,1e-1,1,10] # IF you add 100 to the batch SSE will explode. Use it in report
if plot_mode:
	plt.figure(5)
for l in landas:
	results = solve_lrn(x_train, y_train, alpha=0.001,landa=l,n_epoch=1000)
	if plot_mode:
		plt.xlabel('Iterations')
		plt.ylabel('SSE')
		plt.plot(results[1],results[2])
if plot_mode:
	plt.legend(['Lambda= {}'.format(x) for x in landas], loc='upper right')
	plt.show()


training_sse=list()
dev_sse=list()
for l in landas:
	results = solve_lrn(x_train, y_train, alpha=0.1,landa=l,n_epoch=100)
	w=results[0]
	sse=test(w,x_cross,y_cross)
	dev_sse.append(sse)
	training_sse.append(results[2][-1])
print('training sse for all the lambda values are:\n {}:\n'.format(training_sse))
print('dev sse for all the lambda values are:\n {}:\n'.format(dev_sse))
if plot_mode:
	plt.figure(6)
	plt.xlabel('lambda')
	plt.ylabel('SSE')
	plt.plot([log(landa,10) for landa in landas],dev_sse)
	plt.plot([log(landa,10) for landa in landas],training_sse)
	plt.legend(['Validation SSE','Training SSE'], loc='upper right')
	plt.show()


# In[128]:


# JUST CHECKING STUFF, NOT FOR THE HOMEWORK
# a=np.arange(5)
'''
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
'''


# In[14]:


# ============= Part 3.a =======================s
data=gen_data("resources/PA1_train.csv",normalization=False)
x_train=data[0]
y_train=data[1]
data=gen_data("resources/PA1_train.csv",normalization=False)
x_cross=data[0]
y_cross=data[1]
#alphas=[1,1e-3,1e-6,1e-9,1e-15,1e-30,1e-100,0]
alphas=[1,1e-3,1e-6,1e-9,1e-15,0]

training_sse=list()
dev_sse=list()
for a in alphas:
	results = solve_lr(x_train, y_train, alpha=a,n_epoch=100)
	w=results[0]
	sse=test(w,x_cross,y_cross)
	dev_sse.append(sse)
	training_sse.append(results[2][-1])
print('training sse for all the alpha values are:\n {}:\n'.format(training_sse))
print('dev sse for all the alpha values are:\n {}:\n'.format(dev_sse))
if plot_mode:
	plt.figure(5)
	plt.xlabel('alpha')
	plt.ylabel('SSE')
	plt.plot(alphas,dev_sse)
	plt.plot(alphas,training_sse)
	plt.legend(['Validation SSE','Training SSE'], loc='upper right')
	plt.show()


# In[13]:


# Predictions:
data=gen_data("PA1_train.csv",normalization=True)
x_train=data[0]
y_train=data[1]

x_test=gen_test_data("resources/PA1_test.csv",normalization=True)
print("x_train is:")
print(x_train)
print("y_train is:")
print(y_train)
results = solve_lrn(x_train, y_train, alpha=0.001,landa=0.001,n_epoch=10000)
print("results returned:")
print (results)
w=results[0]
y_test=predict(w,x_test)
y_test=np.squeeze(np.asarray(y_test))
print("Predicted y is:")
print(y_test)
np.savetxt("resources/Predicted_y.csv", y_test, delimiter=",")
print(w)


# In[227]:


x1 = np.arange(12).reshape((6, 2))
x1 = np.asmatrix(x1)
print('shape of x1: {}'.format(x1.shape))
x2 = np.arange(6)
x2 = np.asmatrix(x2).T
print('shape of x2: {}'.format(x2.shape))
print('x1: \n')
print(x1)
print('x2: \n')
print(x2)
mult=np.multiply(x1, x2)
print('mult is: {}'.format(mult))
mult_sum=np.sum(mult,axis=0)
print('\nsum_mult is  {}'.format(mult_sum))
mult_sum[0]


# In[ ]:




