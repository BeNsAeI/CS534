#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import neccasary libraries
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
#os.chdir(path)
#os.getcwd()


# In[3]:


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
    while grad_norm > 0.01:
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


# In[4]:


# ========== Part 0.(a) ================
train_data=pd.read_csv("resources/PA1_train.csv")
train_data=train_data.drop('dummy',1)
train_data=train_data.drop('id',1)


# In[5]:


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


# In[6]:


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


# In[7]:


# ========== Part 0.(d) ================
plt.figure(0)
plt.plot(train_data['sqft_living15'],train_data['price'],'ro')
plt.xlabel('square footage')
plt.ylabel('price')
plt.figure(1)
plt.plot(train_data['bedrooms'],train_data['price'],'ro')
plt.xlabel('bedrooms')
plt.ylabel('price')


# In[8]:


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
#alphas=[3,2,1.99,1.5,1.2,1.1,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
alphas=[1.99,1.5,1.2,1.1,1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]


plt.figure(3)
for a in alphas:
    results = solve_lr(x_train, y_train, alpha=a,n_epoch=10000)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.plot(results[1],results[2])
plt.legend(['alpha= {}'.format(x) for x in alphas], loc='upper right')
plt.show()


# In[10]:


# ============= Part 1.b =======================s
data=gen_data("resources/PA1_dev.csv",normalization=True)
x_cross=data[0]
y_cross=data[1]
training_sse=list()
dev_sse=list()
for a in alphas:
    results = solve_lr(x_train, y_train, alpha=a,n_epoch=10000)
    w=results[0]
    sse=test(w,x_cross,y_cross)
    dev_sse.append(sse)
    training_sse.append(results[2][-1])
print('training sse for all the alpha values are:\n {}:\n'.format(training_sse))
print('dev sse for all the alpha values are:\n {}:\n'.format(dev_sse))
plt.figure(5)
plt.xlabel('alpha')
plt.ylabel('SSE')
plt.plot(alphas,dev_sse)
plt.plot(alphas,training_sse)
plt.legend(['Validation SSE','Training SSE'], loc='upper right')
plt.show()


# In[ ]:


# ============= Part 1.c =======================s
results = solve_lr(x_train, y_train, alpha=1.99,n_epoch=10000)    # compare the weights with alpha=1, there is no negative weight when alpha =1 which makes more sense in terms of interpretation.
w=results[0]
weight_df=pd.DataFrame([train_data._info_axis[:-1],w[1:]],index=['Feature','Weight'])   # w[1:] because the first item in w is the bias term ---  train-data._info_axis[:-1] because the last column there is the price.
print(results[2][-1])
print(weight_df)



# In[ ]:


# ============= Part 2.a =======================s

landas=[0,1e-3,1e-2,1e-1,1,10,100]
plt.figure(5)
for l in landas:
    results = solve_lrn(x_train, y_train, alpha=1,landa=l,n_epoch=10000)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.plot(results[1],results[2])
plt.legend(['Landa= {}'.format(x) for x in landas], loc='upper right')
plt.show()


training_sse=list()
dev_sse=list()
for l in landas:
    results = solve_lrn(x_train, y_train, alpha=1,landa=l,n_epoch=10000)
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


# In[ ]:


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



# In[133]:


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
    results = solve_lr(x_train, y_train, alpha=a,n_epoch=10000)
    w=results[0]
    sse=test(w,x_cross,y_cross)
    dev_sse.append(sse)
    training_sse.append(results[2][-1])
print('training sse for all the alpha values are:\n {}:\n'.format(training_sse))
print('dev sse for all the alpha values are:\n {}:\n'.format(dev_sse))
plt.figure(5)
plt.xlabel('alpha')
plt.ylabel('SSE')
plt.plot(alphas,dev_sse)
plt.plot(alphas,training_sse)
plt.legend(['Validation SSE','Training SSE'], loc='upper right')
plt.show()


# In[144]:


# Predictions:
x_test=gen_test_data("resources/PA1_test.csv",normalization=True)
results = solve_lrn(x_train, y_train, alpha=1.9,landa=0,n_epoch=10000)
w=results[0]
y_test=predict(w,x_test)
y_test=np.squeeze(np.asarray(y_test))
y_test
np.savetxt("Predicted_y.csv", y_test, delimiter=",")
print(w)


# In[ ]:




