import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os,time
from tempfile import TemporaryFile
outfile = TemporaryFile()
#path="E:\Computer Science\Mchine learning Fall 2018\Implementation assignment 2"
#os.chdir(path)
os.getcwd()
np.random.seed(0)
train_file="pa2_train.csv"
valid_file="pa2_valid.csv"
test_file="pa2_test_no_label.csv"

#outfile.seek(0) # Only needed here to simulate closing & reopening file
#np.load(outfile)
def gen_train(file):
    train_data=pd.read_csv(file)
    y_train=train_data.ix[:,0]
    y_train=np.asarray(y_train)
    for i in range(len(y_train)):
        if y_train[i]==3:
            y_train[i]=1
        elif y_train[i]==5:
            y_train[i]=-1

    #print(train_data.shape)
    #train_data.drop(train_data.columns[0], axis=1) # Useless, the DRIO command does not really delete the column it justs hides it from the dataframe.
    #print(train_data.shape)

    b=np.ones(train_data.shape[0])
    train_data.insert(loc=0, column=1, value=b)
    x_train=np.asarray((train_data))
    x_train=x_train[:,1:]
    #print(x_train.shape)
    return x_train,y_train
def gen_test(file):
    test_data=pd.read_csv(file)
    b=np.ones(test_data.shape[0])
    #print(test_data.shape)
    test_data.insert(loc=0, column=1, value=b)
    #print(test_data.shape)
    x_test=np.asarray((test_data))
    return x_test

x_train,y_train=gen_train(train_file)
x_valid,y_valid=gen_train(valid_file)
x_test=gen_test(test_file)



def sign(x):
    if x >= 0.0:
        return 1
    else:
        return -1


def err(y_pred, y_act):
    error = 0
    for i in range(len(y_pred)):
        if y_pred[i] * y_act[i] <0:   # if y_pred[i] != y_act[i]:
            error += 1
    return error / len(y_act) * 100

def index(a):
    p=[]
    for i in range(len(a)):
        if a[i]==True:
            p.append(i)
    p=np.asarray(p)
    return p


def k(x1,x2,p):
    return (1+np.dot(x1,x2))**p

def gram(x_train,p):
    t0 = time.time()
    n=x_train.shape[0]
    k_matrix=np.zeros(n*n).reshape((n,n))
    for i in range(n):
        for j in range(n):
            x1=x_train[i,:]
            x2=x_train[j,:]
            k_matrix[i,j]=k(x1,x2,p)
    print("It took my PC {} seconds to create the K matrix.".format(time.time()-t0))
    np.save(outfile, k_matrix)
    return k_matrix

def predict(x_valid,sv,y_sv,alpha_sv,p):
    y_pred = np.zeros(x_valid.shape[0])
    indx = index(sv)
    #print(indx)
    #sys.exit()
    for i in range(x_valid.shape[0]):
        s=0
        c=0
        for a,y in zip(alpha_sv,y_sv):
            if c in indx:
                s += a * y * k(x_valid[i], x_valid[i], p)   # in general, if you are planning us sv indexing for x_valid, you should say   x_valid[i][sv][0]
            c+=1
        y_pred[i]=np.sign(s)
    return y_pred

def predict2(x_test,x_train,y_train, alpha,p):
    y_pred = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        s=0
        for x,y,a in zip(x_train,y_train, alpha):
            if a<0.0001:
                continue
            s += a * y * k(x_test[i], x, p)   # in general, if you are planning us sv indexing for x_valid, you should say   x_valid[i][sv][0]
        y_pred[i]=np.sign(s)
    return y_pred

def kernel_perceptron(x_train, y_train, x_valid, y_valid, p , iters):
    train_err = np.zeros(iters)
    valid_err = np.zeros(iters)

    n=x_train.shape[0]
    f=x_train.shape[1]
    alpha=np.zeros(n)
    t = 0
    # k_matrix = gram(x_train, p)
    k_matrix = np.load('k_train.npy')
    #print('shape of k matrix loaded: {}'.format(k_matrix.shape))
    while t < iters:
        for i in range(n):
            u = sign(np.sum(k_matrix[:,i] * alpha * y_train))
            if y_train[i]*u <=0:
                alpha[i]+=1
        sv=alpha>0
        alpha_sv=alpha[sv]
        x_sv=x_train[sv]
        y_sv=y_train[sv]
        print('Epoch {}: total sample size is {} with {} support vecotrs'.format(t+1, n,len(alpha_sv)))
        y_pred=predict2(x_train,x_train,y_train, alpha,p)
        train_err[t] = err(y_pred,y_train)
        y_pred = predict2(x_valid,x_train,y_train, alpha,p)
        valid_err[t] = err(y_pred,y_valid)
        t = t + 1

    return train_err, valid_err,alpha

def predict3(w,x_test):
    y_pred = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        y_pred[i] = sign(np.dot(x_test[i, :], w))
        if int(y_pred[i])==1:
            y_pred[i]=3
        elif int(y_pred[i])==-1:
            y_pred[i]=5
    return y_pred

#K_train=np.save('k_train.npy', gram(x_train,2))
#K_test=np.save('k_test.npy', gram(x_test,2))
#sys.exit()


train_err, valid_err , alpha = kernel_perceptron(x_train, y_train, x_valid, y_valid,2, 8)

       # paramters: x_train, y_train, x_valid, y_valid, p , iters
y_pred=predict2(x_test, x_train,y_train, alpha, 2)
# Paramters: x_test,x_train,y_train, alpha,p
for i in range(x_test.shape[0]):
    if int(y_pred[i]) == 1:
        y_pred[i] = 3
    elif int(y_pred[i]) == -1:
        y_pred[i] = 5
np.savetxt("kplabel.csv", y_pred, delimiter=",",fmt='%i')

#sys.exit()

#EVERYTHING IS PREDICTED AS 5 rather than 5 and 3
train_err, valid_err , alpha = kernel_perceptron(x_train, y_train, x_valid, y_valid,2, 15)   # Accordingly, the best epoch is 4
epoch = np.arange(len(train_err)) + 1
plt.figure(0)
plt.plot(epoch, train_err, '-r', label= 'Training error')
plt.plot(epoch, valid_err, '-b', label= 'Validation error')
plt.gca().legend(('Training error','Validation error'))
plt.xlabel('Epochs')
plt.ylabel('Error %')
plt.show()
