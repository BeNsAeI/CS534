import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
path="E:\Computer Science\Mchine learning Fall 2018\Implementation assignment 2"
os.chdir(path)
os.getcwd()
np.random.seed(0)
train_file="pa2_train.csv"
valid_file="pa2_valid.csv"
test_file="pa2_test_no_label.csv"


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
    if x >= 0:
        return 1
    else:
        return -1


def err(y_pred, y_act):
    error = 0
    for i in range(len(y_pred)):
        if y_pred[i] * y_act[i] <0:   # if y_pred[i] != y_act[i]:
            error += 1
    return error / len(y_act) * 100


def validate(w, x_valid, y_valid):
    y_pred = np.zeros(x_valid.shape[0])
    for i in range(x_valid.shape[0]):
        y_pred[i] = sign(np.dot(x_valid[i, :], w))
    return err(y_pred, y_valid)


def online_perceptron(x_train, y_train, x_valid, y_valid, iters):
    train_err = np.zeros(iters)
    valid_err = np.zeros(iters)
    w = np.zeros(x_train.shape[1])
    t = 0
    while t < iters:
        y_pred = np.zeros(len(y_train))
        for i in range(x_train.shape[0]):
            # print(np.dot(x_train[i,:],w))
            # print(x_train[i,:].shape)
            # print(w.shape)
            # sys.exit()
            y_pred[i] = sign(np.dot(x_train[i, :], w))
            if y_train[i] * y_pred[i] < 0:
                w = w + y_train[i] * x_train[i, :]
                #t = t + 1
        train_err[t] = err(y_pred, y_train)
        valid_err[t] = validate(w, x_valid, y_valid)
        t = t + 1
    return w, train_err, valid_err

def predict(w,x_test):
    y_pred = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        y_pred[i] = sign(np.dot(x_test[i, :], w))
        if int(y_pred[i])==1:
            y_pred[i]=3
        elif int(y_pred[i])==-1:
            y_pred[i]=5
    return y_pred


w, train_err, valid_err = online_perceptron(x_train, y_train, x_valid, y_valid, 33)
epoch = np.arange(len(train_err)) + 1
plt.figure(0)
plt.plot(epoch, train_err, '-r', label= 'Training error')
plt.plot(epoch, valid_err, '-b', label= 'Validation error')
plt.gca().legend(('Training error','Validation error'))
plt.xlabel('Epochs')
plt.ylabel('Error %')
plt.show()
y_pred=predict(w,x_test)
np.savetxt("oplabel.csv", y_pred, delimiter=",",fmt='%i')
