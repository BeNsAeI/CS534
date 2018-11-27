import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import time
t0=time.time()
path="E:\Computer Science\Mchine learning Fall 2018\Implementation assignment 3"
os.chdir(path)
os.getcwd()
np.random.seed(0)
train_file="pa3_train_reduced.csv"
valid_file="pa3_valid_reduced.csv"
#test_file="pa2_test_no_label.csv"


def gen_train(file):
    train_data=pd.read_csv(file,header=None)
    train_data=np.asarray(train_data)
    m=train_data.shape[0]
    n=m=train_data.shape[1]
#    for i in range(m):
#        if int(train_data[:,0][i])==3:
#            train_data[:, 0][i]=1
#        elif int(train_data[:,0][i])==5:
#            train_data[:, 0][i]=-1
    return train_data
def gen_test(file):
    test_data=pd.read_csv(file,header=None)
    test_data=np.asarray(test_data)
    x_test=np.insert(test_data,0,1,axis=1)
    return x_test

train=gen_train(train_file)
#train=train[:500,:]

validation=gen_train(valid_file)
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for i in range(len(dataset)):
        if dataset[i,index+1] < value:
            left.append(dataset[i,:])
        else:
            right.append(dataset[i,:])
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[0] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
def order(x_col,y_train):
    x_col_ordered=sorted(x_col)
    y_train_ordered=[y for _,y in sorted(zip(x_col,y_train))]
    return x_col_ordered,y_train_ordered

def find_turning_points(dataset,col):
    ''' given a feature, f, this function should reorder the training examples based on values of feature f.
    Then it should find the turning points (where the label class changes one-by-one)
    Also the gini index is calculated for each turning point.
    Finally, a tuple of (turning point, gini_indx) is stored in turning_points.
    '''
    x_train=dataset[:,1:]
    y_train=dataset[:,0]
    turning_points=list()
    x_col, ordered_labels = order(x_train[:,col], y_train)
    initial_step=0
    for x,y in zip(x_col,ordered_labels):
        if initial_step==0:
            current_label = y
            current_point = x
            initial_step += 1
            continue
        if y!=current_label:
            current_label = y
            current_point = x

            #turning_points.append((x+current_point)/2.0) # Average of x+ and x- points will be considered as the border.

            turning_points.append((current_point,x_train[:,col].tolist().index(current_point),col))
            #turning_points.append((x_train[:, col].tolist().index(current_point), col))
            #current_label=y
            #current_point=x
    return turning_points
# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[0] for row in dataset))    #   [3.0, 5.0]
    #print(class_values)
    #sys.exit()
    b_index, b_value, b_score, b_groups = 999, 2000, 999, None
    try:
        n=dataset.shape[1]-1
    except AttributeError:
        dataset=np.array(dataset)
        n=dataset.shape[1]-1
    turning_points_all=list()
    for j in range(n):
        turning_points_all.extend(find_turning_points(dataset,j))
    print('length of current dataset: {}'.format(len(dataset)))
    print('length of current candidate turning points: {}'.format(len(turning_points_all)))

    for case in turning_points_all:
        # note that index in find_turning_points is the counter for an element in a column
        # wheras index in this function is the counter for an element in a row.
        # The equivalency for index in this function is col in the the find_turning_points function.
        value=case[0]
        index=case[1] # counter for columns. not used here
        # the test_split function checks for each row and its value in the col column. if higher -> left, if not, right.
        col=case[2]
        groups = test_split(col, value, dataset)
        gini = gini_index(groups, class_values)
        if len(groups[0])==0 or len(groups[1])==0:
            continue
        if gini < b_score:
            b_index, b_value, b_score, b_groups = col, value, gini, groups
    print("The index for this round: {}".format(b_index))
    if len(turning_points_all)==0:
        b_groups=([dataset[0].tolist()],[])
        b_index=0
        #print(b_groups)
        #sys.exit()
    #print(type(b_groups))
    #print(type(b_groups[0]))
    #print(len(b_groups))
    #sys.exit()
    return {'index' :b_index, 'value' :b_value, 'groups' :b_groups}

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    print("Depth: {}".format(depth))
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def to_terminal(group):
    outcomes = [row[0] for row in group]
    return max(set(outcomes), key=outcomes.count)

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
def err(y_pred, y_act):
    error = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_act[i]:   # if y_pred[i] != y_act[i]:
            error += 1
    return error / len(y_act) * 100


tree = build_tree(train, 3, 2)
#print(tree)
def predict_dataset(dataset,stump={'index': 3, 'right': 1, 'value': -27.3539, 'left': -1}):
    predicted=list()
    x_validation = dataset[:, 1:]
    y_validation = dataset[:, 0]
    for row in x_validation:
        predicted.append(predict(stump, row))
    return predicted

def predict_dataset2(dataset,stump):
    predicted=list()
    x_validation = dataset[:, 1:]
    y_validation = dataset[:, 0]
    for row in x_validation:
        predicted.append(predict(stump, row))
    return predicted
#y_validation_predicted=predict_dataset(validation,stump={'index': 3, 'right': 1, 'value': -27.3539, 'left': -1})
#y_train_predicted=predict_dataset(train,stump={'index': 3, 'right': 1, 'value': -27.3539, 'left': -1})
y_validation_predicted2=predict_dataset2(validation,tree)

y_train_predicted2=predict_dataset2(train,tree)
#print(y_train_predicted2)
y_validation=validation[:,0]
y_train=train[:,0]
#print(y_train)
#print("using stump \n")
#print(err(y_validation_predicted, y_validation))
#print(err(y_train_predicted, y_train))
print("Using tree \n")

print(err(y_validation_predicted2, y_validation))
print(err(y_train_predicted2, y_train))
print("Running time: {}".format(time.time()-t0))
'''
tree=tree_structure(train, 20, 1)
stump = {'index': 3, 'right': 1, 'value': -27.3539, 'left': 0}
tree=tree_structure(train,5,1)
print(tree)
predicted=predict(validation,stump)
y_valid=validation[:,0]
'''
