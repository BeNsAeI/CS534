import os
import pickle
import argparse
import time
import math
import numpy as np
import multiprocessing as mp
from subprocess import call
from collections import namedtuple
from random import randint

path = "Sources/"
training = "pa3_train_reduced"
validation = "pa3_valid_reduced"
format = ".csv"
DEBUG = False
VERBOSE = False
PLOT = False
MULTIPROC = False
multi_output = mp.Queue()
threshol_grain = 1
maximum_depth = 20
maximum_depth_rf = 9
node_count = 0
feature_list = []
RF_tree_count = 3
RF_feature_count = 10

# Result
## myResult = Result(1, 0.5, 0.7)
## Or
## myResult = Result(ID=1, TA=0.5, VA=0.7)
Result = namedtuple("Result", "ID TA VA") 

# State
## myState = State(X, Y)
## Or
## myState = State(X=X, Y=Y)
State = namedtuple("State", "X Y")

# Data structure
## myData = Data(55, 20)
## Or
## myData = Data(Positive=55, Negative=20)
Data = namedtuple("Data", "Positive Negative")

# i Branch condition is x_6 < 3:
## myCondition = Condition(6, '<', 3)
##Or
## myCondition = Condition(Feature=6, Type='<', Threshold=3)
Condition = namedtuple("Condition", "Feature Type Threshold")

# A convinient Node structure
# Example Declearation:
## myNode = Node(myData, myState, myCondition, 0x---, 0x---, 0x---)
## Or
## myNode = Node(Data=myData, State=myState, Condition=myCondition, Parent=0x---, Left=0x---, Right=0x---)
Node = namedtuple("Node", "Data State Condition Parent Left Right")

def get_data(filename, test=False):
    x = np.genfromtxt(filename, delimiter=',', dtype=float)
    y = None
    if not test:
        y = x[:, 0]
        y[y == 5] = -1
        y[y == 3] = 1
        x = np.delete(x, [0], 1)
    return x, y

def d_print(value):
    if VERBOSE:
        print(value)

# Count
def C(Y):
    pos = Y[Y>0].shape[0]
    neg = Y.shape[0] - pos
    return Data(Positive=pos, Negative=neg)

def get_Y_state(state, condition):
    left_Y = state.Y
    feature = state.X[:, condition.Feature]
    feature = np.vstack((np.arange(feature.shape[0]), feature)).T
    if condition.Type == '<':
        delete_Y_left =  (feature[feature[:,1] > condition.Threshold])[:, 0]
    else:
        delete_Y_left =  (feature[feature[:,1] <= condition.Threshold])[:, 0]
    left_Y = np.delete(left_Y, delete_Y_left , axis=0)
    count_state = C(state.Y)
    count_left = C(left_Y)
    count_right= Data(Positive=count_state.Positive - count_left.Positive,
                      Negative=count_state.Negative - count_left.Negative)
    return count_left, count_right

def get_state(state, condition):
    left_X = state.X
    left_Y = state.Y
    right_X = state.X
    right_Y = state.Y
    delete_rows_left = []
    delete_rows_right = []
    delete_Y_left = []
    delete_Y_right = []

    feature = state.X[:, condition.Feature]
    if condition.Type == '<':
        for i in range(0,feature.shape[0]):
            if feature[i] < condition.Threshold:
                delete_rows_right.append(i)
                delete_Y_right.append(i)
            else:
                delete_rows_left.append(i)
                delete_Y_left.append(i)
    else:
        for i in range(0,feature.shape[0]):
            if feature[i] >= condition.Threshold:
                delete_rows_right.append(i)
                delete_Y_right.append(i)
            else:
                delete_rows_left.append(i)
                delete_Y_left.append(i)
    left_X = np.delete(left_X,  delete_rows_left, axis=0)
    right_X= np.delete(right_X, delete_rows_right, axis=0)
    left_Y = np.delete(left_Y, delete_Y_left , axis=0)
    right_Y=np.delete(right_Y, delete_Y_right, axis=0)
    left_state = State(X=left_X, Y=left_Y)
    right_state = State(X=right_X, Y=right_Y)
    return left_state, right_state

def split(node, condition):
    left_state = None
    right_state = None
    left_state, right_state = get_state(node.State, condition)

    left_data = C(left_state.Y)
    left_condition = condition

    right_data = C(right_state.Y)
    right_condition = None
    if condition.Type == '>':
        right_condition = Condition(Feature=condition.Feature, Type='<', Threshold=condition.Threshold)
    else:
        right_condition = Condition(Feature=condition.Feature, Type='>', Threshold=condition.Threshold)

    left_node = Node(Data=left_data,
        State=left_state,
        Condition=left_condition,
        Parent=node,
        Left=None,
        Right=None)

    right_node = Node(Data=right_data,
        State=right_state,
        Condition=right_condition,
        Parent=node,
        Left=None,
        Right=None)

    new_node=Node(Data=node.Data,
        State=node.State,
        Condition=node.Condition,
        Parent=node.Parent,
        Left=left_node,
        Right=right_node)

    return new_node

# Probability
def P(big_pos_c, big_neg_c, small_pos_c, small_neg_c):
    return (float(small_pos_c + small_neg_c)/float(big_pos_c + big_neg_c))

# Gini-index
def U(small_pos_c, small_neg_c):
    p_pos = float(small_pos_c) / float(small_pos_c + small_neg_c)
    p_neg = float(small_neg_c) / float(small_pos_c + small_neg_c)
    return (1 - (p_pos * p_pos) - (p_neg * p_neg))

# Benefit:
def B(big_data, small_left_data, small_right_data):

    big_pos_c = big_data.Positive
    big_neg_c = big_data.Negative

    small_left_pos_c = small_left_data.Positive
    small_left_neg_c = small_left_data.Negative

    small_right_pos_c = small_right_data.Positive
    small_right_neg_c = small_right_data.Negative

    U_A = U(big_pos_c, big_neg_c)
    U_AL = U(small_left_pos_c, small_left_neg_c)
    U_AR = U(small_right_pos_c, small_right_neg_c)
    p_l = P(big_pos_c, big_neg_c, small_left_pos_c, small_left_neg_c)
    p_r = P(big_pos_c, big_neg_c, small_right_pos_c, small_right_neg_c)
    return U_A - p_l * U_AL - p_r * U_AR

def plot_accuracies(train_accuracies, valid_accuracies,iter):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
    plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies)
    plt.gca().legend(('Training Accuracy','Validation Accuracy'))
    #plt.show()
    plt.savefig('figure-'+ iter + '.png');

def print_tree(root, count=0):
    tab = ""
    for i in range(0,count):
        tab += "."
    if(root.Condition != None):
        print(str(tab)+
            "Node depth: "+
            str(count)+
            ", X:"+
            str(root.State.X.shape)+
            ", Y:"+
            str(root.State.Y.shape)+
            ", ["+
            str(root.Data.Positive)+
            ","+
            str(root.Data.Negative)+
            "]: x_"+
            str(root.Condition.Feature)+
            " "+
            str(root.Condition.Type)+
            " "+
            str(root.Condition.Threshold)+
            ".")
    else:
        print(str(tab)+
            "Node depth: "+
            str(count)+
            ", X:"+
            str(root.State.X.shape)+
            ", Y:"+
            str(root.State.Y.shape)+
            ", ["+
            str(root.Data.Positive)+
            ","+
            str(root.Data.Negative)+
            "].")
    if root.Left != None:
        print_tree(root.Left, count+1)
    if root.Right != None:
        print_tree(root.Right,count+1)

def find_threshold(state, feature_index):
    thresholds = []
    global feature_list
    feature = state.X[:, feature_index]
    feature = np.vstack((feature, state.Y)).T
    feature = feature[feature[:, 0].argsort()]
    for i in range(1,feature.shape[0]):
        if np.sign(feature[i-1,1]) != np.sign(feature[i,1]):
            threshold = (feature[i,0] + feature[i-1,0])/2
            if not ([feature_index,threshold] in feature_list):
                thresholds.append(threshold)
    return thresholds

def find_best_benefit(state, big_data, random=False):
    global feature_list
    best_benefit = -np.inf
    index_benefit = -np.inf
    best_condition = None
    max_index = None
    if random:
        global RF_feature_count
        max_index = np.random.choice(np.arange(state.X.shape[1]), size=RF_feature_count, replace=False)
    else:
        max_index = np.arange(0,state.X.shape[1])

    for i in max_index:
        feature = state.X[:, i]
        feature = np.vstack((feature, state.Y)).T
        feature = feature[feature[:, 0].argsort()]
        thresholds = find_threshold(state, i)
        for j in thresholds:
            condition = Condition(Feature=i, Type='<', Threshold=j)
            left_data, right_data = get_Y_state(state, condition)
            index_benefit = B(big_data, left_data, right_data)
            if (best_benefit < index_benefit):
                best_benefit = index_benefit
                best_condition = condition
    return best_condition

def train(root,depth_cap=maximum_depth, random=False, plot=False, proc=False):
    if depth_cap == 0:
        return root
    if depth_cap == maximum_depth:
        d_print("\n___\nTraining. Max depth: "+ str(depth_cap))
    global node_count
    global feature_list
    big_data = C(root.State.Y)
    best_condition = find_best_benefit(root.State,
        big_data,
        random=random)
    if best_condition != None:
        feature_list.append([best_condition.Feature,best_condition.Threshold])
        root = split(root, best_condition)
        node_count += 1
        d_print("Node "+ str(maximum_depth - depth_cap) + ": " + str(node_count) + " has " + str(best_condition))
        left_node = train(root.Left, depth_cap=depth_cap - 1, random=random, plot=plot)
        right_node= train(root.Right, depth_cap=depth_cap - 1, random=random, plot=plot)
        root = Node(Data=root.Data,
            State=root.State,
            Condition=root.Condition,
            Parent=root.Parent,
            Left=left_node,
            Right=right_node)
    #d_print(feature_list)
    return root

def walk(node, row, true_label):
    if node.Left == None and node.Right == None:
        if node.Data.Positive >= node.Data.Negative:
            return +1
        else:
            return -1
    else:
        if node.Left.Condition.Type == '<':
            if row[node.Left.Condition.Feature] < node.Left.Condition.Threshold:
                return walk(node.Left, row, true_label)
            else:
                return walk(node.Right, row, true_label)
        else:
            if row[node.Left.Condition.Feature] >= node.Left.Condition.Threshold:
                return walk(node.Left, row, true_label)
            else:
                return walk(node.Right, row, true_label)
    d_print("None of the conditions in walk were satisfied.")
    exit(1)

def validate(x_validate, y_validate, root, plot=PLOT, proc=MULTIPROC):
    # Get the tree
    true_label = None
    correct_total = 0
    for true_label, row in zip(y_validate,x_validate):
        prediction = walk(root, row, true_label)
        if np.sign(true_label) == np.sign(prediction):
            correct_total += 1
    return (float(correct_total) / float(y_validate.shape[0]))

def validate_rf(x_validate, y_validate, roots, plot=PLOT, proc=MULTIPROC):
    true_label = None
    correct_total = 0
    for true_label, row in zip(y_validate,x_validate):
        predictions = 0
        for i in roots:
            predictions += walk(i, row, true_label)
        if predictions == 0:
            predictions += 1
        if np.sign(true_label) == np.sign(predictions):
            correct_total += 1
    #    d_print(str(predictions) + ", " + str(true_label) + ", " + str(np.sign(true_label) == np.sign(predictions)))
    return (float(correct_total) / float(y_validate.shape[0]))

def delete_tree(root):
    if root.Left != None:
        delete_tree(root.Left)
    if root.Right != None:
        delete_tree(root.Right)
    del root

def train_DT(root, depth_cap=maximum_depth, plot=False, proc=False):
    root = train(root, depth_cap, random=False, plot=plot, proc=proc)
    # Print the tree:
    print_tree(root)
    d_print("Reading in validation Data...")
    # plot the tree:
    x_validate, y_validate = get_data(path+validation+format, test=False)
    train_accuracy = validate(root.State.X, root.State.Y, root, plot=PLOT, proc=MULTIPROC)
    print("Training accuracy is: " + str(train_accuracy) + ".")
    valid_accuracy = validate(x_validate, y_validate, root, plot=PLOT, proc=MULTIPROC)
    print("Validation accuracy is: " + str(valid_accuracy) + ".")
    return train_accuracy, valid_accuracy

def train_RF(root, depth_cap=maximum_depth, local_tree_count=RF_tree_count, plot=False, proc=False):
    roots = []
    x_train = root.State.X
    y_train = root.State.Y
    global feature_list
    global node_count
    feature_list = []
    for i in range(0,local_tree_count):
        node_count = 0
        if (i+1) % 10 == 1 and (i+1) != 11:
            d_print("Training " + str(i+1) + "st tree")
        elif (i+1) % 10 == 2 and (i+1) != 12:
            d_print("Training " + str(i+1) + "nd tree")
        elif (i+1) % 10 == 3 and (i+1) != 13:
            d_print("Training " + str(i+1) + "rd tree")
        else:
            d_print("Training " + str(i+1) + "th tree")
        X, Y = sample(root.State.X, root.State.Y)
        root_data = C(Y) #root.State.Y)
        root_state = State(X=X, Y=Y) #State(X=root.State.X, Y=root.State.Y)
        tmp_root = Node(Data=root_data, State=root_state, Condition=None, Parent=None, Left=None, Right=None)
        roots.append(tmp_root)
        roots[i] = train(roots[i], depth_cap, random=True, plot=plot, proc=proc)
    for i in roots:
        #print_tree(i)
        accuracy = validate(root.State.X, root.State.Y, i, plot=plot, proc=proc)
        d_print("Indovidual tree accuracy is: "+ str(accuracy) + ".")
    x_validate, y_validate = get_data(path+validation+format, test=False)
    train_accuracy = validate_rf(root.State.X, root.State.Y, roots, plot=plot, proc=proc)
    print("Train accuracy is: "+ str(train_accuracy) + ".")
    valid_accuracy = validate_rf(x_validate, y_validate, roots, plot=plot, proc=proc)
    print("Validation accuracy is: "+ str(valid_accuracy) + ".")
    if proc:
        global multi_output
        data_point = Result(ID=local_tree_count, TA=train_accuracy, VA=valid_accuracy)
        d_print(data_point)
        multi_output.put(data_point)
    return train_accuracy, valid_accuracy

def sample(x, y):
    size = x.shape[0]
    inds = np.arange(size)
    sampled = np.random.choice(inds, size=size, replace=True)
    return x[sampled, :], y[sampled]

def validate_adab(x_validate, y_validate, root, plot=PLOT, proc=MULTIPROC):
    # Get the tree
    bool_prediciton = []
    true_label = None
    correct_total = 0
    for true_label, row in zip(y_validate,x_validate):
        prediction = walk(root, row, true_label)
        if np.sign(true_label) == np.sign(prediction):
            correct_total += 1
            bool_prediciton.append(True)
        else:
            bool_prediciton.append(False)
    return (float(correct_total) / float(y_validate.shape[0])), bool_prediciton

def adaboost_error(x_validate, y_validate, root, weights):
    bool_prediciton = []
    error = 0
    for true_label, row, weight in zip(y_validate, x_validate, weights):
        prediction = walk(root, row, true_label)
        if np.sign(true_label) == np.sign(prediction):
            bool_prediciton.append(True)
        else:
            error += weight
            bool_prediciton.append(False)
    print "errors made: ", bool_prediciton.count(False)
    print "error: ", error
    return error, bool_prediciton

def adaboost_predict(x_validate, y_validate, root, model):
    error = 0
    predicted=list()
    predicted_all=list()
    for true_label, row in zip(y_validate, x_validate):
        prediction = 0
        for tree_alpha in model:
            tree=tree_alpha[0]
            alpha = tree_alpha[1]
            prediction += alpha * walk(root, row, true_label)
        if np.sign(true_label) != np.sign(prediction):
            error += 1
    print "errors made: ", error
    return (error/float(y_validate.shape[0]))

def main():
    #Parsing arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action='store_true', help="if set, script will show plots")
    parser.add_argument("-m", "--multi_process", action='store_true', help="if set, script will run in parallel")
    parser.add_argument("-d", "--debug", action='store_true', help="if set, debug mode is activated")
    parser.add_argument("-v", "--verbose", action='store_true', help="if set, verbose mode is activated")
    args = parser.parse_args()
    global DEBUG
    global VERBOSE
    global PLOT
    global MULTIPROC
    DEBUG = args.debug
    VERBOSE = args.debug or args.verbose
    PLOT = args.plot
    MULTIPROC = args.multi_process
    d_print(args)
    # Getting the data:
    d_print("Reading in Data...")
    x_train, y_train = get_data(path+training+format, test=False)
    global feature_list
    # Part 1: Train Trees with depth varying from 1 to 32 for plotting purposes
    # Part 3: AdaBoost
    L = [5]
    num_rows = x_train.shape[0]
    D = np.ones(num_rows) * 1.0 / num_rows
    train_accuracies = []
    valid_accuracies = []
    out = []
    for n in L:
        for k in range(n):
            weighted_dataset = []
            for i in range(num_rows):
                weighted_dataset.append(x_train[i] * D[i])
            weighted_dataset = np.asarray(weighted_dataset)

            root_data = C(y_train)
            root_state = State(X=weighted_dataset, Y=y_train)
            d_print("Initiating the Node")
            root = Node(Data=root_data, State=root_state, Condition=None, Parent=None, Left=None, Right=None)

            start = time.time()
            root = train(root, 9, random=False, plot=PLOT, proc=MULTIPROC)
            end = time.time()
            error, bool_prediction = adaboost_error(root.State.X, root.State.Y, root, D)

            alpha=0.5 * math.log((1-error)/float(error))

            for i in range(len(D)):
                if bool_prediction[i]==True:
                    D[i] = math.exp(-1 * alpha) * D[i]
                else:
                    D[i] = math.exp(alpha) * D[i]
            sum_D = float(sum(D))
            D=[d / sum_D for d in D]

            out.append((root,alpha))
            d_print(str(k)+": Training took: " + str(end - start) + " seconds")
            d_print("Training accuracy: " + str(1 - error))

        print "Start validation"
        x_validate, y_validate = get_data(path+validation+format, test=False)
        error = adaboost_predict(x_validate, y_validate, root, out)
        d_print("Validation accuracy: " + str(1 - error))
    #cleaning up
    d_print("\n___\nCleaning up ...")
    del root_data
    del root_state
    delete_tree(root)
    d_print("Done.")

if __name__ == "__main__":
    main()
