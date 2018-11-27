import os
import pickle
import argparse
import time
import numpy as np
import multiprocessing as mp
from subprocess import call
from collections import namedtuple


path = "Sources/"
training = "pa3_train_reduced"
validation = "pa3_valid_reduced"
format = ".csv"
DEBUG = False
VERBOSE = False
PLOT = False
MULTIPROC = False
threshol_grain = 1
maximum_depth = 20
node_count = 0
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
	# Format only use the column indicated by condition.Feature
	sum = np.sign(Y).sum()
	pos = (Y.shape[0] - sum)/2
	neg = pos
	if sum > 0:
		pos += abs(sum)
	else:
		neg += abs(sum)
	return Data(Positive=pos, Negative=neg)

def get_Y_state(state, condition):
	left_Y = state.Y
	right_Y = state.Y
	delete_Y_left = []
	delete_Y_right = []

#	start = time.time()
	feature = state.X[:, condition.Feature]
	feature = np.vstack((np.arange(feature.shape[0]), feature)).T
	if condition.Type == '<':
		delete_Y_right = (feature[feature[:,1] > condition.Threshold])[:, 0]
		delete_Y_left =  (feature[feature[:,1] <= condition.Threshold])[:, 0]
	else:
		delete_Y_right = (feature[feature[:,1] <= condition.Threshold])[:, 0]
		delete_Y_left =  (feature[feature[:,1] > condition.Threshold])[:, 0]
#	end = time.time()
#	d_print(end - start)
	left_Y = np.delete(left_Y, delete_Y_left , axis=0)
	right_Y=np.delete(right_Y, delete_Y_right, axis=0)
	return left_Y, right_Y

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
	return (small_pos_c + small_neg_c)/(big_pos_c + big_neg_c)

# Gini-index
def U(small_pos_c, small_neg_c):
	p_pos = small_pos_c / small_pos_c + small_neg_c
	p_neg = small_neg_c / small_pos_c + small_neg_c
	return 1 - (p_pos * p_pos) - (p_neg * p_neg)

# Benefit:
def B(big_data, small_left_data, small_right_data):
	B = None

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

def find_threshold(state, j_range_low, j_range_upper, feature_list, big_data):
	best_benefit = -10.0
	best_condition = None
	for i in range(0,state.X.shape[1]):
		for j in range(j_range_low, j_range_upper):
			if not ([i,j] in feature_list):
				condition = Condition(Feature=i, Type='<', Threshold=j*threshol_grain)
				left_state_Y, right_state_Y = get_Y_state(state, condition)
				if not (left_state_Y.shape[0] == 0 or right_state_Y.shape[0] == 0):
					left_data = C(left_state_Y)
					right_data = C(right_state_Y)
					index_benefit = B(big_data, left_data, right_data)
					if (best_benefit <= index_benefit):
						best_benefit = index_benefit
						best_condition = condition
	return best_condition

def train(root, type, feature_list, depth_cap=maximum_depth, par_id=None, marker=None, plot=False, proc=False):
	if depth_cap == maximum_depth:
		d_print("\n___\nTraining "+ type+". Max depth: "+ str(depth_cap))
	global node_count
	if type == "Decision Tree" and depth_cap > 0:
		big_data = C(root.State.Y)
		start = time.time()
		best_condition = find_threshold(root.State,
			-1400/threshol_grain,
			1600/threshol_grain,
			feature_list,
			big_data)
		end = time.time()
		d_print(end - start)
		if best_condition != None:
			feature_list.append([best_condition.Feature,best_condition.Threshold])
			root = split(root, best_condition)
			node_count += 1
			d_print("Node "+ str(maximum_depth - depth_cap) + ": " + str(node_count) + " has " + str(best_condition))

			left_node = train(root.Left, "Decision Tree", feature_list, depth_cap=depth_cap - 1, plot=plot)
			right_node= train(root.Right,"Decision Tree", feature_list, depth_cap=depth_cap - 1, plot=plot)
			root = Node(Data=root.Data,
				State=root.State,
				Condition=root.Condition,
				Parent=root.Parent,
				Left=left_node,
				Right=right_node)
		return root

def walk(node, row, true_label):
	# grab the feature
	# check the condition
	# select branch
	# recurse
	if node.Left == None and node.Right == None:
		if node.Data.Positive > node.Data.Negative:
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
				return walk(node.Right, row, true_label)
			else:
				return walk(node.Left, row, true_label)
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

def delete_tree(root):
	if root.Left != None:
		delete_tree(root.Left)
	if root.Right != None:
		delete_tree(root.Right)
	del root

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
	d_print("Producing root data")
	root_data = C(y_train)
	d_print("Generating State")
	root_state = State(X=x_train, Y=y_train)
	d_print("Initiating the Node")
	root = Node(Data=root_data, State=root_state, Condition=None, Parent=None, Left=None, Right=None)
	d_print("Root: "+str(root))
	feature_list = []
	# Part 1: Train a Tree
	root = train(root, "Decision Tree", feature_list, depth_cap=maximum_depth, plot=PLOT, proc=MULTIPROC)
	# Print the tree:
	print_tree(root)
	d_print("Reading in validation Data...")
	# plot the tree:
	x_validate, y_validate = get_data(path+validation+format, test=False)
	accuracy = validate(x_train, y_train, root, plot=PLOT, proc=MULTIPROC)
	d_print("Training accuracy is: " + str(accuracy) + ".")
	accuracy = validate(x_validate, y_validate, root, plot=PLOT, proc=MULTIPROC)
	d_print("Validation accuracy is: " + str(accuracy) + ".")
	# Part 2: Random Forest
	#train(root, "Random Forest", feature_list, plot=PLOT, proc=MULTIPROC)
	# Part 3: AdaBoost
	#train(root, "Adaboost", feature_list, plot=PLOT, proc=MULTIPROC)
	#cleaning up
	d_print("\n___\nCleaning up ...")
	del root_data
	del root_state
	delete_tree(root)
	d_print("Done.")

if __name__ == "__main__":
	main()
