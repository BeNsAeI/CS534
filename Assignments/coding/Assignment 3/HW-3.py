import os
import argparse
import numpy as np
import multiprocessing as mp
from collections import namedtuple

output = mp.Queue()
processes = []

path = "Sources/"
training = "pa3_train_reduced"
validation = "pa3_valid_reduced"
format = ".csv"
DEBUG = False
VERBOSE = False
PLOT = False

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
def C(Y, ret_out=None):
	# Format only use the column indicated by condition.Feature
	out = None
	sum = Y.sum()
	pos = (Y.shape[0] - sum)/2 #0
	neg = (Y.shape[0] - sum)/2 #0
	if sum > 0:
		pos += sum
	else:
		neg += sum
	out = Data(Positive=pos, Negative=neg)
	if ret_out != None:
		ret_out.put(out)
	return out

def get_state(state, condition):
	left_X = np.zeros(100)
	left_Y = np.zeros(100)
	right_X = np.zeros(100)
	right_Y = np.zeros(100)

	feature = state.X[:, condition.Feature]
	if condition.Type == '<':
		for i in range(0,feature.shape[0]):
			if feature[i] < condition.Threshold:
				left_X = np.insert(left_X, 1, state.X[i,:], axis=0)
				left_Y = np.insert(left_Y, 1, state.Y[i])
			else:
				right_X = np.insert(right_X, 1, state.X[i,:], axis=0)
				right_Y = np.insert(right_Y, 1, state.Y[i])
	else:
		for i in range(0,feature.shape[0]):
			if feature[i] > condition.Threshold:
				left_X = np.insert(left_X, 1, state.X[i,:], axis=0)
				left_Y = np.insert(left_Y, 1, state.Y[i])
			else:
				right_X = np.insert(right_X, 1, state.X[i,:], axis=0)
				right_Y = np.insert(right_Y, 1, state.Y[i])
	left_X = np.delete(left_X, [0], 0)
	left_Y = np.delete(left_Y, [0], 0)
	right_X = np.delete(right_X, [0], 0)
	right_Y = np.delete(right_Y, [0], 0)
	left_state = State(X=left_X, Y=left_Y)
	right_state = State(X=right_X, Y=right_Y)
	return left_state, right_state

def split(node, condition):
	left_state = None
	right_state = None
	left_state, right_state = get_state(node.State, condition)

	left_data = C(left_state.Y)
	left_condition = condition

	right_state = None
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

	right_state= Node(Data=right_data,
		State=right_state,
		Condition=right_condition,
		Parent=node,
		Left=None,
		Right=None)

	new_node = Node(Data=node.Data,
		State=node.State,
		Condition=node.Condition,
		Parent=node.Parent,
		Left=left_node,
		Right=right_node)
	return new_node

# Probability
def P(big_pos_c, big_neg_c, small_pos_c, small_neg_c):
	return ((small_pos_c + small_neg_c)/(big_pos_c + big_neg_c))

# Gini-index
def U(small_pos_c, small_neg_c):
	p_pos = small_pos_c / (small_pos_c + small_neg_c)
	p_neg = small_neg_c / (small_pos_c + small_neg_c)
	return 1 - (p_pos * p_pos) - (p_neg * p_neg)

# Benefit:
def B(main_state, left_state, right_state, condition):
	d_print("Computing benefit: " + str(condition))
	B = None

	global processes
	global output
	processes.append(mp.Process(target=C, args=(main_state.Y,output)))
	processes.append(mp.Process(target=C, args=(left_state.Y,output)))
	processes.append(mp.Process(target=C, args=(right_state.Y,output)))
	for p in processes:
		p.start()
	for p in processes:
		p.join()
	results = [output.get() for p in processes]
	processes = []

	big_data = results[0] #C(main_state.Y)
	big_pos_c = big_data.Positive
	big_neg_c = big_data.Negative

	small_left_data = results[1] #C(left_state.Y)
	small_left_pos_c = small_left_data.Positive
	small_left_neg_c = small_left_data.Negative

	small_right_data = results[2] #C(right_state.Y)
	small_right_pos_c = small_right_data.Positive
	small_right_neg_c = small_right_data.Negative

	U_A = U(big_pos_c, big_neg_c)
	U_AL = U(small_left_pos_c, small_left_neg_c)
	U_AR = U(small_right_pos_c, small_right_neg_c)
	p_l = P(big_pos_c, big_neg_c, small_left_pos_c, small_left_neg_c)
	p_r = P(big_pos_c, big_neg_c, small_right_pos_c, small_right_neg_c)
	return U_A - p_l * U_AL - p_r * U_AR

def train(root, type, depth_cap=20, plot=False):
	d_print("\n___\nTraining "+ type+". Max depth: "+ str(depth_cap))
	d_print(root.State.X.shape[1])
	if type == "Decision Tree" and depth_cap > 0:
		best_benefit = 0
		best_condition = None
		for i in range(1,root.State.X.shape[1]):
			for j in range(-14/4,16/4):
				condition = Condition(Feature=i, Type='<', Threshold=j*500)
				left_state, right_state = get_state(root.State, condition)
				index_benefit = B(root.State, left_state, right_state, condition)
				if (best_benefit < index_benefit):
					best_benefit = index_benefit
					best_condition = condition
					if index_benefit == 1:
						break
		root = split(root, best_condition)
		train(root.Left, "Decision Tree", depth_cap - 1, plot=plot)
		train(root.Right, "Decision Tree", depth_cap - 1, plot=plot)
				
	# come up with condition
	# compute Benefit for that condition
	# find the best benefit
	# construct a pair of childs for left and right
	# add that to the root
	# call train on each one of the two children

def main():
	#Parsing arguments:
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--plot", action='store_true', help="if set, script will show plots")
	parser.add_argument("-d", "--debug", action='store_true', help="if set, debug mode is activated")
	parser.add_argument("-v", "--verbose", action='store_true', help="if set, verbose mode is activated")
	args = parser.parse_args()
	global DEBUG
	global VERBOSE
	global PLOT
	DEBUG = args.debug
	VERBOSE = args.debug or args.verbose
	PLOT = args.plot
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
	# Part 1: Train a Tree
	train(root, "Decision Tree", plot=PLOT)
	# Part 2: Random Forest
	train(root, "Random Forest", plot=PLOT)
	# Part 3: AdaBoost
	train(root, "Adaboost", plot=PLOT)
	#cleaning up
	d_print("\n___\nCleaning up ...")
	d_print("Done.")

if __name__ == "__main__":
	main()
