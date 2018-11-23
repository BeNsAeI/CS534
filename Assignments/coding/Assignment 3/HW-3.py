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
	pos = (Y.shape[0] - sum)/2
	neg = (Y.shape[0] - sum)/2
	if sum > 0:
		pos += abs(sum)
	else:
		neg += abs(sum)
#	if(Y.shape[0] == 0):
#		print Y
#		d_print(sum)
#		d_print(pos)
#		d_print(neg)
	out = Data(Positive=pos, Negative=neg)
	if ret_out != None:
		ret_out.put(out)
	return out

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
			if feature[i] > condition.Threshold:
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

	right_node= Node(Data=right_data,
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
	if (small_pos_c + small_neg_c == 0):
		d_print(small_pos_c)
		d_print(small_neg_c)
		exit(1)
	return 1 - (p_pos * p_pos) - (p_neg * p_neg)

# Benefit:
def B(main_state, left_state, right_state):
	B = None

	if main_state.Y.shape[0] == 0:
		d_print("main_state.Y.shape[0] is 0")
		exit(1)

	big_data = C(main_state.Y)
	big_pos_c = big_data.Positive
	big_neg_c = big_data.Negative

	small_left_data = C(left_state.Y)
	small_left_pos_c = small_left_data.Positive
	small_left_neg_c = small_left_data.Negative

	small_right_data = C(right_state.Y)
	small_right_pos_c = small_right_data.Positive
	small_right_neg_c = small_right_data.Negative

	if(big_pos_c + big_neg_c == 0):
		d_print("Main division by 0")
		d_print(main_state.Y)
		exit(1)
	if(small_left_pos_c + small_left_neg_c == 0):
		d_print("Left Division by 0")
		d_print(left_state.Y)
		exit(1)
	if (small_right_pos_c + small_right_neg_c == 0):
		d_print("Right division by 0")
		d_print(right_state.Y)
		exit(1)

	U_A = U(big_pos_c, big_neg_c)
	U_AL = U(small_left_pos_c, small_left_neg_c)
	U_AR = U(small_right_pos_c, small_right_neg_c)
	p_l = P(big_pos_c, big_neg_c, small_left_pos_c, small_left_neg_c)
	p_r = P(big_pos_c, big_neg_c, small_right_pos_c, small_right_neg_c)
	return U_A - p_l * U_AL - p_r * U_AR

def print_tree(root, count=0):
	tab = ""
	for i in range(0,count):
		tab += "\t"
	if(root.Condition != None):
		print(str(tab)+
			"Node depth: "+
			str(count)+
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
			", ["+
			str(root.Data.Positive)+
			","+
			str(root.Data.Negative)+
			"].")
	if root.Left != None:
		print_tree(root.Left, count+1)
	if root.Right != None:
		print_tree(root.Right,count+1)

def train(root, type, feature_list, depth_cap=20, plot=False):
	d_print("\n___\nTraining "+ type+". Max depth: "+ str(depth_cap))
	d_print(root.State.X.shape[0])
	if root.State.X.shape[0] == 0:
		d_print("Branch terminated.")
		return
	d_print(root.State.X.shape[1] - len(feature_list))
	if type == "Decision Tree" and depth_cap > 0:
		best_benefit = 0.0
		best_condition = None
		for i in range(0,root.State.X.shape[1]):
			if i in feature_list:
				#d_print("skipping feature "+ str(i))
				continue
			else:
				for j in range(-14/4,16/4):
					condition = Condition(Feature=i, Type='<', Threshold=j*500)
					left_state, right_state = get_state(root.State, condition)
					if left_state.Y.shape[0] == 0 or right_state.Y.shape[0] == 0:
						#d_print("skipping "+str(condition))
						continue
					else:
						index_benefit = B(root.State, left_state, right_state)
						#d_print(best_benefit)
						#d_print(index_benefit)
						if (best_benefit > index_benefit):
							#d_print(best_benefit)
							#d_print(index_benefit)
							best_benefit = index_benefit
							best_condition = condition
		if best_condition != None:
			feature_list.append(best_condition.Feature)
			root = split(root, best_condition)
			d_print("Node "+ str(20 - depth_cap) + " has " + str(best_condition))
			train_processes = []
			train_processes.append(mp.Process(target=train,
				args=(root.Left,
					"Decision Tree",
					feature_list,
					depth_cap - 1,
					plot)))
			train_processes.append(mp.Process(target=train,
				args=(root.Right,
					"Decision Tree",
					feature_list,
					depth_cap - 1,
					plot)))
			for p in train_processes:
				p.start()
			for p in train_processes:
				p.join()
			#train(root.Left, "Decision Tree", feature_list, depth_cap - 1, plot=plot)
			#train(root.Right, "Decision Tree", feature_list, depth_cap - 1, plot=plot)
		# print the tree:
		if depth_cap == 20:
			print_tree(root)
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
	feature_list = []
	# Part 1: Train a Tree
	train(root, "Decision Tree", feature_list, plot=PLOT)
	# Part 2: Random Forest
	train(root, "Random Forest", feature_list, plot=PLOT)
	# Part 3: AdaBoost
	train(root, "Adaboost", feature_list, plot=PLOT)
	#cleaning up
	d_print("\n___\nCleaning up ...")
	d_print("Done.")

if __name__ == "__main__":
	main()
