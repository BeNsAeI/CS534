import os
import argparse
import numpy as np
from collections import namedtuple

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
## myData = Data(Left=55, Right=20)
Data = namedtuple("Data", "Left Right")

# i Branch condition is x_6 < 3:
## myCondition = Data(6, '<', 3)
##Or
## myCondition = Data(Feature=6, Type='<', Threshold=3)
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
	x = np.insert(x , [1], 1, axis=1)
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
	out = None
	left = 0
	right = 0
	for i in Y:
		if i in Y:
			if i < 0:
				left +=1
			else:
				right += 1
	out = Data(Left=left, Right=right)
	return out

def split(node, condition):
	left_state = None
	left_data = None
	left_condition = condition

	right_state = None
	right_data = None
	right_condition = None # Calculate right condition

#	feature = X[:, condition.Feature]
#	new_X = None
#	new_Y = None
#	if condition.Type == '<':
#		for i in range(0,feature.size()):
#			if feature[i] < condition.Threshold:
#				new_X = np.insert(new_X, 0, X[i,:], axis=0)
#				new_Y = np.insert(new_Y, 0, Y[i])
#	if condition.Type == '>':
#		for i in range(0,feature.size()):
#			if feature[i] > condition.Threshold:
#				new_X = np.insert(new_X, 0, X[i,:], axis=0)
#				new_Y = np.insert(new_Y, 0, Y[i])
#	if condition.Type == '==':
#		for i in range(0,feature.size()):
#			if feature[i] == condition.Threshold:
#				new_X = np.insert(new_X, 0, X[i,:], axis=0)
#				new_Y = np.insert(new_Y, 0, Y[i])
#	X, Y, out = C(new_X, new_Y)

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

# Benefit:
def B(left_state, right_state, condition):
	B = None
	XL = None
	XR = None
	YL = None
	YR = None
	return B

def train(root, type, depth_cap=20, plot=False):
	d_print("\n___\nTraining "+ type+". Max depth: "+ str(depth_cap))
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
