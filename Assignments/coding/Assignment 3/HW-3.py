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
# Data structure
## myData = Data(55, 20)
## Or
## myData = Data(Left=55, Right=20)
Data = namedtuple("Data", "Left Right")
# A convinient Tree structure
# Example Declearation:
## myTree = Tree(0.5, 0x---, 0x---, 0x---)
## Or
## myTree = Tree(Data=0.5, Parent=0x---, Left=0x---, Right=0x---)
Tree = namedtuple("Tree", "Data Parent Left Right")

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

def count():

def benefit():

def train(root, x, y, plot):
	d_print("\n___\nTraining...")
	

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
	root_data = None
	d_print("Initiating the Tree")
	root = Tree(Data=root_data, Parent=None, Left=None, Right=None)
	d_print("Root: "+str(root))
	train(root, x_train, y_train, PLOT)
	#cleaning up
	d_print("\n___\nCleaning up ...")
	d_print("Done.")

if __name__ == "__main__":
	main()
