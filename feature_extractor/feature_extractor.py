#!/usr/bin/python
from pprint import pprint

import numpy
import mdp
import cv
import pylab

import os
from os import path

from mlboost.core.ppdataset import Dataset
from optparse import OptionParser

parser = OptionParser(description=__doc__)

parser.add_option("-f", dest="filename", default = None, help="filename of the data")
parser.add_option("-s", dest="show", default=False, action='store_true', help="show first two dimensions of new dataset")
parser.add_option("-a", dest="algorithm", default="lle", help="set algorithm [lle, hlle]")
parser.add_option("-k", dest="k", type=int, help="number of nearest neighbors to use")
parser.add_option("-d", dest="outputDimensions", default=2, type=int, help="number of dimensions to output")
parser.add_option("-o", dest="outputFile", default="output.arff", help="the file name for the arff output")
parser.add_option("-n", dest="numberOfInstances", help="the number of instances to use from the dataset")

options, args = parser.parse_args()

COLORS = ["r","b","g","y","m","c","b"]
LABELS = ["happy", "sad", "angry", "surprised", "scared", "disgusted", "neutral"]

def generateARFF(result, labels, output):
	output.write("@RELATION emotion\n\n")

	for i in range(len(result[0,:])):
		output.write("@ATTRIBUTE Dimension_%s CONTINUOUS\n" % i)	

	output.write("@ATTRIBUTE Emotion {%s}\n" % ", ".join(LABELS))

	output.write("\n@DATA\n")
	for i in range(len(result[:,0])):
		output.write("%s, %s\n" % (", ".join(map(str, result[i])), LABELS[int(labels[i, 0])]))


print "Loading data set..."
dataset = numpy.loadtxt(open(options.filename, 'rb'), delimiter=",")
print "Data set loaded.\n"

# Split out label column
labels = dataset[:,[len(dataset[0]) - 1]]

dataset = dataset[:,0:len(dataset[0]) - 1]

originalNumberOfInstances = len(dataset[:,0])

if options.numberOfInstances:
	labels = labels[0:int(options.numberOfInstances), :]
	dataset = dataset[0:int(options.numberOfInstances), :]

# The default value for k is the number of instances minus 2, the maximum possible value of k
if not options.k:
	options.k = len(dataset[:,0]) - 1


print "Using %d nearest neighbors." % options.k
print "Using %d out of %d instances." % (len(dataset[:,0]), originalNumberOfInstances) 
print "Source Dimensionality: %d; Target Dimensionality %d.\n" % (len(dataset[0,:]), options.outputDimensions)

if options.algorithm == 'lle':
	print "Running LLE..."
	lle = mdp.nodes.LLENode(options.k, output_dim=options.outputDimensions)
	result = lle(dataset)	

elif options.algorithm == 'hlle':
	print "Running HLLE..."
	hlle = mdp.nodes.HLLENode(options.k, output_dim=options.outputDimensions)(dataset)
	result = hlle(dataset)

print "Finished running.\n"
print "Final Dimensionality: %d.\n" % len(result[0,:])

print "Generating ARFF files..."
with open(options.outputFile, "w") as output:
	generateARFF(result, labels, output)	
print "Finished generating ARFF files.\n"

# Show the first two dimensions of the new data set.
if options.show:
	print "First Two Dimensions of Final Result:"
	x, y = numpy.append(result[:,0:1], labels, 1), numpy.append(result[:,1:2], labels, 1)

	for i, (a, b) in enumerate(zip(x, y)):
		print "%s, %s" % (a, b)

	pylab.title("Result")

	xlabel = 'dim 1'
	ylabel = 'dim 2'

	for i in range(0, 7):
		x_points = x[x[:,1] == i]
		y_points = y[y[:,1] == i]

		pylab.scatter(x_points[:,0], y_points[:,0], c=COLORS[i])
		pylab.xlabel(xlabel)
		pylab.ylabel(ylabel)
	
	pylab.show()
