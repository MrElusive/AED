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
parser.add_option("-t", dest="target", default = None, help="target class name")
parser.add_option("-r", dest="ratio", default = 0.1, type=float, help="ratio of target displayed")
parser.add_option("-s", dest="show", default=True, action='store_true', help="show new dataset")
parser.add_option("-v", dest="verbose", default=False, action='store_true', help="show all intermediate structures")
parser.add_option("-a", dest="algorithm", default="lle", help="set algorithm")
parser.add_option("-k", dest="k", default=10, type=int, help="number of nearest neighbors to use")
parser.add_option("-d", dest="outputDimensions", default=2, type=int, help="number of dimensions to output")

options, args = parser.parse_args()

if options.filename == None:
	IMAGE_BASE_DIR = "../test_images/%s"
	OUTPUT_DIR = "../arff"
	OUTPUT_FILE = "myImages.arff"

	IMAGE_SUB_DIRS = ["happy"]#, "sad", "angry", "neutral", "surprised", "scared"]


	imageFiles = []
	for subDir in IMAGE_SUB_DIRS:
		imageDir = IMAGE_BASE_DIR % subDir
		imageFiles.extend([path.join(imageDir, imageFile) for imageFile in os.listdir(imageDir) if path.isfile(path.join(imageDir, imageFile))])


	hc = cv.Load("haarcascade_frontalface_default.xml")

	dataFile = open("data.csv", "w")
	for i in range(100 * 100):
		dataFile.write(str(i))
		if i < 100 * 100 - 1:
			dataFile.write(", ")

	dataFile.write("\n")

	for imageFile in imageFiles:

		# This code was taken from http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html

		# Load data for detecting faces
		image = cv.LoadImage(imageFile, 0)

		# Grab faces
		faces = cv.HaarDetectObjects(image, hc, cv.CreateMemStorage())

		cv.NamedWindow('final-result')

		# Crop faces
		for (x, y, w, h), n in faces:
			cropped = cv.CreateImage((w, h), 8, 0)
			cv.Copy(cv.GetSubRect(image, (x, y, w, h)), cropped)
			imageStats = cv.AvgSdv(cropped)
			s2n = imageStats[0][0] / imageStats[1][0]
			
			if imageStats[1][0] > 15.0:
				print (x, y, w, h)
				print imageStats
				print s2n

				resized = cv.CreateImage((100, 100), 8, 0)
				cv.Resize(cropped, resized);

				if s2n < 5:
					smoothed = cv.CreateImage((100, 100), 8, 0)
					cv.Smooth(resized, smoothed, 3, 3)
				else:
					smoothed = resized

				edged = cv.CreateImage((100, 100), 32, 0)
				cv.Laplace(smoothed, edged, 3)

				scaled = cv.CreateImage((100, 100), 8, 0)
				cv.ConvertScaleAbs(edged, scaled)

				finalResult = scaled

				cv.ShowImage('final-result', finalResult)
				while True:
					keyPressed = chr(cv.WaitKey(0) & 255)
		
					if keyPressed == 'n':
						print "Image not accepted"
						break
					elif keyPressed == 'y':
						print "Image accepted"

						numpy.savetxt(dataFile, numpy.concatenate(numpy.asarray(cv.GetMat(finalResult)))[None], fmt="%d", delimiter=", ")
						
						break
					else:
						print "Press 'y' or 'n'"

else:
	print "Loading data set"
	# load dataset
	dataset = numpy.loadtxt(open(options.filename, 'rb'), delimiter=",", skiprows=1)#Dataset(options.filename)

	xlabel = 'first eigen vector'
	ylabel = 'second eigen vector'

	if options.algorithm == 'lle':
		print "Running LLE"
		print str(dataset)
		lle = mdp.nodes.LLENode(options.k, output_dim=options.outputDimensions)
		result = lle(dataset)

		pylab.title("After LLE")
		xlabel = 'dim 1'
		ylabel = 'dim 2'
		x, y = result[:,0], result[:,1]

	elif options.algorithm == 'hlle':
		hlle = mdp.nodes.HLLENode(options.k, output_dim=options.outputDimensions)(dataset)
		x, y = hlle[:,0], hlle[:,1]

	pylab.plot(x, y, '.')
	pylab.xlabel(xlabel)
	pylab.ylabel(ylabel)
	pylab.show() 
		
	# print new dataset
	if options.show:
		for i,(a,b) in enumerate(zip(x,y)):
			print "%s, %s" % (a, b)


'''
with open(path.join(OUTPUT_DIR, OUTPUT_FILE), 'w') as outputFile:
	for i in range(100 * 100):
		outputFile.write(str(i))
		if i != 100 * 100 - 1:
			outputFile.write(", ")
	
	outputFile.write("\n")

	for imageFile in imageFiles:
		image = Image.open(imageFile).convert('L').resize((100, 100))
		
		data = list(image.getdata())

		for i in range(len(data)):
			outputFile.write(str(data[i]))
			if i != len(data) - 1:
				outputFile.write(", ")
		
		outputFile.write("\n")
'''
