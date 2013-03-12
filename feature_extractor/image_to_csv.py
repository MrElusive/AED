#!/usr/bin/python
import os
import pickle
from os import path
from optparse import OptionParser

import cv
import numpy
import yaml

HC = cv.Load("haarcascade_frontalface_default.xml")

def createUniqueOutputFile(outputFile):
	newFile = outputFile
	i = 1

	while (path.exists(newFile)):
		(pathHead, pathTail) = path.split(outputFile)
		(baseFileName, ext) = path.splitext(path.basename(pathTail))

		newFile = path.join(pathHead, "%s (%d)%s" % (baseFileName, i, ext))
		i += 1
		
	return newFile

def parseParameters(parameterFile):
	if parameterFile:
		with open(parameterFile) as parameters:
			return yaml.safe_load(parameters)
	else:
		return {
			'stddev': 15.0,
			'final-image-size': (250, 250),
			'signal-to-noise': 1,
			'smooth-type': cv.CV_GAUSSIAN,
			'param1': 3,
			'param2': 0,
			'param3': 0,
			'param4': 0,
			'aperture-size': 3
		} 

# This code is based off the tutorial at http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html
def processImageFile(imageFile, output, param, verbose, imagePreferenceFile):
	global imagePreference
	
	# Load data for detecting faces
	image = cv.LoadImage(imageFile, 0)

	# Grab faces
	faces = cv.HaarDetectObjects(image, HC, cv.CreateMemStorage())

	cv.NamedWindow('Preview Result')

	# Process faces
	for (x, y, w, h), n in faces:
		cropped = cv.CreateImage((w, h), 8, 0)
		cv.Copy(cv.GetSubRect(image, (x, y, w, h)), cropped)

		imageStats = cv.AvgSdv(cropped)

		mean = imageStats[0][0]
		stdDev = imageStats[1][0]
		signalToNoise = mean / stdDev
	
		# If the image has a standard deviation above a given threshold, then consider it to be a real face.
		if stdDev > param['stddev']:
			if (verbose):
				print (x, y, w, h)
				print imageStats
				print signalToNoise

			resized = cv.CreateImage(param['final-image-size'], 8, 0)
			cv.Resize(cropped, resized);

			# Only smooth an image if the signal-to-noise ratio is below agiven threshold
			if signalToNoise < param['signal-to-noise']:
				smoothed = cv.CreateImage(param['final-image-size'], 8, 0)
				cv.Smooth(resized, smoothed, param['smooth-type'], param['param1'], param['param2'], param['param3'], param['param4'])
			else:
				smoothed = resized

			# Discover the edges in the image
			edged = smoothed #cv.CreateImage(param['final-image-size'], 32, 0)
			#cv.Laplace(smoothed, edged, param['aperture-size'])

			# Scale in terms of bit-depth, not image size
			scaled = cv.CreateImage(param['final-image-size'], 8, 0)
			cv.ConvertScaleAbs(edged, scaled)

			finalResult = scaled

			cv.ShowImage('Preview Result', finalResult)
			while True:
				if (imageFile, n) in imagePreference:
					keyPressed = imagePreference[(imageFile, n)]
				else:
					keyPressed = chr(cv.WaitKey(0) & 255)
	
				if keyPressed == 'n' or keyPressed == 'N':
					print "Image skipped!"
					break
				elif keyPressed == 'y' or keyPressed == 'Y':
					imageLabel = int(path.basename(imageFile)[0:1])
					imageArray = numpy.asarray(cv.GetMat(finalResult))
					imageArray = numpy.concatenate(imageArray)
					imageArray = numpy.append(imageArray, imageLabel)
					numpy.savetxt(output, imageArray[None], fmt="%d", delimiter=", ")
					print "Image added!"
					
					break
				else:
					print "Please enter either 'y'/'Y' or 'n'/'N'"


			imagePreference[(imageFile, n)] = keyPressed 
			pickle.dump(imagePreference, file(imagePreferenceFile, 'w'))

parser = OptionParser(description=__doc__)

parser.add_option("-i", dest="imageFolder", default="../images", help="image folder")
parser.add_option("-o", dest="outputCSVFile", default="output.csv", help="output csv file")
parser.add_option("-v", dest="verbose", default=False, action='store_true', help="show verbose output")
parser.add_option("-c", dest="clobber", default=False, action='store_true', help="always clobber the output file")
parser.add_option("-p", dest="parameterFile", default=None, help="designate the parameters for processing the image")
parser.add_option("-r", dest="imagePreferenceFile", default=None, help="keeps track of which images are accepted and which are not")

options, args = parser.parse_args()

imageDirWorkList = [options.imageFolder]
outputFile = options.outputCSVFile

if not options.clobber:
	outputFile = createUniqueOutputFile(outputFile)

param = parseParameters(options.parameterFile)

if options.imagePreferenceFile and path.exists(options.imagePreferenceFile):
	try:
		imagePreference = pickle.load(file(options.imagePreferenceFile))
	except pickle.UnpicklingError:
		print "Unable to unpickle the file '%s'." % options.imagePreferenceFile
		exit(1)
else:
	imagePreference = {}

with open(outputFile, 'w') as output:
	while imageDirWorkList:
		imageDir = imageDirWorkList.pop()

		for dirObject in os.listdir(imageDir):
			dirObject = path.join(imageDir, dirObject)
			if path.isdir(dirObject):
				imageDirWorkList.append(dirObject)
			elif path.isfile(dirObject):
				processImageFile(dirObject, output, param, options.verbose, options.imagePreferenceFile)
			else:
				print "Encountered something that is neither a file or directory: %s" % dirObject

