from PIL import Image
from PIL import ImageFilter
import os
from os import path
import cv

IMAGE_DIR = "../test_images/%s"
HAPPY_DIR = IMAGE_DIR % "happy"
SAD_DIR = IMAGE_DIR % "sad"
ANGRY_DIR = IMAGE_DIR % "angry"
NEUTRAL_DIR = IMAGE_DIR % "neutral"
SURPRISED_DIR = IMAGE_DIR % "surprised"
SCARED_DIR = IMAGE_DIR % "scared"

OUTPUT_DIR = "./csv_output"

imageFiles = []
imageFiles.extend([path.join(HAPPY_DIR, imageFile) for imageFile in os.listdir(HAPPY_DIR) if path.isfile(path.join(HAPPY_DIR, imageFile))])
imageFiles.extend([path.join(SAD_DIR, imageFile) for imageFile in os.listdir(SAD_DIR) if path.isfile(path.join(SAD_DIR, imageFile))])
imageFiles.extend([path.join(ANGRY_DIR, imageFile) for imageFile in os.listdir(ANGRY_DIR) if path.isfile(path.join(ANGRY_DIR, imageFile))])
imageFiles.extend([path.join(NEUTRAL_DIR, imageFile) for imageFile in os.listdir(NEUTRAL_DIR) if path.isfile(path.join(NEUTRAL_DIR, imageFile))])
imageFiles.extend([path.join(SURPRISED_DIR, imageFile) for imageFile in os.listdir(SURPRISED_DIR) if path.isfile(path.join(SURPRISED_DIR, imageFile))])
imageFiles.extend([path.join(SCARED_DIR, imageFile) for imageFile in os.listdir(SCARED_DIR) if path.isfile(path.join(SCARED_DIR, imageFile))])

for imageFile in imageFiles:
	#image = Image.open(imageFile).convert('L').resize((100, 100))
	image = Image.open(imageFile).convert('L')
	image.thumbnail((100, 100), Image.ANTIALIAS)
	
	cvImage = cv.CreateImageHeader(image.size, cv.IPL_DEPTH_8U, 1)
	cv.SetData(cvImage, image.tostring())	

	hc = cv.Load("haarcascade_frontalface_default.xml")
	faces = cv.HaarDetectObjects(cvImage, hc, cv.CreateMemStorage())

	try:
		(x, y, w, h) = faces[0][0]
		image = image.crop((x, y, x + w, y + h))
		image.thumbnail((100, 100), Image.ANTIALIAS)
	except:
		pass
	
	image.show()

	count = 1
	y_hash = 0
	for x in range(image.size[0]):
		for y in range(image.size[1]):
			y_hash += count * image.getpixel((x, y))
			count += 1
	
	count = 1
	x_hash = 0
	for y in range(image.size[1]):
		for x in range(image.size[0]):
			x_hash += count * image.getpixel((x, y))
			count += 1

	print "%s, %s" % (x_hash, y_hash)
