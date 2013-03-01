from PIL import Image
import os
from os import path

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

with open(path.join(OUTPUT_DIR, "images.csv"), 'w') as outputFile:
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
