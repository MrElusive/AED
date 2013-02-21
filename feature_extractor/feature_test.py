# PIL is an imaging library that gives us some basic filters and image processing methods
from PIL import Image
from PIL import ImageFilter
import os
import cv

IMAGE_DIR = "../test_images"

# Grab the names of all the image files in IMAGE_DIR
imageFiles = [imageFile for imageFile in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, imageFile))]

"""
# Apply and show the effect of the filters below
for imageFile in imageFiles:
	try:
		image = Image.open(os.path.join(IMAGE_DIR, imageFile)).resize((512, 1024), Image.ANTIALIAS)

		image.filter(ImageFilter.FIND_EDGES).show()
		image.filter(ImageFilter.CONTOUR).show()
		image.filter(ImageFilter.DETAIL).filter(ImageFilter.FIND_EDGES).show()
		image.filter(ImageFilter.EDGE_ENHANCE).filter(ImageFilter.FIND_EDGES).show()
		image.filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES).show()
	except IOError:
		print("%s is not a valid image file" % imageFile)

	break
"""

# Apply a contour filter to the first image and save it to a file so we can use it in the cv library
contourImage = Image.open(os.path.join(IMAGE_DIR, imageFiles[0])).filter(ImageFilter.CONTOUR)
contourImage.save("contourImage.jpg")

cv.NamedWindow('window', cv.CV_WINDOW_AUTOSIZE)
image = cv.LoadImageM("contourImage.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)

# Extract and print SURF features.  We should think about how we can feed these features into a machine learner.
(keypoints, descriptors) = cv.ExtractSURF(image, None, cv.CreateMemStorage(), (0, 1000, 3, 1))

print len(keypoints), len(descriptors)
for ((x, y), laplacian, size, dir, hessian) in keypoints:
	cv.Circle(image, (int(x), int(y)), size, 0)
	print "x=%d y=%d laplacian=%d size=%d dir=%f hessian=%f" % (x, y, laplacian, size, dir, hessian)

cv.ShowImage('a_window', image)
cv.WaitKey(15000)
