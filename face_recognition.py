# Based on and modifed from https://realpython.com/face-recognition-with-python/

import cv2
import sys

# import the path to face cascade definitions from configuration
from config import CASCPATH

# Set up faceCascade
# Load the face cascade into memory
faceCascade = cv2.CascadeClassifier(CASCPATH)

# For each file passed in the command line do the following
for file in sys.argv[1:]:

	# Open the file and save it to an opencv image
	image = cv2.imread(file)
	# Convert the above opencv image to grayscale
	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Search the image (grayscale version)
	#   scaleFactor - allows us to compensate for images different distances from the camera
	#   minNeighbors - allows us to eliminate false positives by saying we have to have several positives in the same area
	#      more info on minNeighbors - https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter
	#   minSize - defines the size of the box to use
	#   flags - need to look for more information on this
	faces = faceCascade.detectMultiScale(
		gray_scale,
		scaleFactor=1.1,
		minNeighbors=15,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	# Output the number of faces found in the image
	print(f"Found {len(faces)} face{'' if len(faces) == 1 else 's'} in {file}")

	for (x, y, w, h) in faces:
		# Draw a rectangle around each face
		#   Look for more information on rectangle
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display the image
	cv2.imshow("Faces found", image)
	# Wait for a button press to close the image
	cv2.waitKey(0)