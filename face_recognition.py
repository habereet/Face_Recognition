import cv2
import sys

from config import CASCPATH

faceCascade = cv2.CascadeClassifier(CASCPATH)

for file in sys.argv[1:]:

	image = cv2.imread(file)
	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray_scale,
		scaleFactor=1.1,
		minNeighbors=15,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)

	print("Found " + str(len(faces)) + " faces in " + file)

	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow("Faces found", image)
	cv2.waitKey(0)