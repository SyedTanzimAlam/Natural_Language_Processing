import cv2
import sys
import fitz
doc = fitz.open(r'C:\Users\tehzeeb\Downloads\FaceDetect-master\FaceDetect-master\Doc3.pdf')
for i in range(len(doc)):
    for img in doc.getPageImageList(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:       # this is GRAY or RGB
            pix.writePNG("p%s.png" % (i))
        else:               # CMYK: convert to RGB first
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("p%s.png" % (i))
            pix1 = None
        pix = None

# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(r'C:\Users\tehzeeb\Downloads\FaceDetect-master\FaceDetect-master\p0.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(3, 3),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	a=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
	print(a.shape)
	print(x)
	print(y)
	print(x+w)
	print(y+h)
	cv2.imwrite("Faces found.PNG",a)
	
	
cv2.imshow("Faces found", a)
cv2.waitKey(5000)


##########################################################################################


from imutils import paths
import numpy as np
import imutils
import cv2

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# loop over the input image paths

# load the image, resize it, and convert it to grayscale
image = cv2.imread("Faces found.PNG")
print("LOADING IMAGE")
#h,w,_=image.shape
#image= image[h//2:h,0:w]
image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# smooth the image using a 3x3 Gaussian, then apply the blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# serieso of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=4)

# during thresholding, it's possible that border pixels were
# included in the thresholding, so let's set 5% of the left and
# right borders to zero
p = int(image.shape[1] * 0.05)
thresh[:, 0:p] = 0
thresh[:, image.shape[1] - p:] = 0

# find contours in the thresholded image and sort them by their
# size
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and use the contour to
	# compute the aspect ratio and coverage ratio of the bounding box
	# width to the width of the image
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	crWidth = w / float(gray.shape[1])

	# check to see if the aspect ratio and coverage width are within
	# acceptable criteria
	if ar > 5 and crWidth > 0.75:
		# pad the bounding box since we applied erosions and now need
		# to re-grow it
		pX = int((x + w) * 0.03)
		pY = int((y + h) * 0.03)
		(x, y) = (x - pX, y - pY)
		(w, h) = (w + (pX * 2), h + (pY * 2))

		# extract the ROI from the image and draw a bounding box
		# surrounding the MRZ
		roi = image[y:y + h, x:x + w].copy()
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		break

# show the output images
cv2.imshow("Image", image)
cv2.imshow("ROI", roi)
cv2.waitKey(5000)