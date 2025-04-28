import cv2
import numpy as np


image = cv2.imread('gugu.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(image, blockSize=6, ksize=1, k=0.04)

harris = cv2.dilate(harris, None)

image[harris > 0.01 * harris.max()] = [255]

cv2.imshow('Harris Corner', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
