import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--filename", help="image file name")
args = parser.parse_args()

img_file = args.filename
# img_file = 'D:/Desktop/computer_vision/objets4.jpg'

# read grayscale image
img = cv2.imread(img_file, 0)
plt.imshow( img, cmap='gray')
plt.title(img_file)

# smoothing with median filter
img = cv2.medianBlur(img, 5)

# threshold img with adaptive thresh gaussian
img_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 289, 10)

# morphology close with negative image: Dilation followed by Erosion with kernel size (15, 15)
kernel = np.ones((15, 15), np.uint8)
rs = cv2.morphologyEx(~img_binary, cv2.MORPH_CLOSE, kernel)
plt.figure()
plt.imshow(rs, cmap='gray')
plt.title('thresholded')

# connected component to count objects
num_labels, labels = cv2.connectedComponents(rs)

# get rid of the objects with area <150 and print result
num = num_labels
for i in range(1, num_labels):
  if np.sum(labels==i)<150:
    num-=1
print(f'The number of objects in the image ({img_file}) :{num-1: 5d}')

# print area of each object
area=[]
for i in range(num_labels):
    # plt.figure()
    # plt.imshow(labels==i, cmap='gray')
    area.append((labels==i).sum())
area.sort(reverse=True)
print(f'area : {area}')

plt.show()
