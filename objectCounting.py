import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--filename", help="data file name")
args = parser.parse_args()

img_file = args.filename

img = cv2.imread(img_file, 0)
cv2.imshow('original', img)

# remove sine noise in frequence domain 
if 'sinus' in img_file:
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 8
    mask[crow, ccol-8] = 0
    mask[crow, ccol+8] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)/np.prod(img.shape)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img = np.array(img_back, dtype=np.uint8)    
    cv2.imshow('remove sine noise', img)

# gamma correction to brighten img with gamma = 0.2
if img.max()<150:
    gamma_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gamma_img[i, j] = np.math.pow(img[i, j], 0.2)
    cv2.normalize(gamma_img, gamma_img, 0, 255, cv2.NORM_MINMAX)
    img= np.array(gamma_img, dtype=np.uint8)
    cv2.imshow('gamma', img)
    
# smoothing
img = cv2.medianBlur(img, 5)

# histogram equalization
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
b = clahe.apply(img)

# threshold
img_binary = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)[1]

# erode with kernel size (3, 3)
kernel = np.ones((3, 3))
rs = cv2.erode(img_binary, kernel)

# connected component
num_labels, labels = cv2.connectedComponents(rs)

# remove small objects
num = num_labels
for i in range(1, num_labels):
  if np.sum(labels==i)<100:
    num-=1

print(f'The number of objects in the image ({img_file}) :{num-1: 5d}')

cv2.imshow('post  process', b)
cv2.imshow('threshold', img_binary)
cv2.imshow('final', rs)

cv2.waitKey(0)
cv2.destroyAllWindows()

