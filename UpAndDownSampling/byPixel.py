from itertools import chain
import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

""" NOTES:
def upsample2D_slow(img, lagrange2D, m1, m2):
	l1, l2 = img.shape
	upsampled = np.zeros(np.array([*img.shape])*np.array([m1,m2]))
	u1, u2 = upsampled.shape
	for x in range(u1):
		for y in range(u2):
			for i in range(l1):
				for j in range(l2):
					upsampled[x,y] += img[i,j] * lagrange2D(x/m1-i, y/m2-j)


#Alternatively, we can write above function like this, which helps with converting this to a FFT procedure
def upsample2D_alt(img, lagrange2D, m1, m2)
	for k1 in range(m1):
		for k2 in range(m2):
			for i1 in range(l1):
				for i2 in range(l2):
					for j1 in range(l1):
						for j2 in range(l2):
							upsampled[k1+i1*m1,k2+i2*m2] += img[j1,j2] * lagrange2D(k1/m1+i1-j1, k2/m2+i2-j2)
	
	return upsampled
"""

# From Ex2. Support only on [-2,2]
def lagrange(x):
	x = max(x,-x)
	if 0 <= x and x <= 1:
		return x**3 - 2*x**2 + 1
	elif 1 < x and x <= 2:
		return -x**3 + 5*x**2 - 8*x + 4
	else:
		return 0

def lagrange2D(x,y):
	return lagrange(x)*lagrange(y)

def downsample2D(img, m1, m2):
	return img[::m1,::m2]

# derived from the commented functions above
def upsample2D(img, lagrange2D, m1, m2):
	l1, l2 = img.shape
	upsampled = np.zeros(np.array([*img.shape])*np.array([m1,m2]))

	r1m1 = range(-l1//2,0) 
	r2m1 = range(l1//2)

	r1m2 = range(-l2//2,0)
	r2m2 = range(l2//2)

	for k1 in range(m1):
		for k2 in range(m2):
			_filter = np.array([[lagrange2D(k1/m1 + i1, k2/m2 + i2) for i2 in chain(r2m2,r1m2)] for i1 in chain(r2m1,r1m1)])
			upsampled[k1::m1, k2::m2] = np.real(ifft2(fft2(_filter)*fft2(img)))

	return upsampled

img = np.array(Image.open("Shannon.jpg").convert("L"))/255 # open + convert to grayscale image; intensity ranging from 0 to 1
m1, m2 = 2,2

downsampled = downsample2D(img,m1,m2)
upsampled = upsample2D(downsampled,lagrange2D,m1,m2)

def printMSE(orig, comp):
	o1,o2 = orig.shape
	mse = np.sum((orig-comp[:o1,:o2])**2)/len(orig)
	print("MSE:", mse)
	return mse

printMSE(img, upsampled)

# Plot
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.title("Original")
plt.imshow(img,			cmap="gray", vmin=0, vmax=1)

plt.subplot(132)
plt.title("Downsampled")
plt.imshow(downsampled,	cmap="gray", vmin=0, vmax=1)

plt.subplot(133)
plt.title("Upsampled")
plt.imshow(upsampled,	cmap="gray", vmin=0, vmax=1)

plt.tight_layout()

plt.savefig("Exercise02sol-5.pdf")