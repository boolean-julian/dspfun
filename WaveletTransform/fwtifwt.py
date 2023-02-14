import numpy as np
from numpy import sqrt, sin
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

np.set_printoptions(precision=2)

def coef_to_filter(coef):
	N = len(coef)
	return np.concatenate((coef[:0:-1], coef)), np.arange(-N+1, N)

def periodize(h,h_ind,size):
	h_per = np.zeros(size)
	for k in h_ind:
		h_per[h_ind[k]] = h[k]
	return h_per

def _cyclic_perm(h,n):
	return np.concatenate((h[-n:],h[:-n]))

def circulant(h,step=1):
	H = np.zeros((len(h)//step,len(h)))
	for j in range(0,len(h)//step):
		H[j] = _cyclic_perm(h,j*step)
	return H

decomposition_coef = 2*np.array([
	 0.557543526229,
	 0.295635881557,
	-0.028771763114,
	-0.045635881557,
	 0
])

reconstruction_coef = np.array([
	 0.602949018236,
	 0.266864118443,
	-0.078223266529,
	-0.016864118443,
	 0.026748757411
])

h_dec, h_dec_ind = coef_to_filter(decomposition_coef)
g_dec = np.array([(-1)**k*h_dec[k] for k in range(len(h_dec))])
g_dec_ind = h_dec_ind+1

h_rec, h_rec_ind = coef_to_filter(reconstruction_coef)
g_rec = np.array([(-1)**k*h_rec[k] for k in range(len(h_dec))])
g_rec_ind = h_rec_ind+1


def fwt2d(c0, J=3, h = h_dec, h_ind = h_dec_ind, g = g_dec, g_ind = g_dec_ind):
	curr_x = c0.shape[0]
	curr_y = c0.shape[1]
	curr = c0.copy()
	
	for k in range(J):
		h0_per = periodize(h,h_ind,curr_x)
		g0_per = periodize(g,g_ind,curr_x)

		h1_per = periodize(h,h_ind,curr_y)
		g1_per = periodize(g,g_ind,curr_y)

		H0 = circulant(h0_per, step=2)
		G0 = circulant(g0_per, step=2)

		H1 = circulant(h1_per, step=2)
		G1 = circulant(g1_per, step=2)

		W0 = np.vstack((H0, G0))
		W1 = np.hstack((H1.T, G1.T))
		curr[:curr_x, :curr_y] =  W0 @ curr[:curr_x, :curr_y] @ W1

		curr_x = curr_x//2
		curr_y = curr_y//2

	return curr


def ifwt2d(c0, J=3, h = h_rec, h_ind = h_rec_ind, g = g_rec, g_ind = g_rec_ind):
	denom = 2**(J-1)
	curr_x = c0.shape[0]//denom
	curr_y = c0.shape[1]//denom
	curr = c0.copy()

	for k in range(J):
		h0_per = periodize(h,h_ind,curr_x)
		g0_per = periodize(g,g_ind,curr_x)

		h1_per = periodize(h,h_ind,curr_y)
		g1_per = periodize(g,g_ind,curr_y)

		H0 = circulant(h0_per, step=2)
		G0 = circulant(g0_per, step=2)

		H1 = circulant(h1_per, step=2)
		G1 = circulant(g1_per, step=2)

		W0 = np.hstack((H0.T, G0.T))

		W1 = np.vstack((H1, G1))

		curr[:curr_x, :curr_y] =  W0 @ curr[:curr_x, :curr_y] @ W1

		curr_x = int(curr_x*2)
		curr_y = int(curr_y*2)

	return curr

c0 = np.array(Image.open("Shannon.jpg").convert('L'))[:256,:256]
c0 = c0 / 255

a0 = fwt2d(c0, J=1)
b0 = ifwt2d(a0, J=1)

plt.figure(figsize=(16,5))

plt.subplot(131)
plt.imshow(c0, vmin = 0, vmax = 1, cmap="gray")
plt.title("Input image")

plt.subplot(2,6,3)
plt.imshow(a0[:128,:128], cmap="gray")
plt.title(r"$c_{-1}$")

plt.subplot(2,6,4)
plt.imshow(a0[:128,128:], cmap="gray")
plt.title(r"$d^2_{-1}$")

plt.subplot(2,6,9)
plt.imshow(a0[128:,:128], cmap="gray")
plt.title(r"$d^1_{-1}$")

plt.subplot(2,6,10)
plt.imshow(a0[128:,128:], cmap="gray")
plt.title(r"$d^3_{-1}$")

plt.subplot(133)
plt.imshow(b0, vmin = 0, vmax = 1, cmap="gray")
plt.title("Inverse wavelet transform")

plt.tight_layout()
plt.show()