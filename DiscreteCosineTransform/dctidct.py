import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.set_printoptions(precision=2, linewidth=250, threshold = 100)
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

###############################################################
# Shared functions for all exercises						  #
###############################################################

# implementation of 2d-dct2 via fast dct2
def dct2_2d(A):
	B = np.zeros(A.shape)
	C = np.zeros(A.shape)

	for i in range(A.shape[0]):
		B[i] = dct2(A[i])

	for j in range(A.shape[1]):
		C[:,j] = dct2(B[:,j])

	return C

# implementation of 2d-idct2 via fast idct2
def idct2_2d(A):
	B = np.zeros(A.shape)
	C = np.zeros(A.shape)

	for i in range(A.shape[0]):
		B[i] = idct2(A[i])

	for j in range(A.shape[1]):
		C[:,j] = idct2(B[:,j])

	return C

# fast dct4 (via dct2)
def dct4(x):
	N = len(x)

	if N == 2:
		return np.sqrt(2) * slowdct4()@x

	u = T1(N) @ x
	v = np.concatenate((dct2(u[:N//2]),dct2(u[N//2:])))
	y = A1(N) @ v

	return P(N).T @ y

# (slow) implementation of dct4 for the base case N = 2
def slowdct4(N=2):
	T = np.zeros((N,N))
	for j in range(N):
		for k in range(N):
			T[j,k] = np.cos((2*k+1)*(2*j+1)*np.pi/(4*N))
	return np.sqrt(2/N) * T

# T_N(1) matrix from script with the diagonal and counter diagonal cosine and sine terms.
# used to calculate the fast dct4
def T1(N):
	N1 = N//2

	# assemble sin/cos matrix
	cos 	= np.array([np.cos((2*k+1) * np.pi/(4*N)) for k in range(N1)])
	sin 	= np.array([np.sin((2*k+1) * np.pi/(4*N)) for k in range(N1)]) 
	C 		= np.diag(np.concatenate((cos,cos[::-1]))) + np.diag(np.concatenate((-sin,sin[::-1])))[::-1]
	
	# assemble diagonal matrix with 1s and -1s
	d1 		= np.ones(N1)
	d2 		= np.array([1 if i%2==0 else -1 for i in range(N1)])
	D 		= np.diag(np.concatenate((d1,d2)))

	return D@C

# A_N(1) matrix from script with the three diagonals. used for the fast dct4
# the factor 1/sqrt(2) was left away to enhance performance
def A1(N):
	N1 = N//2

	# assemble matrix with 3 diagonals
	ones = np.ones(N1-1)
	padded_ones = np.concatenate(([0],ones,[0]))

	left 	= np.diag(padded_ones, k=N1-1)
	right	= np.diag(padded_ones, k=1-N1)
	center 	= np.diag(np.concatenate(([np.sqrt(2)],ones,-ones,[-np.sqrt(2)])))
	
	lcr		= left + center + right

	# assemble permutation matrix
	perm = np.diag(np.concatenate((np.ones(N1), np.zeros(N1))))
	flip = np.diag([1 if i%2==1 else -1 for i in range(N1)])[::-1]
	perm[N1:, N1:] = flip

	return lcr@perm

# implementation of fast dct2 (via dct4)
def dct2(x):
	N = len(x)

	if N == 2:
		return np.array([x[0]+x[1], x[0]-x[1]])

	u = T0(N)@x
	v = np.concatenate((dct2(u[:N//2]),dct4(u[N//2:])))

	return P(N).T @ v

# implementation of fast idct2 (via dct4)
def idct2(x):
	N = len(x)

	if N == 2:
		return np.array([x[0]+x[1], x[0]-x[1]])

	u = P(N)@x
	v = np.concatenate((idct2(u[:N//2]),dct4(u[N//2:])))

	return T0(N).T @ v

# T_N(0) matrix from script that has the two V shapes in it. used for the fast dct2
# the factor 1/sqrt(2) was left away to enhance performance
def T0(N):
	N1 = N//2
	
	T = np.eye(N)
	T[N1:,N1:] = -T[N1:,N1:][::-1]
	T[N1:,:N1] = T[:N1,:N1]
	T[:N1,N1:] = -T[N1:,N1:]

	return T

# permutation matrix (last step in dct2 and dct4)
def P(N):
	A = np.concatenate((np.eye(N)[::2],np.eye(N)[1::2]))
	return A

###############################################################
# Exercise 17 												  #
###############################################################

fig, ax = plt.subplots(2, figsize=(12,8))

xa = np.arange(2**8)
xb = np.array([1 if i%2==0 else -1 for i in range(512)])

fa = 1/np.sqrt(len(xa))
fb = 1/np.sqrt(len(xb))

ya = fa*idct2(xa) # remember: idct2 == dct2.T == dct3
yb = fb*idct2(xb)

print("Exercise 17")
print("Output from a:", ya)
print("Output from b:", yb)
print("\nCheck results by applying inverse transform:")
print("Output from a:", fa*dct2(ya))
print("Output from b:", fb*dct2(yb))

ax[0].plot(ya, linewidth=0.8, color="cornflowerblue")
ax[0].set_title("(a)")
ax[1].plot(yb, linewidth=0.8, color="cornflowerblue")
ax[1].set_title("(b)")

plt.tight_layout()

plt.savefig("17.pdf")
print("\n")
"""
Exercise 17
Output from a: [ 1.34e+03 -1.49e+03  6.44e+02 ... -1.96e-01  8.20e-02 -5.68e-02]
Output from b: [ 1.29e-02  1.31e-02  1.27e-02 ...  4.09e+00 -6.78e+00  2.04e+01]

Check results by applying inverse transform:
Output from a: [-2.84e-14  1.00e+00  2.00e+00 ...  2.53e+02  2.54e+02  2.55e+02]
Output from b: [ 1. -1.  1. ... -1.  1. -1.]
"""


###############################################################
# Exercise 18 												  #
###############################################################

A = np.array([
	[11, 16, 21, 25, 27, 27, 27, 27],
	[16, 23, 25, 28, 31, 28, 28, 28],
	[22, 27, 32, 35, 30, 28, 28, 28],
	[31, 33, 34, 32, 32, 31, 31, 31],
	[31, 32, 33, 34, 34, 27, 27, 27],
	[33, 33, 33, 33, 32, 29, 29, 29],
	[34, 34, 33, 35, 34, 29, 29, 29],
	[34, 34, 33, 33, 35, 30, 30, 30]
])

print("Exercise 18")
print("input matrix:")
print(A)

fac = 1/np.product(A.shape)

dctA = np.sqrt(fac) * dct2_2d(A)
print("\ndct2(A):")
print(dctA)

B = np.sqrt(fac) * idct2_2d(dctA)
print("\nidct2(dct2(A)) (to check whether result is correct):")
print(B)

"""
Exercise 18
input matrix:
[[11 16 21 25 27 27 27 27]
 [16 23 25 28 31 28 28 28]
 [22 27 32 35 30 28 28 28]
 [31 33 34 32 32 31 31 31]
 [31 32 33 34 34 27 27 27]
 [33 33 33 33 32 29 29 29]
 [34 34 33 35 34 29 29 29]
 [34 34 33 33 35 30 30 30]]

dct2(A):
[[ 2.36e+02 -1.03e+00 -1.21e+01 -5.20e+00  2.12e+00 -1.67e+00 -2.71e+00  1.32e+00]
 [-2.26e+01 -1.75e+01 -6.24e+00 -3.16e+00 -2.86e+00 -6.95e-02  4.34e-01 -1.19e+00]
 [-1.09e+01 -9.26e+00 -1.58e+00  1.53e+00  2.03e-01 -9.42e-01 -5.67e-01 -6.29e-02]
 [-7.08e+00 -1.91e+00  2.25e-01  1.45e+00  8.96e-01 -7.99e-02 -4.23e-02  3.32e-01]
 [-6.25e-01 -8.38e-01  1.47e+00  1.56e+00 -1.25e-01 -6.61e-01  6.09e-01  1.28e+00]
 [ 1.75e+00 -2.03e-01  1.62e+00 -3.42e-01 -7.76e-01  1.48e+00  1.04e+00 -9.93e-01]
 [-1.28e+00 -3.60e-01 -3.17e-01 -1.46e+00 -4.90e-01  1.73e+00  1.08e+00 -7.61e-01]
 [-2.60e+00  1.55e+00 -3.76e+00 -1.84e+00  1.87e+00  1.21e+00 -5.68e-01 -4.46e-01]]

idct2(dct2(A)) (to check whether result is correct):
[[11. 16. 21. 25. 27. 27. 27. 27.]
 [16. 23. 25. 28. 31. 28. 28. 28.]
 [22. 27. 32. 35. 30. 28. 28. 28.]
 [31. 33. 34. 32. 32. 31. 31. 31.]
 [31. 32. 33. 34. 34. 27. 27. 27.]
 [33. 33. 33. 33. 32. 29. 29. 29.]
 [34. 34. 33. 35. 34. 29. 29. 29.]
 [34. 34. 33. 33. 35. 30. 30. 30.]]
 """


###############################################################
# Exercise 19 												  #
###############################################################

def get_quantization_matrix(blocksize=8):
	q = np.zeros((blocksize, blocksize))
	for j in range(blocksize):
		for k in range(blocksize):
			if j+k <= 3:
				q[j,k] = 1
	return q

def blockdiv(A,blocksize=8):
	heigt, width = A.shape
	nheigt, rheigt = heigt//blocksize, heigt%blocksize
	nwidth, rwidth = width//blocksize, width%blocksize
	
	twidth = 0
	theigt = 0

	if rheigt:
		nheigt += 1
		theigt = blocksize-rheigt
		A = np.concatenate((A,np.zeros((theigt, A.shape[1]))))

	if rwidth:
		nwidth += 1
		twidth = blocksize-rwidth
		A = np.concatenate((A.T,np.zeros((A.shape[0], twidth)).T)).T

	blocks = np.zeros((nheigt,nwidth,blocksize,blocksize))
	for j in range(nheigt):
		for k in range(nwidth):
			blocks[j,k,:,:] = A[j*blocksize:(j+1)*blocksize, k*blocksize:(k+1)*blocksize]

	return blocks, blocksize-rheigt, blocksize-rwidth

def blockmerge(blocks, theigt=0, twidth=0):
	blocksize = blocks.shape[-1]
	
	nheigt = blocks.shape[0]
	nwidth = blocks.shape[1]

	heigt = nheigt * blocksize
	width = nwidth * blocksize

	M = np.zeros((heigt, width))

	for j in range(nheigt):
		for k in range(nwidth):
			M[j*blocksize:(j+1)*blocksize,k*blocksize:(k+1)*blocksize] = blocks[j,k,:,:]

	return M[:heigt-theigt, :width-twidth]

def compress(img, blocksize = 8):
	blocks, _th, _tw = blockdiv(img, blocksize)
	
	Q = get_quantization_matrix(blocksize)

	dctblocks = np.zeros(blocks.shape)
	for j in range(blocks.shape[0]):
		for k in range(blocks.shape[1]):
			dctblocks[j,k] = Q*dct2_2d(blocks[j,k]) # element-wise multiplication
	dctblocks = 1/blocksize * dctblocks

	return dctblocks, _th, _tw

def reconstruct(blocks, _th, _tw):
	idctblocks = np.zeros(blocks.shape)
	for j in range(blocks.shape[0]):
		for k in range(blocks.shape[1]):
			idctblocks[j,k] = idct2_2d(blocks[j,k])

	idctblocks = 1/blocks.shape[-1] * idctblocks

	return blockmerge(idctblocks, _th, _tw)


img = np.array(Image.open("Shannon.jpg").convert("L"))
jpg, _th, _tw = compress(img)
rec = reconstruct(jpg, _th, _tw)

fig, ax = plt.subplots(1,2, figsize=(9,6))
ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original image")
ax[1].imshow(rec, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Reconstructed image")

plt.tight_layout()

plt.savefig("19.pdf")