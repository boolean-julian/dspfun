from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

# input
nbins = 5
img = np.array(Image.open("Shannon.jpg").convert("L")) # open + convert to grayscale image; intensity ranging from 0 to 1

# create histogram
hist = np.zeros(256)
for datum in img.flatten():
	hist[datum] += 1
hist = hist/np.sum(hist)

# create pdf-optimal bins ("decision levels")
bins = []
curr_color = int(np.min(img))
for k in range(nbins):
	bins.append(curr_color)
	curr_sum = 0
	for color in range(curr_color, len(hist)):
		curr_sum += hist[color]
		curr_color = color
		if curr_sum > 1/nbins:
			break
bins.append(int(np.max(img)))

print("Quantization intervals:", bins) # Quantization intervals: [0, 19, 46, 106, 204, 255]

# reconstruction values
r = np.zeros(nbins)
for qlvl in range(len(bins)-1):
	for color in range(bins[qlvl], bins[qlvl+1]+1):
		#print(str(bins[qlvl]) + "\t" + str(color) + "\t" + str(hist[color]))
		r[qlvl] += hist[color]*color
r *= nbins

print("Reconstruction values:", r) # Reconstruction values: [  7.86  32.85  72.97 167.29 245.6 ]

# reconstruct image with nbins colors
reconstructed = np.zeros(img.shape)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		for qlvl in range(len(bins)-1):
			if bins[qlvl] <= img[i,j] < bins[qlvl+1]+1:
				reconstructed[i,j] = r[qlvl]
				break

# plot histograms
plt.figure(figsize=(8,4))

plt.subplot(131)
plt.title("Original image")
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

plt.subplot(132)
plt.title("PDF-optimized bins")
plt.hist(img.flatten(), density=True, bins=bins, edgecolor="black", linewidth=0.5, color="gray")

# plot images
plt.subplot(133)
plt.title("Reconstructed image")
plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)

plt.tight_layout()

plt.show()