import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, f8, c16

start_value = 0
threshold = 2
max_iterations = 100
num_samples = 3000

@vectorize([f8(c16)])
def f(c):
	# Start value
	z = start_value

	# Iteration
	k = 0
	while np.abs(z) <= threshold:
		z = z**2+c
		if (k > max_iterations):
			break
		k += 1

	return k/max_iterations

# 2d conversion
def px(x,y):
	return f(x+1j*y)

# Plot
figure = plt.figure(figsize=(12,12), frameon=False)
axis = plt.Axes(figure, [0., 0., 1., 1.])

figure.add_axes(axis)
axis.axis('off')

xs = np.linspace(-2, 0.5, num_samples)
ys = np.linspace(-1.25, 1.25, num_samples)

X, Y = np.meshgrid(xs, ys)

Z = px(X, Y)

plt.contourf(X, Y, Z, cmap="binary")
plt.savefig("mandelbrot_set.png")