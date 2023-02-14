import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, fft, fftfreq

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

samples = 1000
ts = np.linspace(-3*np.pi, 3*np.pi, samples)

def f(t):
	return 0.1*np.cos(30*t)+0.9*np.cos(t)

# signal
ys = f(ts)
# fourier transformed signal
fs = fft(ys)
# sample frequencies
ws = fftfreq(samples)*np.pi
w0 = np.pi/20

# filter 1
idx = np.where(np.abs(ws) > w0)[0]
fs2 = fs.copy()
fs2[idx] = 0

# inverse transform
ys2 = ifft(fs2)

# filter 2
idy = np.where(np.abs(ws) < w0)[0]
fs3 = fs.copy()
fs3[idy] = 0

# inverse transform
ys3 = ifft(fs3)


##################
fig, axs = plt.subplots(3,2, figsize=(16,9))

axs[0,0].plot(ts,ys,color="cornflowerblue")
axs[0,0].set_title("Original signal")

axs[0,1].plot(ws,np.abs(fs), color="seagreen")
axs[0,1].axvline(w0, linestyle="dashed", color="lightcoral")
axs[0,1].axvline(-w0, linestyle="dashed", color="lightcoral")
axs[0,1].set_title("Fourier transform")

axs[1,1].plot(ws,np.abs(fs2), color="seagreen")
axs[1,1].set_title("Filtered transform (low pass)")

axs[1,0].plot(ts,np.real(ys2),color="cornflowerblue")
axs[1,0].set_ylim((-1,1))
axs[1,0].set_title("Filtered signal (low pass)")

axs[2,1].plot(ws,np.abs(fs3), color="seagreen")
axs[2,1].set_title("Filtered transform (high pass)")

axs[2,0].plot(ts,np.real(ys3),color="cornflowerblue")
axs[2,0].set_ylim((-1,1))
axs[2,0].set_title("Filtered signal (high pass)")

plt.suptitle("Fourier transform and ideal low and high pass filters")

plt.tight_layout()
plt.show()