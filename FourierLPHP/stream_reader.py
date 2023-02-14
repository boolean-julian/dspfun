import pyaudio
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

mpl.rcParams['toolbar'] = 'None'
plt.rcParams.update({
	"font.family": "serif"
})

plt.style.use("dark_background")

pa = pyaudio.PyAudio()

channels			= 1
rate 				= 21000
input_device_index 	= 1
frames_per_buffer 	= 1024

stream = pa.open(
	format 				= pyaudio.paInt16,
	channels			= channels,
	rate 				= rate,
	input 				= True,
	input_device_index	= input_device_index,
	frames_per_buffer 	= frames_per_buffer
)

bgcolor = "#282828"
fig, ax = plt.subplots(2,1, facecolor=bgcolor)
fig.canvas.manager.full_screen_toggle() # toggle fullscreen mod
fig.canvas.toolbar_visible = False
plt.tight_layout()

fourier_plot, = ax[0].plot([], [], '-', color = "#91bfbf", linewidth=2)

maximum_plot, = ax[0].plot([], [], 'o', color = "#ede480", linewidth=2)
maximum_text  = ax[0].text(None, None, "")

signal_plot,  = ax[1].plot([], [], '-', color = "#f49c61", linewidth=2)

ax[0].set_facecolor(bgcolor)
ax[1].set_facecolor(bgcolor)

fftfreq = np.fft.rfftfreq(frames_per_buffer, 1/rate)
def init():
	fourier = np.zeros(frames_per_buffer)

	ax[0].set_title("Frequency domain")

	ax[0].set_xlim(np.min(fftfreq), np.max(fftfreq)//2)
	ax[0].set_ylim(0,100*np.iinfo(np.int16).max)

	fourier_plot.set_data(fftfreq,fourier)

	signal 	= np.zeros(frames_per_buffer)
	time 	= np.linspace(0,rate//frames_per_buffer,frames_per_buffer)

	ax[1].set_title("Time domain")

	ax[1].set_xlim(0, rate//frames_per_buffer)
	ax[1].set_ylim(np.iinfo(np.int16).min, np.iinfo(np.int16).max)

	signal_plot.set_data(time,signal)

	return fourier_plot, signal_plot, maximum_plot, maximum_text,

def animate(frame):
	signal = np.frombuffer(stream.read(frames_per_buffer), dtype=np.int16)

	fourier = np.abs(np.fft.rfft(signal))

	idx_max 	= np.argmax(fourier)
	fourier_max = fourier[idx_max]
	fftfreq_max = fftfreq[idx_max]
	
	peak_threshold = 5e3
	if fourier_max > peak_threshold:
		maximum_plot.set_data(fftfreq_max,fourier_max)
		maximum_text = ax[0].text(fftfreq_max+16,fourier_max+1000,str(int(fftfreq_max)))
	else:
		maximum_plot.set_data([], []) # out of bounds
		maximum_text = ax[0].text(None, None, "")


	fourier_plot.set_ydata(fourier)
	signal_plot.set_ydata(signal)

	return fourier_plot, signal_plot, maximum_plot, maximum_text,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1, interval=1, blit=True)
plt.show()

stream.stop_stream()
stream.close()
pa.terminate()