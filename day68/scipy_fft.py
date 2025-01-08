import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import fftpack

sns.set_style("dark")

fre = 100
fre_samp = 100

t = np.linspace(0, 2, 2 * fre_samp, endpoint=False) #0 to 2 seconds with a frequency of 100 Hz
a = np.sin(fre * 2 * np.pi * t)   #generates a sinusoidal signal

plt.plot(t, a)
plt.xlabel('time')
plt.ylabel('signal')
plt.show()

A = fftpack.fft(a)  # contains the complex Fourier coefficients.
frequency = fftpack.fftfreq(len(a)) * fre_samp #FFT coefficients, scaled by the sampling frequency 

plt.stem(frequency, np.abs(A), use_line_collection=True)
plt.xlabel('freq in Hz')
plt.ylabel('freq in Mag')
plt.show()
