import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# Load your data
df = pd.read_csv('data/processed_data.csv', index_col=0)
prices = df.iloc[:, 0].values

# Figure 1: Time Series Plot
plt.figure(figsize=(10, 4))
plt.plot(prices, color='blue', label='Normalized Price')
plt.title('Time Series: TCS Stock Price (Normalized)')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.legend()
plt.savefig('results/time_series.png')
print("✅ Figure 1: Time Series saved.")

# Figure 2: Frequency Spectrum (FFT)
n = len(prices)
yf = fft(prices)
xf = fftfreq(n, 1)[:n//2] # Only positive frequencies

plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/n * np.abs(yf[0:n//2]), color='red')
plt.title('Frequency Spectrum: Global Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.savefig('results/frequency_spectrum.png')
print("✅ Figure 2: Frequency Spectrum saved.")