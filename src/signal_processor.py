import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os  # Added this

def generate_spectrogram_mvp():
    # Load the data we prepared in Task 1
    df = pd.read_csv('data/processed_data.csv', index_col=0)
    # We'll use the main stock price for the STFT signal
    price_signal = df.iloc[:, 0].values 
    
    # 3.2.3 Sliding Window Mechanism
    L = 32 
    H = 8
    
    # 3.2.4 Computation of Spectrogram
    f, t, Zxx = signal.stft(price_signal, nperseg=L, noverlap=L-H)
    spectrogram = np.abs(Zxx)**2 
    
    # --- SAVE VISUAL ---
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, spectrogram, shading='gouraud')
    plt.title('Spectrogram S(t, f)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [Days]')
    plt.savefig('results/spectrogram_task2.png')
    
    # --- SAVE RAW DATA FOR TRAIN.PY ---
    os.makedirs('data', exist_ok=True)
    np.save('data/spec_data.npy', spectrogram)
    
    print("Task 2 Complete. Visual and .npy data saved.")
    return spectrogram

if __name__ == "__main__":
    generate_spectrogram_mvp()