# 📈 FinanceTimeSeriesForecast: Signal-Based CNN Modeling
> **Author:** Harikrishnan M  
> **University Reg No:** [TCR24CS032]  
> **Department:** Computer Science and Engineering  
> **Institution:** Government Engineering College, Thrissur

---

## 🚀 The Core Concept
This project transforms financial time series (TCS Stock Data) into non-stationary signals. By applying **Short-Time Fourier Transforms (STFT)**, we generate 2D Spectrograms to train a **Convolutional Neural Network (CNN)** for price trend prediction.

---

## 🛠️ Project Structure & Reproduction Commands

### 1.🛠️ Project Structure
```text
.
├── LICENSE
├── README.md                     # Project documentation
├── Requirements.txt               # Python dependencies
├── data/
│   ├── processed_data.csv        # Scaled financial data
│   └── spec_data.npy             # Pre-computed STFT spectrograms
├── results/
│   ├── model_weights.pth         # Trained CNN weights (.pth)
│   ├── report.pdf                # Compiled Academic Report
│   ├── report.tex                # LaTeX Source Code
│   ├── spectrogram.png           # Figure: Time-Frequency Plot
│   ├── spectrum.png              # Figure: FFT Magnitude Plot
│   └── timeseries.png            # Figure: Normalized Price Plot
└── src/
    ├── data_loader.py            # Financial data ingestion (yfinance)
    ├── generate_plots.py         # Signal visualization logic
    ├── model.py                  # CNN Architecture (PyTorch)
    ├── signal_processor.py       # STFT and FFT implementation
    └── train.py                  # Model training and evaluation
```
### 2. Repository Setup & Cleanup
Run these commands to prepare your environment and remove unnecessary clutter:

```bash
# Clone the repository
git clone [https://github.com/Caissaiamkaiser/FinanceTimeSeriesForecast-1.git](https://github.com/Caissaiamkaiser/FinanceTimeSeriesForecast-1.git)
cd FinanceTimeSeriesForecast-1

# Purge unnecessary folders and LaTeX 'shit'
rm -rf notebooks
rm -f results/*.aux results/*.log results/*.out results/*.fdb_latexmk

# Install dependencies
pip install -r Requirements.txt
```

#### 4. Project Dashboard

`https://ha7-piixel.github.io/FinanceTimeSeriesForecast/`
