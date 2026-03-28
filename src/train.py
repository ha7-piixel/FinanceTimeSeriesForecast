import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from model import FinanceCNN  # Importing the class we wrote in Step 3

def train_model():
    # 1. Load the Spectrogram and the actual Prices
    spec_data = np.load('data/spec_data.npy') # Shape: [Freq, Time]
    prices = pd.read_csv('data/processed_data.csv').iloc[:, 1].values # TCS prices
    
    # 2. Prepare Tensors
    # We need to reshape the spectrogram to [Batch, Channel, Height, Width]
    # For simplicity, we'll treat the whole spectrogram as one "image" to predict 
    # the most recent price.
    X = torch.tensor(spec_data).float().unsqueeze(0).unsqueeze(0) 
    y = torch.tensor([prices[-1]]).float() # Target: The latest price
    
    # 3. Initialize Model, Loss, and Optimizer
    h, w = spec_data.shape
    model = FinanceCNN(h, w)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop (Short and sweet)
    print("Starting training...")
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.6f}")

    # 5. Save the results
    torch.save(model.state_dict(), 'results/model_weights.pth')
    print(" Task 4: Training complete. Model saved to results/model_weights.pth")

if __name__ == "__main__":
    train_model()