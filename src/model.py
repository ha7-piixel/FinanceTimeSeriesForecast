import torch
import torch.nn as nn
import torch.nn.functional as F

class FinanceCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(FinanceCNN, self).__init__()
        # Layer 1: Look for patterns in the spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Layer 2: Refine those patterns
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate the size after two 2x2 pools
        self.final_height = input_height // 4
        self.final_width = input_width // 4
        
        # Fully connected layers for Regression (predicting a single number)
        self.fc1 = nn.Linear(32 * self.final_height * self.final_width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output is the predicted price
        return x

print(" Task 3: CNN Model Architecture defined.")