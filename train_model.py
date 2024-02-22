import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple feedforward neural network using PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Load the dataset
dataset = np.load('dataset.npz')
X_train, y_train = dataset['X'], dataset['y']

# Reshape the target array to be a column vector
y_train = y_train.reshape(-1, 1)

# Initialize the model
model = SimpleNN(input_dim=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model parameters
torch.save(model.state_dict(), 'model.pth')
