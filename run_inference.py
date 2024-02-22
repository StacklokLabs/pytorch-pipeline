import torch
import numpy as np
from train_model import SimpleNN  # Assuming SimpleNN is defined in model.py

# Load the saved model parameters
model = SimpleNN(input_dim=20)  # Assuming 20 features
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Define a function to make predictions
def predict(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy().squeeze()

# Example usage:
new_data = np.random.randn(10, 20)  # Example: 10 samples with 20 features each
predictions = predict(new_data)
print("Predictions:", predictions)
