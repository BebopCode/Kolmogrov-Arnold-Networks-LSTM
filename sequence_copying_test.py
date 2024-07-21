import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from kan_lstm import KanLSTM

def generate_sequences(batch_size, seq_length, input_size):
    return torch.rand(batch_size, seq_length, input_size)

def train_and_evaluate(model, sequences, criterion, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, _ = model(sequences)
        loss = criterion(output, sequences)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return losses

def copy_test(model, sequences):
    with torch.no_grad():
        output, _ = model(sequences)
    return output
# Hyperparameters
input_size = 20
hidden_size = 20
kan_hidden_size = 5
num_layers = 5
batch_size = 32
seq_length = 20
num_epochs = 250
learning_rate = 0.001

# Generate sequences
sequences = generate_sequences(batch_size, seq_length, input_size)

# Initialize models
kan_lstm = KanLSTM(input_size, hidden_size, kan_hidden_size, num_layers)
standard_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# Loss function and optimizers
criterion = nn.MSELoss()
kan_optimizer = optim.Adam(kan_lstm.parameters(), lr=learning_rate)
std_optimizer = optim.Adam(standard_lstm.parameters(), lr=learning_rate)

# Train and evaluate
kan_losses = train_and_evaluate(kan_lstm, sequences, criterion, kan_optimizer, num_epochs)
std_losses = train_and_evaluate(standard_lstm, sequences, criterion, std_optimizer, num_epochs)

# Test sequence copying
kan_output = copy_test(kan_lstm, sequences)
std_output = copy_test(standard_lstm, sequences)

# Calculate mean squared errors
kan_mse = torch.mean((kan_output - sequences) ** 2).item()
std_mse = torch.mean((std_output - sequences) ** 2).item()

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(kan_losses, label='KanLSTM')
plt.plot(std_losses, label='Standard LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()
plt.savefig('Comparison LSTM.png')
# Print final MSE for both models
print(f"KanLSTM final MSE: {kan_mse:.6f}")
print(f"Standard LSTM final MSE: {std_mse:.6f}")