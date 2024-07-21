from kan_lstm import KanLSTM
import torch
# Example usage
input_size = 10
hidden_size = 20
kan_hidden_size = 5
num_layers = 2
seq_len = 5
batch_size = 3

model_kan = KanLSTM(input_size, hidden_size,kan_hidden_size, num_layers)

def generate_sequence(seq_length, input_size):
    return torch.randint(0, input_size, (seq_length, 1))


def train_copy_task(model, seq_length, input_size, n_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        seq = generate_sequence(seq_length, input_size)
        inputs = nn.functional.one_hot(seq, input_size).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, input_size), seq.view(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Test the model
def test_copy_task(model, seq_length, input_size):
    model.eval()
    with torch.no_grad():
        seq = generate_sequence(seq_length, input_size)
        inputs = nn.functional.one_hot(seq, input_size).float()
        outputs = model(inputs)
        predicted = outputs.argmax(dim=-1)
        correct = (predicted == seq).float().mean()
        print(f'Test Accuracy: {correct.item():.4f}')