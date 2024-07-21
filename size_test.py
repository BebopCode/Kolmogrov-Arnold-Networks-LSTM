import torch
import torch.nn as nn
from kan_lstm import KanLSTMCell, KanLSTM

def test_custom_lstm():
    # Parameters
    input_size = 10
    hidden_size = 20
    kan_hidden_size = 5
    num_layers = 2
    seq_length = 5
    batch_size = 3

    # Create custom LSTM
    custom_lstm = KanLSTM(input_size, hidden_size, kan_hidden_size, num_layers)

    # Create input
    x = torch.randn(seq_length, batch_size, input_size)

    # Forward pass
    output, (hn, cn) = custom_lstm(x)

    # Check output shape
    assert output.shape == (seq_length, batch_size, hidden_size)
    print(f'Result shape {hn.shape},exptected_shape[{num_layers},{batch_size},{hidden_size}]')
    assert hn.shape == (num_layers, batch_size, hidden_size)
    assert cn.shape == (num_layers, batch_size, hidden_size)

    # Compare with PyTorch LSTM
    torch_lstm = nn.LSTM(input_size, hidden_size, num_layers)
    torch_output, (torch_hn, torch_cn) = torch_lstm(x)

    # Check if outputs are close
    assert torch.allclose(output, torch_output, atol=1e-4)
    assert torch.allclose(hn, torch_hn, atol=1e-4)
    assert torch.allclose(cn, torch_cn, atol=1e-4)

    print("All tests passed!")

test_custom_lstm()