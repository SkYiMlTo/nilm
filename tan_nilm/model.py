import torch
import torch.nn as nn

class TAN(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1):
        super(TAN, self).__init__()

        print("DEBUG — Model initialized with input_size =", input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        output = self.fc(context)
        return output
