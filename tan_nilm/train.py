# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import NILMDataset  # remove the dot if running locally

CONFIG = {
    'dataset_path': './dataset/ukdale/ukdale_tan.h5',
    'seq_len': 256,
    'stride': 256,
    'houses': ['house_1'],
    'app_channels': [1, 2],   # predict these channels
    'batch_size': 16,
    'hidden_size': 128,
    'epochs': 5,
    'lr': 0.001,
}

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------
class TAN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(TAN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden)
        attn_scores = self.attention(lstm_out) # (batch, seq_len, 1)
        weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)  # (batch, hidden)
        output = self.fc(context)              # (batch, output_size)
        return output


# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------
def train():
    # ---------------------------
    # Dataset
    # ---------------------------
    dataset = NILMDataset(
        CONFIG['dataset_path'],
        seq_len=CONFIG['seq_len'],
        stride=CONFIG['stride'],
        houses=CONFIG['houses'],
        app_channels=CONFIG['app_channels']
    )
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    print("Number of elements", len(loader))

    # ---------------------------
    # Model
    # ---------------------------
    output_size = len(CONFIG['app_channels'])
    model = TAN(input_size=1, hidden_size=CONFIG['hidden_size'], output_size=output_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        i = 0
        # Training loop
        for x, y in loader:
            i+=1
            x, y = x.to(device), y.to(device)

            # x: (B, seq_len, 1)
            if x.dim() == 4:
                x = x.squeeze(-1)
                x = x.sum(dim=-1, keepdim=True)

            # target: last timestep, all appliance channels
            target = y[:, -1, :, 0]

            # normalize
            x = x / 1e8
            target = target / 1e8

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {epoch_loss/len(loader):.6f} - {i}")


if __name__ == "__main__":
    train()
