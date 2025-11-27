import torch
from torch.utils.data import DataLoader
from tan_nilm.model import TAN
from tan_nilm.dataset import NILMDataset
from tan_nilm.config import CONFIG

# Load dataset
dataset = NILMDataset(CONFIG['dataset_path'],
                      houses=CONFIG['houses'],
                      appliances=CONFIG['appliances'],
                      sequence_length=CONFIG['sequence_length'])
test_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TAN(input_size=1, hidden_size=128, output_size=1)
model.load_state_dict(torch.load(CONFIG['model_save_path']))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        print("Predicted:", output[0].cpu().numpy())
        print("Actual:", y[0].cpu().numpy())
        break
