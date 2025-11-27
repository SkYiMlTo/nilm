import torch
from utils import compute_mae, energy_error

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        target = Y[:, -1, :]
        mae = compute_mae(pred, target)
        energy_err = energy_error(pred, target)
    print("Evaluation - MAE:", mae.tolist(), "Energy Error:", energy_err.tolist())
