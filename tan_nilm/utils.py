import torch

def compute_mae(pred, target):
    """
    Compute Mean Absolute Error per appliance
    pred, target: [batch, num_appliances]
    """
    return torch.mean(torch.abs(pred - target), dim=0)

def energy_error(pred, target):
    """
    Compute relative energy error per appliance
    """
    energy_pred = pred.sum(dim=0)
    energy_true = target.sum(dim=0)
    return torch.abs(energy_pred - energy_true) / (energy_true + 1e-8)
