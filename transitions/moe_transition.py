# moe_transition.py

import os, json, random
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Deep Mixture-of-Experts: K independent MLPs
# ---------------------------

class TransitionMLP(nn.Module):
    def __init__(self, dim, hidden=512, depth=3):
        super().__init__()
        
        layers = []
        in_dim = dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden

        # predict next embedding in same space
        layers.append(nn.Linear(in_dim, dim))  
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def _make_loader(X, Y, batch_size=256, shuffle=True):
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_experts(X, Y,
                  num_experts=5,
                  hidden=512,
                  depth=3,
                  epochs=5,
                  batch_size=256,
                  lr=1e-3,
                  save_dir="experts_lmsys",
                  seed=0):
    """
    Trains K independent experts on identical data (different inits).
    Saves each to save_dir/expert_{k}.pt, plus a small metadata.json.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = X.shape[1]
    loader = _make_loader(X, Y, batch_size=batch_size, shuffle=True)

    losses = []
    for k in range(num_experts):
        torch.manual_seed(seed + k)
        model = TransitionMLP(dim, hidden=hidden, depth=depth).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()

        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        losses.append(float(loss.detach().cpu()))

        torch.save(model.state_dict(), os.path.join(save_dir, f"expert_{k}.pt"))

    meta = {
        "dim": int(dim),
        "num_experts": int(num_experts),
        "hidden": int(hidden),
        "depth": int(depth),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "final_batch_loss_per_expert": losses,
        "device_used": device,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------
# (Optional) tiny inference helper for later use
# ---------------------------

def load_experts(save_dir):
    with open(os.path.join(save_dir, "metadata.json")) as f:
        meta = json.load(f)
    dim = meta["dim"]
    models = []
    for k in range(meta["num_experts"]):
        m = TransitionMLP(dim, hidden=meta["hidden"], depth=meta["depth"])
        m.load_state_dict(torch.load(os.path.join(save_dir, f"expert_{k}.pt"), map_location="cpu"))
        m.eval()
        models.append(m)
    return models, meta

def random_moe_predict(models, z_t):
    i = random.randrange(len(models))
    with torch.no_grad():
        return models[i](torch.as_tensor(z_t, dtype=torch.float32)).numpy()
