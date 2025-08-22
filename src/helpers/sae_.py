import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_lambda=5.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = F.relu(self.encoder(x))  # ReLU like the paper
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = torch.mean(torch.abs(z))
        return recon_loss + self.sparsity_lambda * sparsity_loss


def l2_normalize_rows(x, eps=1e-8):
    norm = torch.norm(x, dim=1, keepdim=True) + eps
    return x / norm


def train_sae(
    model,
    data,
    epochs=100,
    lr=1e-3,
    batch_size=512,
    device=None,
    normalize_input=True,
    lr_cooldown=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    if not isinstance(data, torch.Tensor):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()

    if normalize_input:
        data_tensor = l2_normalize_rows(data_tensor)

    data_tensor = data_tensor.to(device)
    dataloader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.9999))  # like paper

    for epoch in range(epochs):
        total_loss = 0.0
        # Optional linear LR decay in last 20% of training
        if lr_cooldown and epoch > 0.8 * epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (1 - (epoch - 0.8 * epochs) / (0.2 * epochs))

        for batch in dataloader:
            batch = batch.to(device)
            if normalize_input:
                batch = l2_normalize_rows(batch)

            optimizer.zero_grad()
            x_hat, z = model(batch)
            loss = model.loss(batch, x_hat, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(data_tensor)
        print(f"[SAE] Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}")
