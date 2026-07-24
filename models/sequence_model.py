"""
LSTM autoencoder over sliding windows of an entity's event sequence.

Each categorical field is embedded, concatenated with the scaled numeric
features at every timestep, encoded by an LSTM down to a single latent
vector (the encoder's final hidden state), then a decoder LSTM re-expands
that latent vector back out across the window and a linear layer projects
it back into the embedded+numeric feature space. The model is trained to
minimize reconstruction MSE over the whole window; at scoring time we only
look at the reconstruction error of the *last* timestep of each window,
which is the error for the specific event that window "ends on" -- this is
what becomes each event's risk score.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .data_prep import NUMERIC_COLUMNS
from .vocab import CATEGORICAL_COLUMNS


def _sliding_windows_1d(arr, window):
    return np.lib.stride_tricks.sliding_window_view(arr, window)


def _sliding_windows_2d(arr, window):
    # arr: (n, f) -> (n - window + 1, window, f)
    w = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # (n-window+1, f, window)
    return np.ascontiguousarray(np.swapaxes(w, 1, 2))


def build_windows(df, cat_arrays, numeric_array, eligible_entity_ids, window_size=cfg.WINDOW_SIZE):
    """Returns cat_windows (dict[col] -> int64[N, T]), numeric_windows (float32[N, T, F]),
    and target_index (the original df.index each window's *last* row corresponds to)."""
    cat_window_chunks = {col: [] for col in CATEGORICAL_COLUMNS}
    numeric_chunks = []
    target_index_chunks = []

    positions_by_entity = df.groupby("entity_id", sort=False).indices  # entity_id -> positional index array (sorted order)

    for entity_id, positions in positions_by_entity.items():
        if entity_id not in eligible_entity_ids:
            continue
        n_local = len(positions)
        if n_local < window_size:
            continue

        for col in CATEGORICAL_COLUMNS:
            local_arr = cat_arrays[col][positions]
            cat_window_chunks[col].append(_sliding_windows_1d(local_arr, window_size))

        local_numeric = numeric_array[positions]
        numeric_chunks.append(_sliding_windows_2d(local_numeric, window_size))

        local_df_index = df.index.to_numpy()[positions]
        target_index_chunks.append(local_df_index[window_size - 1:])

    cat_windows = {col: np.concatenate(chunks, axis=0) for col, chunks in cat_window_chunks.items()}
    numeric_windows = np.concatenate(numeric_chunks, axis=0).astype(np.float32)
    target_index = np.concatenate(target_index_chunks, axis=0)
    return cat_windows, numeric_windows, target_index


class WindowDataset(Dataset):
    def __init__(self, cat_windows, numeric_windows):
        self.cat_windows = {col: torch.as_tensor(arr, dtype=torch.long) for col, arr in cat_windows.items()}
        self.numeric_windows = torch.as_tensor(numeric_windows, dtype=torch.float32)

    def __len__(self):
        return self.numeric_windows.shape[0]

    def __getitem__(self, idx):
        cats = {col: arr[idx] for col, arr in self.cat_windows.items()}
        return cats, self.numeric_windows[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocabs, embed_dims, numeric_dim, hidden_dim=cfg.HIDDEN_DIM,
                 num_layers=cfg.NUM_LSTM_LAYERS, dropout=cfg.DROPOUT):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocabs[col].size, embed_dims[col], padding_idx=0)
            for col in CATEGORICAL_COLUMNS
        })
        input_dim = sum(embed_dims.values()) + numeric_dim
        self.input_dim = input_dim
        dropout_arg = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_arg)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_arg)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def _build_input(self, cat_inputs, numeric):
        embeds = [self.embeddings[col](cat_inputs[col]) for col in CATEGORICAL_COLUMNS]
        return torch.cat(embeds + [numeric], dim=-1)

    def forward(self, cat_inputs, numeric):
        x = self._build_input(cat_inputs, numeric)
        _, (h, _) = self.encoder(x)
        latent = h[-1]  # final layer's final hidden state: [B, hidden_dim]
        T = x.size(1)
        latent_seq = latent.unsqueeze(1).expand(-1, T, -1)
        dec_out, _ = self.decoder(latent_seq)
        recon = self.output_proj(dec_out)
        return recon, x


def train_autoencoder(model, dataset, n_epochs=cfg.N_EPOCHS, batch_size=cfg.BATCH_SIZE,
                       lr=cfg.LEARNING_RATE, device="cpu", verbose=True):
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(n_epochs):
        total_loss, n_batches = 0.0, 0
        for cats, numeric in loader:
            cats = {col: t.to(device) for col, t in cats.items()}
            numeric = numeric.to(device)
            optimizer.zero_grad()
            recon, target = model(cats, numeric)
            loss = loss_fn(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        if verbose:
            print(f"  epoch {epoch + 1}/{n_epochs}  reconstruction MSE = {avg_loss:.4f}")
    return history


@torch.no_grad()
def last_step_reconstruction_errors(model, cat_windows, numeric_windows, device="cpu", batch_size=1024):
    model.to(device)
    model.eval()
    dataset = WindowDataset(cat_windows, numeric_windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    for cats, numeric in loader:
        cats = {col: t.to(device) for col, t in cats.items()}
        numeric = numeric.to(device)
        recon, target = model(cats, numeric)
        per_step_mse = ((recon - target) ** 2).mean(dim=-1)  # [B, T]
        errors.append(per_step_mse[:, -1].cpu().numpy())
    return np.concatenate(errors, axis=0)
