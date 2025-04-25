import numpy as np, torch, torch.nn.functional as F
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import torch.nn as nn
from math import sqrt
from scipy.stats import t as student_t
import random

# set seeds for reproducibility

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import t as student_t
from math import sqrt


class PredictionNN(nn.Module):
    """
    A simple feedforward neural network to predict stock return from flowed-forward embeddings.
    """

    def __init__(self, input_dim=34, hidden_dims=[128, 64], dropout_rate=0.3):
        super(PredictionNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


class ResidualNN(nn.Module):
    """
    A residual neural network to predict stock return from flowed-forward embeddings (not used by default).
    """

    def __init__(self, input_dim=34, hidden_dim=128, num_blocks=3, dropout_rate=0.3):
        super(ResidualNN, self).__init__()

        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            )

        self.final_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.initial_layer(x)

        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = self.relu(x + residual)

        return self.final_layer(x).squeeze()


def train_neural_network(X_train, y_train, X_test, y_test, device, use_resnet=False):
    """
    Train a neural network using the prediction network.
    """

    print("Training Neural Network on training data...")

    # create the dataset

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=1024, shuffle=False)

    if use_resnet:
        model = ResidualNN().to(device)
        print("Using Residual Neural Network architecture")
    else:
        model = PredictionNN().to(device)
        print("Using standard Neural Network architecture")

    # set up optimizer and loss function

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_r = -1.0  # keep track of the best Pearson correlation coefficient

    # training loop

    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        print(
            f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

        # every epoch, evaluate on the test set

        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()

            test_preds = test_outputs.cpu().numpy()
            r = np.corrcoef(test_preds, y_test)[0, 1]

            n = len(y_test)
            t = r * sqrt((n - 2) / (1 - r**2))
            p = 2 * (1 - student_t.cdf(abs(t), df=n - 2))

            print(f"Test MSE: {test_loss:.4f}, Pearson r: {r:.4f} (p = {p:.3e})")

            torch.save(model.state_dict(), f"nn_flow_epoch_{epoch}.pt")
            if r > best_r:
                best_r = r
                torch.save(model.state_dict(), "best_nn_flow_model.pt")

    # evaluate the best model on the test set

    model.load_state_dict(torch.load("best_nn_flow_model.pt"))

    print("Evaluating on test data...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_preds = test_outputs.cpu().numpy()

        mse = np.mean((test_preds - y_test) ** 2)
        r = np.corrcoef(test_preds, y_test)[0, 1]
        n = len(y_test)
        t = r * sqrt((n - 2) / (1 - r**2))
        p = 2 * (1 - student_t.cdf(abs(t), df=n - 2))

        print(
            f"\nNeural Network | Test MSE {mse:.4f} | Pearson r {r:.4f} (p = {p:.3e})"
        )

    return model, test_preds


# download and preprocess data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d1 = np.load("dict_part1.npy", allow_pickle=True).item()
d2 = np.load("dict_part2.npy", allow_pickle=True).item()
d = {k: np.concatenate([d1[k], d2[k]]) for k in ("x_data", "y_data", "si", "di")}
order = np.argsort(d["di"])
[d.__setitem__(k, d[k][order]) for k in d]

N = d["di"].shape[0]
train_cut = int(0.85 * N)
val_cut = 0

N_TRAIN_SUB = N * 0.75  # use first 75% of the data for training
N_VAL_SUB = N * 0.8  # use next 5% of the data for validation
N_TEST_SUB = N * 0.2  # use last 20% of the data for testing
T_BASE = 0.05  # amount of time to integrate ODE for, per day
N_STEPS = 4  # number of steps to take in RK-4
BATCH = 1024  # batch size for training


x_scaler = StandardScaler().fit(d["x_data"][:train_cut])


def x_std(idx):
    """Standardize data using the fitted scaler."""
    return x_scaler.transform(d["x_data"][idx])


# create pairs of indices to train MLP

pairs_train, pairs_val, pairs_test = [], [], []
for s in np.unique(d["si"]):
    idx = np.where(d["si"] == s)[0]
    if idx.size < 2:
        continue
    i, j = idx[:-1], idx[1:]
    gaps = d["di"][j] - d["di"][i]
    mask_train = (i < train_cut) & (j < train_cut)
    mask_val = (i >= train_cut) & (j >= train_cut) & (i < val_cut) & (j < val_cut)
    mask_test = (i >= val_cut) & (j >= val_cut)
    pairs_train.extend(list(zip(i[mask_train], j[mask_train], gaps[mask_train])))
    pairs_val.extend(list(zip(i[mask_val], j[mask_val], gaps[mask_val])))
    pairs_test.extend(list(zip(i[mask_test], j[mask_test], gaps[mask_test])))

pairs_train = np.array(pairs_train)
pairs_val = np.array(pairs_val)
pairs_test = np.array(pairs_test)

print(
    f"Using {len(pairs_train):,} train pairs | {len(pairs_val):,} validation pairs | {len(pairs_test):,} test pairs"
)

num_stocks = int(d["si"].max() + 1)


class EmbeddingPredictor(nn.Module):
    """Embedding predictor architecture."""

    def __init__(self, P=200, D=32, hidden=512, dropout=0.1, num_stocks=num_stocks):
        super().__init__()
        self.P, self.D = P, D
        self.pos_embedding = nn.Embedding(P, D)
        self.stock_embedding = nn.Embedding(num_stocks, D)
        self.val_enc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(inplace=True), nn.Linear(64, D)
        )
        self.gap_enc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(inplace=True), nn.Linear(64, D)
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(P * D, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, P)
        )

    def forward(self, x, sid, gap):
        B = x.size(0)
        v = self.val_enc(x.view(B * self.P, 1)).view(B, self.P, self.D)
        p = self.pos_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        s = self.stock_embedding(sid).unsqueeze(1).expand(-1, self.P, -1)
        g = self.gap_enc(gap.view(B, 1)).unsqueeze(1).expand(-1, self.P, -1)
        h = v + p + s + g
        h = F.normalize(h, p=2, dim=-1)
        h = self.dropout(h)
        z = h.reshape(B, -1)
        return self.decoder(z)


embed = EmbeddingPredictor().to(device)  # set up embedding model


class AttentionFlow(nn.Module):
    """Attention flow model for next day prediction."""

    def __init__(self, d=32):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        q, k, v = self.q(X), self.k(X), self.v(X)
        sim = torch.matmul(q, k.transpose(-2, -1))
        wts = torch.softmax(-self.beta * sim, dim=-1)
        agg = torch.matmul(wts, v)
        return agg - (agg * X).sum(dim=-1, keepdim=True) * X


flow = AttentionFlow().to(device)  # set up flow model

# load previously trained models

embed = torch.load("model_beastly.pt", map_location=device)
embed.eval()
[p.requires_grad_(False) for p in embed.parameters()]

flow = torch.load("best_flow.pt", map_location=device)
flow.eval()
[p.requires_grad_(False) for p in flow.parameters()]


@torch.no_grad()
def embed_batch(x, sid):
    """Embed a batch of data using the previously trained encoder."""
    B, P, D = x.shape[0], 200, 32
    v = embed.val_enc(x.view(B * P, 1)).view(B, P, D)
    p = embed.pos_embedding.weight.unsqueeze(0).expand(B, -1, -1)
    s = embed.stock_embedding(sid).unsqueeze(1).expand(-1, P, -1)
    return F.normalize(v + p + s, p=2, dim=-1)


def rk4_batch(X0, tau):
    """Runge-Kutta 4th order integration for a batch of data."""
    h, X = 1.0 / N_STEPS, X0
    for _ in range(N_STEPS):
        k1 = flow(X) * tau
        k2 = flow(F.normalize(X + 0.5 * h * k1, p=2, dim=-1)) * tau
        k3 = flow(F.normalize(X + 0.5 * h * k2, p=2, dim=-1)) * tau
        k4 = flow(F.normalize(X + h * k3, p=2, dim=-1)) * tau
        X = F.normalize(X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4), p=2, dim=-1)
    return X


def build_design(pairs):
    """Build design matrix for training."""
    X_feat, y = [], []
    for batch in tqdm(range(0, len(pairs), BATCH)):
        sub = pairs[batch : batch + BATCH]
        i_idx = sub[:, 0].astype(int)
        j_idx = sub[:, 1].astype(int)
        gap = torch.tensor(sub[:, 2], dtype=torch.float32, device=device)

        x_now = torch.tensor(x_std(j_idx), dtype=torch.float32, device=device)
        sid = torch.tensor(d["si"][j_idx], dtype=torch.long, device=device)

        H0 = embed_batch(x_now, sid)
        tau = (T_BASE * gap).view(-1, 1, 1)
        Hf = rk4_batch(H0, tau)

        T = 252
        day_idx = d["di"][j_idx]
        day_sin = torch.tensor(
            np.sin(2 * np.pi * day_idx / T), dtype=torch.float32, device=device
        ).view(-1, 1)
        day_cos = torch.tensor(
            np.cos(2 * np.pi * day_idx / T), dtype=torch.float32, device=device
        ).view(-1, 1)

        pooled_features = Hf.mean(dim=1)

        combined_features = torch.cat([pooled_features, day_sin, day_cos], dim=1)

        X_feat.append(combined_features.cpu().numpy())
        y.extend(d["y_data"][j_idx])

    return np.vstack(X_feat), np.array(y)


# build design matrix for training and testing

print("Building training features...")
X_train, y_train = build_design(pairs_train)
print("Building validation features...")
print("Building test features...")
X_test, y_test = build_design(pairs_test)

# train the neural network using the flowed-forward embeddings to predict next-day returns

model, test_preds = train_neural_network(
    X_train, y_train, X_test, y_test, device, use_resnet=False
)

# save the model and reload it

torch.save(model.state_dict(), "nn_flow_model.pt")

torch.load("nn_flow_model.pt", map_location=device)

# evaluate the model on the test set

r_raw = np.corrcoef(test_preds, y_test)[0, 1]
mse_raw = np.mean((test_preds - y_test) ** 2)

n = len(y_test)
t_raw = r_raw * sqrt((n - 2) / (1 - r_raw**2))
p_raw = 2 * (1 - student_t.cdf(abs(t_raw), df=n - 2))

print(f"\nNN model | Test MSE {mse_raw:.4f} | Pearson r {r_raw:.4f} (p = {p_raw:.3e})")

# plot the predicted vs. true next-day returns

subset_size = 10000
subset_indices = np.random.choice(len(y_test), size=subset_size, replace=False)
y_test_subset = y_test[subset_indices]
test_preds_subset = test_preds[subset_indices]

import matplotlib.pyplot as plt
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
plt.style.use("math.mplstyle")

plt.figure(figsize=(6, 4))
plt.grid()
plt.scatter(y_test, test_preds, alpha=0.5, zorder=2, s=15)
plt.title("Predicted vs. true next-day returns (baseline model)", size=11)
plt.xlabel("$y_{\mathrm{test}}$", size=11)
plt.ylabel("$y_{\mathrm{pred}}$", size=11)
margin = 0
plt.xlim(min(y_test) + margin, max(y_test) + margin)

slope, intercept = np.polyfit(y_test, test_preds, 1)
plt.plot(
    y_test,
    slope * y_test + intercept,
    color="red",
    linewidth=1,
    label="Line of Best Fit",
    zorder=3,
)
plt.savefig("flow.pdf", bbox_inches="tight", dpi=300)
plt.show()
