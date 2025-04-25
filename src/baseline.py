import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from scipy.stats import t as student_t
from math import sqrt
import random
from sklearn.preprocessing import StandardScaler

# set seeds for reproducibility

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRAIN_SUB = 100_000  # number of training samples to use
N_TEST_SUB = 10_000  # number of test samples to use
BATCH = 1024  # batch size for training
LEARNING_RATE = 0.001  # learning rate for optimizer
EPOCHS = 50  # number of epochs to train for
DROPOUT_RATE = 0.3  # dropout rate for neural network


class EmbeddingNN(nn.Module):
    """Neural network for predicting next-day returns naively from embeddings."""

    def __init__(self, input_dim=32):
        super(EmbeddingNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(DROPOUT_RATE)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x.squeeze()


@torch.no_grad()
def build_embedding_features(indices, d, embed, BATCH):
    """Build features for training the neural network."""

    X_embed, y = [], []

    for batch_start in tqdm(range(0, len(indices), BATCH)):
        batch_indices = indices[batch_start : batch_start + BATCH]

        x_batch = torch.tensor(x_std(batch_indices), dtype=torch.float32, device=device)
        sid = torch.tensor(d["si"][batch_indices], dtype=torch.long, device=device)

        embeddings = embed_batch(x_batch, sid)

        pooled_embeddings = embeddings.mean(dim=1)

        X_embed.append(pooled_embeddings.cpu().numpy())
        y.extend(d["y_data"][batch_indices])

    return np.vstack(X_embed), np.array(y)


def train_embedding_model():
    """Train a neural network to predict next-day returns from embeddings."""

    # load and preprocess data

    print("Loading data...")

    d1 = np.load("dict_part1.npy", allow_pickle=True).item()
    d2 = np.load("dict_part2.npy", allow_pickle=True).item()
    d = {k: np.concatenate([d1[k], d2[k]]) for k in ("x_data", "y_data", "si", "di")}
    order = np.argsort(d["di"])
    [d.__setitem__(k, d[k][order]) for k in d]

    N = d["di"].shape[0]
    train_cut = int(0.9 * N)

    train_indices = np.arange(train_cut)
    test_indices = np.arange(train_cut, N)

    if len(train_indices) > N_TRAIN_SUB:
        train_indices = np.random.choice(train_indices, size=N_TRAIN_SUB, replace=False)
        train_indices.sort()

    if len(test_indices) > N_TEST_SUB:
        test_indices = np.random.choice(test_indices, size=N_TEST_SUB, replace=False)
        test_indices.sort()

    print(
        f"Using {len(train_indices):,} training samples | {len(test_indices):,} test samples"
    )

    global x_scaler, x_std
    x_scaler = StandardScaler().fit(d["x_data"][train_indices])

    def x_std(idx):
        """Standardize data using the fitted scaler."""
        return x_scaler.transform(d["x_data"][idx])

    # load previously trained embedding model

    print("Loading pretrained embedding model...")
    embed = torch.load("model_beastly.pt", map_location=device)
    embed.eval()
    [p.requires_grad_(False) for p in embed.parameters()]

    global embed_batch

    @torch.no_grad()
    def embed_batch(x, sid):
        """Embed a batch of data using the pretrained embedding model."""
        B, P, D = x.shape[0], 200, 32
        v = embed.val_enc(x.view(B * P, 1)).view(B, P, D)
        p = embed.pos_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        s = embed.stock_embedding(sid).unsqueeze(1).expand(-1, P, -1)
        return F.normalize(v + p + s, p=2, dim=-1)

    # build training and test features using embedding model

    print("Building training features...")
    X_train, y_train = build_embedding_features(train_indices, d, embed, BATCH)
    print("Building test features...")
    X_test, y_test = build_embedding_features(test_indices, d, embed, BATCH)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    # set up model, loss function, and optimizer

    model = EmbeddingNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop

    print("Training neural network...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

                test_outputs = model(X_test_tensor).cpu().numpy()
                test_mse = np.mean((test_outputs - y_test) ** 2)

                r = np.corrcoef(test_outputs, y_test)[0, 1]
                n = len(y_test)
                t_stat = r * sqrt((n - 2) / (1 - r**2))
                p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=n - 2))

                print(
                    f"Test MSE: {test_mse:.4f}, Pearson r: {r:.4f} (p = {p_value:.3e})"
                )
            model.train()

    # evaluate the model on the test set

    print("Final evaluation...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_preds = model(X_test_tensor).cpu().numpy()

        mse = np.mean((test_preds - y_test) ** 2)
        r = np.corrcoef(test_preds, y_test)[0, 1]
        n = len(y_test)
        t_stat = r * sqrt((n - 2) / (1 - r**2))
        p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=n - 2))

        print(
            f"\nNeural Network | Test MSE {mse:.4f} | Pearson r {r:.4f} (p = {p_value:.3e})"
        )

        # plot predicted vs. true next-day returns

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
        plt.savefig("baseline.pdf", bbox_inches="tight", dpi=300)
        plt.show()

    return model, test_preds


# train the MLP which naively uses embeddings to predict next-day returns (without flow)

model, predictions = train_embedding_model()
torch.save(
    model.state_dict(), "embedding_baseline_model.pt"
)  # save the resulting model
