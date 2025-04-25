import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# download and preprocess data

np.random.seed(0)
torch.manual_seed(0)

dict_1 = np.load("dict_part1.npy", allow_pickle=True).item()
dict_2 = np.load("dict_part2.npy", allow_pickle=True).item()
d = {
    "x": np.concatenate([dict_1["x_data"], dict_2["x_data"]]),
    "si": np.concatenate([dict_1["si"], dict_2["si"]]),
    "di": np.concatenate([dict_1["di"], dict_2["di"]]),
}
order = np.argsort(d["di"])
for k in d:
    d[k] = d[k][order]

N = d["di"].shape[0]
cut = int(0.8 * N)
train, test = {k: v[:cut] for k, v in d.items()}, {k: v[cut:] for k, v in d.items()}

scaler = StandardScaler().fit(train["x"])
X_train = torch.tensor(scaler.transform(train["x"]), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(test["x"]), dtype=torch.float32)

si_train = torch.tensor(train["si"], dtype=torch.long)
si_test = torch.tensor(test["si"], dtype=torch.long)
di_train = torch.tensor(train["di"], dtype=torch.long)
di_test = torch.tensor(test["di"], dtype=torch.long)


def make_pairs(si, di):
    """Make pairs of indices to train embedding model."""

    pairs = []
    uniq = torch.unique(si)
    for s in uniq:
        idx = (si == s).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        i, j = idx[:-1], idx[1:]
        gap = di[j] - di[i]
        pairs.extend(zip(i.tolist(), j.tolist(), gap.tolist()))
    return pairs


class NextDayDataset(Dataset):
    """Dataset for next day prediction."""

    def __init__(self, X, si, di):
        self.pairs = make_pairs(si, di)
        self.X, self.si = X, si
        self.gap = torch.tensor([g for *_, g in self.pairs], dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, g = self.pairs[k]
        return self.X[i], self.X[j], self.si[i], self.gap[k]


# set up data loaders and move to GPU

train_ds = NextDayDataset(X_train, si_train, di_train)
test_ds = NextDayDataset(X_test, si_test, di_test)

train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, pin_memory=True)

num_stocks = int(max(train["si"].max(), test["si"].max()) + 1)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


class EmbeddingPredictor(nn.Module):
    """Embedding predictor for next day prediction. Also contains embedding model itself."""

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


# set up model, optimizer, and loss function

model = EmbeddingPredictor().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.5, patience=3, verbose=True
)
crit = nn.MSELoss()
MAX_NORM = 1.0


def run(dl, train=True):
    """Run one epoch of training or evaluation."""
    model.train(train)
    total = 0
    with torch.set_grad_enabled(train):
        for x_t, x_tp1, sid, gap in tqdm(dl, leave=False):
            x_t, x_tp1 = x_t.to(device), x_tp1.to(device)
            sid, gap = sid.to(device), gap.to(device)
            delta_true = x_tp1 - x_t
            delta_pred = model(x_t, sid, gap)
            loss = crit(delta_pred, delta_true)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                opt.step()
            total += loss.item() * x_t.size(0)
    return total / len(dl.dataset)


# training loop

for epoch in range(30):
    tr = run(train_dl, True)
    te = run(test_dl, False)
    sched.step(te)
    print(f"epoch {epoch:2d} | train delta MSE {tr:.4f} | test delta MSE {te:.4f}")

# save the model

torch.save(model.state_dict(), "model_beastly.pt")
torch.save(model, "model_beastly.pt")
