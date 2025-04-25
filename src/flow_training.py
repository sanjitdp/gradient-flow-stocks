import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# set seeds for reproducibility

torch.manual_seed(0)
np.random.seed(0)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

T_BASE = 0.05  # amount of time to integrate ODE for, per day
N_STEPS = 4  # number of steps to take in RK-4

# neural network hyperparameters

BATCH = 256
LR = 1e-3
EPOCHS = 20
CLIP_N = 1.0

# download and preprocess data

d1 = np.load("dict_part1.npy", allow_pickle=True).item()
d2 = np.load("dict_part2.npy", allow_pickle=True).item()
d = {k: np.concatenate([d1[k], d2[k]]) for k in ("x_data", "si", "di")}
order = np.argsort(d["di"])
[d.__setitem__(k, d[k][order]) for k in d]

N = d["di"].shape[0]
cut = int(0.8 * N)
train, test = {k: v[:cut] for k, v in d.items()}, {k: v[cut:] for k, v in d.items()}

scaler = StandardScaler().fit(train["x_data"])


def std(x):
    """Standardize data using the fitted scaler."""
    return torch.tensor(scaler.transform(x), dtype=torch.float32)


X_train, X_test = std(train["x_data"]), std(test["x_data"])
si_train, si_test = map(torch.tensor, (train["si"], test["si"]))
di_train, di_test = map(torch.tensor, (train["di"], test["di"]))
num_stocks = int(max(train["si"].max(), test["si"].max()) + 1)


def make_pairs(si, di):
    """Make pairs of indices to train flow model."""
    pairs = []
    for s in torch.unique(si):
        idx = (si == s).nonzero(as_tuple=False).view(-1)
        if idx.numel() > 1:
            i, j = idx[:-1], idx[1:]
            gap = di[j] - di[i]
            pairs.extend(zip(i.tolist(), j.tolist(), gap.tolist()))
    return pairs


class NextDay(Dataset):
    """Dataset for next day prediction."""

    def __init__(self, X, si, di):
        self.X, self.si, self.pairs = X, si, make_pairs(si, di)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, g = self.pairs[k]
        return (self.X[i], self.X[j], self.si[i], torch.tensor(g, dtype=torch.float32))


# create dataloaders

train_dl = DataLoader(
    NextDay(X_train, si_train, di_train), BATCH, shuffle=True, pin_memory=True
)
test_dl = DataLoader(
    NextDay(X_test, si_test, di_test), BATCH, shuffle=False, pin_memory=True
)

enc = torch.load("model_beastly.pt", map_location=device)
enc.eval()
[p.requires_grad_(False) for p in enc.parameters()]


@torch.no_grad()
def embed_batch(x, sid):
    """Embed a batch of data using the previously trained encoder."""
    B, P, D = x.size(0), 200, 32
    v = enc.val_enc(x.view(B * P, 1)).view(B, P, D)
    p = enc.pos_embedding.weight.unsqueeze(0).expand(B, -1, -1)
    s = enc.stock_embedding(sid).unsqueeze(1).expand(-1, P, -1)
    return F.normalize(v + p + s, p=2, dim=-1)


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


def rk4_batch(X0, field, tau):
    """Runge-Kutta fourth order method for ODE integration."""
    h = 1.0 / N_STEPS
    X = X0
    for _ in range(N_STEPS):
        k1 = field(X) * tau
        k2 = field(F.normalize(X + 0.5 * h * k1, p=2, dim=-1)) * tau
        k3 = field(F.normalize(X + 0.5 * h * k2, p=2, dim=-1)) * tau
        k4 = field(F.normalize(X + h * k3, p=2, dim=-1)) * tau
        X = F.normalize(X + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4), p=2, dim=-1)
    return X


# set up model and optimizer

flow = AttentionFlow().to(device)
opt = torch.optim.AdamW(flow.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.5, patience=3, verbose=True
)


def cosine_set_loss(A, B):
    """Cosine loss for next day prediction."""
    return (1 - (A * B).sum(dim=-1)).mean()


def run_epoch(loader, train=True):
    """Run one epoch of training or evaluation."""
    flow.train(train)
    total = 0.0
    for x_t, x_tp1, sid, gap in tqdm(loader, leave=False):
        x_t, x_tp1, sid, gap = [t.to(device) for t in (x_t, x_tp1, sid, gap)]
        H0 = embed_batch(x_t, sid)
        H1 = embed_batch(x_tp1, sid)
        tau = (T_BASE * gap).view(-1, 1, 1)
        H_pred = rk4_batch(H0, flow, tau)
        loss = cosine_set_loss(H_pred, H1)
        if train:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), CLIP_N)
            opt.step()
        total += loss.item() * x_t.size(0)
    return total / len(loader.dataset)


# training loop

best_test_loss = 1e9
for epoch in range(200):
    tr = run_epoch(train_dl, True)
    te = run_epoch(test_dl, False)
    sched.step(te)
    print(f"epoch {epoch:2d} | train {tr:.4f} | test {te:.4f}")

    if te < best_test_loss:
        best_test_loss = te
        torch.save(flow, "best_flow.pt")
        print(f"saved best model at epoch {epoch} with test loss {te:.4f}")
    torch.save(flow, "flow_epoch{epoch}.pt")
