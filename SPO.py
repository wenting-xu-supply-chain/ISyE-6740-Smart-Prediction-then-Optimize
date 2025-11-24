import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Path to your CSV
DATA_PATH = "rl_ready_sales_data.csv"

df = pd.read_csv(DATA_PATH)

# Focus on single SKU
STORE_ID   = "S001"
PRODUCT_ID = "P0001"

sku = df[(df["Store ID"] == STORE_ID) &
         (df["Product ID"] == PRODUCT_ID)].copy()

# Ensure time order
sku["Date"] = pd.to_datetime(sku["Date"])
sku = sku.sort_values("Date").reset_index(drop=True)

print(sku["Split"].value_counts())
print(sku["Date"].min(), "->", sku["Date"].max())

# Time features
sku["day_of_week"] = sku["Date"].dt.weekday
sku["month"]       = sku["Date"].dt.month

numeric_cols = [
    "Price",
    "Discount",
    "Competitor Pricing",
    "Inventory Level",
    "Units Sold",  # you can also use explicit lags of Demand here
    "Epidemic",
]
cat_cols = [
    "Weather Condition",
    "Seasonality",
    "Promotion",
    "day_of_week",
    "month",
]
feature_cols = numeric_cols + cat_cols

# Train/test split indices (contiguous in time)
sku_train = sku[sku["Split"] == "Train"].copy()
sku_test  = sku[sku["Split"] == "Test"].copy()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols),
    ]
)

# Fit only on Train to avoid leakage
Z_train = preprocess.fit_transform(sku_train[feature_cols])
Z_test  = preprocess.transform(sku_test[feature_cols])

# Convert to dense if sparse
from scipy import sparse
if sparse.issparse(Z_train):
    Z_train = Z_train.toarray()
    Z_test  = Z_test.toarray()

# Merge back into full time series order
Z_all = np.zeros((sku.shape[0], Z_train.shape[1]))
Z_all[:len(Z_train)] = Z_train
Z_all[len(Z_train):] = Z_test

print("Daily feature matrix shape:", Z_all.shape)

# Demand series
D_all = sku["Demand"].values.astype(float)

# Cost parameters for this SKU (constant over time)
c_p = float(sku["Unit_Cost"].iloc[0])          # purchase cost
h   = float(sku["Holding_Cost_Daily"].iloc[0]) # daily holding cost
K   = float(sku["Order_Cost_Fixed"].iloc[0])   # fixed order cost

print("c_p, h, K =", c_p, h, K)


# Planning horizon (days)
H = len(sku_test)  # e.g., monthly planning; can set H = len(sku_test) for full 3 months

T = H  # alias
p_day = Z_all.shape[1]

# Precompute arcs and index mapping
arcs = []  # list of (i, j) with 0 <= i < j <= T
arc_index = [[-1] * (T + 1) for _ in range(T)]  # arc_index[i][j] -> index in arcs
k = 0
for i in range(T):          # start period index (0..T-1)
    for j in range(i+1, T+1):  # end node index (i+1..T)
        arcs.append((i, j))
        arc_index[i][j] = k
        k += 1

M = k  # number of arcs
print("Horizon length T =", T)
print("Number of arcs M =", M, "(should be T*(T+1)/2 =", T*(T+1)//2, ")")

def build_arc_costs_for_horizon(d, arcs, c_p, h, K):
    """
    d: array-like of length H (demands for periods 0..H-1)
    arcs: list of (i, j) arc indices
    returns: c (M,) arc cost vector
    """
    d = np.asarray(d)
    c = np.zeros(len(arcs))
    for k, (i, j) in enumerate(arcs):
        seg = d[i:j]             # demands D[i..j-1]
        qty = seg.sum()
        t_idx = np.arange(i, j)  # i..j-1
        holding = (h * (t_idx - i) * seg).sum()
        c[k] = K + c_p * qty + holding
    return c

def shortest_path_arcs(c, arcs, T, arc_index):
    """
    c: (M,) cost per arc
    arcs: list of (i,j)
    T: horizon length
    arc_index: 2D mapping (i,j) -> arc index
    Returns:
        w: (M,) indicator vector for chosen arcs
        total_cost: scalar
    """
    INF = 1e18
    J = [INF] * (T + 1)  # J[t]: minimal cost to cover periods 0..t-1
    pred = [-1] * (T + 1)

    J[0] = 0.0
    pred[0] = -1

    for t in range(1, T + 1):
        best = INF
        best_i = -1
        for i in range(t):
            idx = arc_index[i][t]
            if idx < 0:
                continue
            val = J[i] + c[idx]
            if val < best:
                best = val
                best_i = i
        J[t] = best
        pred[t] = best_i

    # Backtrack to get chosen arcs
    w = np.zeros(len(arcs))
    t = T
    while t > 0:
        i = pred[t]
        idx = arc_index[i][t]
        w[idx] = 1.0
        t = i
    return w, J[T]

N = sku.shape[0]
train_end = len(sku_train)  # 668
test_start = train_end      # 668
test_end   = N              # 760

# Start indices for horizons fully in Train and Test
train_starts = list(range(0, train_end - H + 1))
test_starts  = list(range(test_start, test_end - H + 1))

print("Number of training horizons:", len(train_starts))
print("Number of test horizons:", len(test_starts))

def build_horizon_dataset(starts, Z_all, D_all, H, arcs, c_p, h, K):
    X_list = []
    C_list = []
    for s in starts:
        Z_seg = Z_all[s : s + H]   # (H, p_day)
        x = Z_seg.reshape(-1)      # flatten to 1D: H * p_day
        d_seg = D_all[s : s + H]   # (H,)
        c_seg = build_arc_costs_for_horizon(d_seg, arcs, c_p, h, K)
        X_list.append(x)
        C_list.append(c_seg)
    return np.vstack(X_list), np.vstack(C_list)

X_train_h, C_train_h = build_horizon_dataset(
    train_starts, Z_all, D_all, H, arcs, c_p, h, K
)
X_test_h, C_test_h = build_horizon_dataset(
    test_starts, Z_all, D_all, H, arcs, c_p, h, K
)

print("Horizon X_train shape:", X_train_h.shape)  # (n_train_horizons, H * p_day)
print("Horizon C_train shape:", C_train_h.shape)  # (n_train_horizons, M)
print("Horizon X_test shape:",  X_test_h.shape)
print("Horizon C_test shape:",  C_test_h.shape)


def spo_plus_loss_and_grad_lotsize(Xbatch, Cbatch, W, b, arcs, T, arc_index):
    """
    SPO+ loss + gradients for multi-period lot sizing (shortest path).
    Xbatch: (B, p) horizon features
    Cbatch: (B, M) true arc cost vectors
    W: (p, M), b: (M,) linear model parameters
    arcs, T, arc_index: lot-sizing graph structure
    """
    B, p = Xbatch.shape
    M = Cbatch.shape[1]

    Chat = Xbatch @ W + b  # (B, M) predicted costs
    loss_vec = np.zeros(B)
    g_chat = np.zeros((B, M))  # gradient wrt predicted costs for each sample

    for i in range(B):
        c_true = Cbatch[i]
        c_hat  = Chat[i]

        # w_true: optimal path under true costs
        w_true, _ = shortest_path_arcs(c_true, arcs, T, arc_index)

        # w_spo: optimal path under modified costs 2 c_hat - c_true
        c_prime = 2 * c_hat - c_true
        w_spo, _ = shortest_path_arcs(c_prime, arcs, T, arc_index)

        # SPO+ gradient wrt c_hat
        g_chat[i, :] = 2.0 * (w_true - w_spo)

        # SPO+ loss: ξ_S(c - 2 ĉ) + 2 ĉ^T w_true - c^T w_true
        d = c_true - 2 * c_hat
        xi = np.dot(d, w_spo)            # support term
        z_star = np.dot(c_true, w_true)  # min cost under true c
        chat_w_true = np.dot(c_hat, w_true)
        loss_vec[i] = xi + 2 * chat_w_true - z_star

    loss = loss_vec.mean()

    # Average gradient over batch
    g_chat /= B
    grad_W = Xbatch.T @ g_chat
    grad_b = g_chat.sum(axis=0)

    return loss, grad_W, grad_b


rng = np.random.default_rng(seed=0)

n_train, p = X_train_h.shape
M = C_train_h.shape[1]

# Initialize parameters
W = rng.normal(scale=0.01, size=(p, M))
b = np.zeros(M)

# Hyperparameters
num_epochs    = 50
learning_rate = 1e-7   # small because problem is large
batch_size    = 32

def iterate_minibatches(X, C, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        yield X[idx], C[idx]

for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0

    for Xb, Cb in iterate_minibatches(X_train_h, C_train_h, batch_size):
        loss, gW, gb = spo_plus_loss_and_grad_lotsize(
            Xb, Cb, W, b, arcs, T, arc_index
        )
        # Gradient step
        W -= learning_rate * gW
        b -= learning_rate * gb

        epoch_loss += loss
        batch_count += 1

    epoch_loss /= batch_count
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | SPO+ horizon loss ≈ {epoch_loss:.4f}")


def evaluate_avg_cost_gap(X_h, C_h, W, b, arcs, T, arc_index):
    n = X_h.shape[0]
    total_pred = 0.0
    total_true = 0.0

    for i in range(n):
        x = X_h[i]
        c_true = C_h[i]
        c_hat  = x @ W + b

        w_true, _ = shortest_path_arcs(c_true, arcs, T, arc_index)
        w_pred, _ = shortest_path_arcs(c_hat,  arcs, T, arc_index)

        cost_true = np.dot(c_true, w_true)
        cost_pred = np.dot(c_true, w_pred)

        total_true += cost_true
        total_pred += cost_pred

    avg_true = total_true / n
    avg_pred = total_pred / n
    avg_regret = avg_pred - avg_true
    return avg_pred, avg_true, avg_regret

avg_pred, avg_true, avg_regret = evaluate_avg_cost_gap(
    X_test_h, C_test_h, W, b, arcs, T, arc_index
)

print("Average realized cost (SPO+ decisions):", avg_pred)
print("Average realized cost (clairvoyant):   ", avg_true)
print("Average regret per horizon:            ", avg_regret)


