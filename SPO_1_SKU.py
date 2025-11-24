import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. Load and Filter Data
# ==========================================
print("Loading data...")
# Make sure this file matches your upload
df = pd.read_csv('rl_ready_sales_data.csv') 

# Filter for specific SKU
store_id = 'S001'
prod_id = 'P0001'
data = df[(df['Store ID'] == store_id) & (df['Product ID'] == prod_id)].copy()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)

print(f"Processing SKU: {store_id} - {prod_id}")
print(f"Train samples: {sum(data['Split']=='Train')}, Test samples: {sum(data['Split']=='Test')}")

# ==========================================
# 2. Feature Engineering
# ==========================================
# Add basic time features
data['day_of_week'] = data['Date'].dt.weekday
data['month'] = data['Date'].dt.month

numeric_features = ['Price', 'Discount', 'Competitor Pricing', 'Inventory Level', 'Units Sold', 'Epidemic']
categorical_features = ['Weather Condition', 'Seasonality', 'Promotion', 'day_of_week', 'month']

# Split Train/Test
train_mask = data['Split'] == 'Train'
test_mask = data['Split'] == 'Test'
train_data = data[train_mask].copy()
test_data = data[test_mask].copy()

# Normalize/Encode
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Fit on Train, Transform both
X_train_raw = preprocessor.fit_transform(train_data)
X_test_raw = preprocessor.transform(test_data)

# Get Demand and Cost Parameters
D_train = train_data['Demand'].values
D_test = test_data['Demand'].values

# We assume these are constant for the SKU as per your description
c_p = data['Unit_Cost'].iloc[0]
h = data['Holding_Cost_Daily'].iloc[0]
K = data['Order_Cost_Fixed'].iloc[0]

# ==========================================
# 3. SPO+ Model Definition (Shortest Path)
# ==========================================
H = 30   # Planning Horizon
T = H    # Number of time steps in horizon

# Generate arcs for a full connected DAG over H nodes
# Arc (i, j) means "Order at day i to cover demand until day j"
arcs = []
for i in range(T):
    for j in range(i + 1, T + 1):
        arcs.append((i, j))
M = len(arcs)

def build_arc_costs(d, arcs, c_p, h, K):
    """Calculate TRUE costs for arcs based on realized demand d."""
    c = np.zeros(len(arcs))
    d = np.array(d)
    for idx, (i, j) in enumerate(arcs):
        segment = d[i:j]
        qty = segment.sum()
        # Holding cost calculation: sum(h * (time_held) * units)
        # Units arriving at i, used at t (where i <= t < j) are held for (t-i) days
        times = np.arange(i, j)
        holding = np.sum(h * (times - i) * segment)
        c[idx] = K + c_p * qty + holding
    return c

def solve_shortest_path(c, T, arcs):
    """Dynamic Programming to find shortest path (optimal plan) given costs c."""
    V = np.full(T + 1, np.inf)
    parent = np.full(T + 1, -1)
    path_arc_idx = np.full(T + 1, -1)
    V[0] = 0
    
    # Forward pass (DP)
    for i in range(T):
        if V[i] == np.inf: continue
        for idx, (u, v) in enumerate(arcs):
            if u == i:
                if V[i] + c[idx] < V[v]:
                    V[v] = V[i] + c[idx]
                    parent[v] = u
                    path_arc_idx[v] = idx
    
    # Backward pass (Reconstruct Path)
    w = np.zeros(len(arcs))
    curr = T
    while curr > 0:
        idx = path_arc_idx[curr]
        if idx != -1:
            w[idx] = 1
        curr = parent[curr]
    return w

def get_spo_grad(c_hat, c_true, T, arcs):
    """Compute SPO+ subgradient."""
    w_true = solve_shortest_path(c_true, T, arcs)
    # SPO+ requires solving for cost (2*c_hat - c_true)
    c_spo = 2 * c_hat - c_true
    w_spo = solve_shortest_path(c_spo, T, arcs)
    grad = 2 * (w_true - w_spo)
    return grad

# ==========================================
# 4. Training Loop
# ==========================================
print("Training SPO+ model...")

# Create sliding windows from Train data
X_batch_list = []
C_batch_list = []

for i in range(len(X_train_raw) - H + 1):
    x_window = X_train_raw[i : i + H].flatten() # Flatten 30 days of features
    d_window = D_train[i : i + H]
    c_window = build_arc_costs(d_window, arcs, c_p, h, K)
    X_batch_list.append(x_window)
    C_batch_list.append(c_window)

X_batch = np.array(X_batch_list)
C_batch = np.array(C_batch_list)

# Linear Model Parameters: c_hat = W^T x + b
P_dim = X_batch.shape[1]
np.random.seed(42)
W = np.random.randn(P_dim, M) * 0.01
b = np.zeros(M)

lr = 1e-5
epochs = 10  # Adjust as needed
batch_size = 32

for epoch in range(epochs):
    indices = np.arange(len(X_batch))
    np.random.shuffle(indices)
    epoch_loss = 0
    
    for start_idx in range(0, len(X_batch), batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        X_sub = X_batch[batch_idx]
        C_sub = C_batch[batch_idx]
        
        # Forward
        C_hat = X_sub @ W + b
        
        # Backward (Gradient Accumulation)
        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)
        
        for k in range(len(X_sub)):
            g = get_spo_grad(C_hat[k], C_sub[k], T, arcs)
            grad_W += np.outer(X_sub[k], g)
            grad_b += g
            
        W -= lr * (grad_W / len(X_sub))
        b -= lr * (grad_b / len(X_sub))
    
    print(f"Epoch {epoch+1}/{epochs} completed")

# ==========================================
# 5. Generate Plan for Test Set
# ==========================================
print("Generating test plan...")

# Strategy: The test set is ~92 days. Our model plans for 30 days.
# We will run the model on consecutive 30-day blocks (Day 0-30, 30-60, 60-90).
# The remaining 2 days will be ignored (orders set to 0) or we can handle specifically.

orders_final = np.zeros(len(test_data))
num_blocks = len(test_data) // H

for block in range(num_blocks + 1):
    start = block * H
    end = start + H
    
    # Stop if we don't have a full 30-day block left
    if end > len(test_data):
        break
        
    # Prepare features
    x_test_window = X_test_raw[start:end].flatten()
    
    # Predict costs
    c_hat_test = x_test_window @ W + b
    
    # Optimize
    w_opt = solve_shortest_path(c_hat_test, T, arcs)
    
    # Convert path to orders
    # If arc (u, v) is chosen, we order at day (start + u)
    # Quantity is sum of Demand[start+u : start+v]
    d_window_test = D_test[start:end]
    
    for idx, val in enumerate(w_opt):
        if val > 0.5: # Arc selected
            u, v = arcs[idx]
            # Calculate quantity required for this segment
            qty = d_window_test[u:v].sum()
            orders_final[start + u] = qty

# Save to CSV
output_df = test_data[['Date', 'Store ID', 'Product ID']].copy()
output_df['SPO_Order_Qty'] = orders_final
output_df.to_csv('last_3_months_demand_forecast_SPO.csv', index=False)

print("Plan saved to 'last_3_months_demand_forecast_SPO.csv'.")
print(output_df.head())