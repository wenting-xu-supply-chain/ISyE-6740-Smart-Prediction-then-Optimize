import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# Global Helpers (Graph & Optimization)
# ==========================================
H = 30   # Planning Horizon
T = H    # Number of time steps in horizon

# Generate arcs for a full connected DAG over H nodes
# This structure is constant for all SKUs
ARCS = []
for i in range(T):
    for j in range(i + 1, T + 1):
        ARCS.append((i, j))
M = len(ARCS)

def build_arc_costs(d, arcs, c_p, h, K):
    """Calculate TRUE costs for arcs based on realized demand d."""
    c = np.zeros(len(arcs))
    d = np.array(d)
    for idx, (i, j) in enumerate(arcs):
        segment = d[i:j]
        qty = segment.sum()
        times = np.arange(i, j)
        # Holding cost: sum(h * (time_held) * units)
        holding = np.sum(h * (times - i) * segment)
        c[idx] = K + c_p * qty + holding
    return c

def solve_shortest_path(c, T, arcs):
    """Dynamic Programming to find shortest path."""
    V = np.full(T + 1, np.inf)
    parent = np.full(T + 1, -1)
    path_arc_idx = np.full(T + 1, -1)
    V[0] = 0
    
    for i in range(T):
        if V[i] == np.inf: continue
        for idx, (u, v) in enumerate(arcs):
            if u == i:
                if V[i] + c[idx] < V[v]:
                    V[v] = V[i] + c[idx]
                    parent[v] = u
                    path_arc_idx[v] = idx
    
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
    c_spo = 2 * c_hat - c_true
    w_spo = solve_shortest_path(c_spo, T, arcs)
    grad = 2 * (w_true - w_spo)
    return grad

def process_single_sku(df, store_id, prod_id):
    """Runs the entire SPO+ pipeline for one SKU."""
    
    # 1. Filter Data
    data = df[(df['Store ID'] == store_id) & (df['Product ID'] == prod_id)].copy()
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Skip if data is insufficient or missing splits
    if data.empty or 'Train' not in data['Split'].values or 'Test' not in data['Split'].values:
        return None

    # 2. Feature Engineering
    data['day_of_week'] = data['Date'].dt.weekday
    data['month'] = data['Date'].dt.month

    numeric_features = ['Price', 'Discount', 'Competitor Pricing', 'Inventory Level‘, 'Epidemic']
    categorical_features = ['Weather Condition', 'Seasonality', 'Promotion', 'day_of_week', 'month']

    train_mask = data['Split'] == 'Train'
    test_mask = data['Split'] == 'Test'
    train_data = data[train_mask].copy()
    test_data = data[test_mask].copy()

    # Check if we have enough training data for at least one horizon
    if len(train_data) < H:
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    try:
        X_train_raw = preprocessor.fit_transform(train_data)
        X_test_raw = preprocessor.transform(test_data)
    except Exception as e:
        print(f"Error preprocessing {store_id}-{prod_id}: {e}")
        return None

    D_train = train_data['Demand'].values
    D_test = test_data['Demand'].values

    # Cost Parameters (Assume constant per SKU)
    c_p = data['Unit_Cost'].iloc[0]
    h = data['Holding_Cost_Daily'].iloc[0]
    K = data['Order_Cost_Fixed'].iloc[0]

    # 3. Prepare Training Batches (Sliding Window)
    X_batch_list = []
    C_batch_list = []

    for i in range(len(X_train_raw) - H + 1):
        x_window = X_train_raw[i : i + H].flatten()
        d_window = D_train[i : i + H]
        c_window = build_arc_costs(d_window, ARCS, c_p, h, K)
        X_batch_list.append(x_window)
        C_batch_list.append(c_window)

    X_batch = np.array(X_batch_list)
    C_batch = np.array(C_batch_list)

    # 4. Train Linear Model (SGD)
    P_dim = X_batch.shape[1]
    np.random.seed(42)
    W = np.random.randn(P_dim, M) * 0.01
    b = np.zeros(M)

    lr = 1e-5
    epochs = 50  # Lower epochs for speed across many SKUs, increase for accuracy
    batch_size = 32

    for epoch in range(epochs):
        indices = np.arange(len(X_batch))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(X_batch), batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            X_sub = X_batch[batch_idx]
            C_sub = C_batch[batch_idx]
            
            C_hat = X_sub @ W + b
            
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)
            
            for k in range(len(X_sub)):
                g = get_spo_grad(C_hat[k], C_sub[k], T, ARCS)
                grad_W += np.outer(X_sub[k], g)
                grad_b += g
                
            W -= lr * (grad_W / len(X_sub))
            b -= lr * (grad_b / len(X_sub))

    # 5. Generate Test Plan (30-day blocks)
    orders_final = np.zeros(len(test_data))
    num_blocks = len(test_data) // H

    for block in range(num_blocks + 1):
        start = block * H
        end = start + H
        
        if end > len(test_data):
            break # Ignore partial tail for safety or handle specifically
            
        x_test_window = X_test_raw[start:end].flatten()
        c_hat_test = x_test_window @ W + b
        w_opt = solve_shortest_path(c_hat_test, T, ARCS)
        
        d_window_test = D_test[start:end]
        
        for idx, val in enumerate(w_opt):
            if val > 0.5:
                u, v = ARCS[idx]
                qty = d_window_test[u:v].sum()
                orders_final[start + u] = qty

    # 6. Return Result DataFrame
    result_df = test_data[['Date', 'Store ID', 'Product ID']].copy()
    result_df['SPO_Order_Qty'] = orders_final
    return result_df

# ==========================================
# Main Execution
# ==========================================
def main():
    print("Loading data...")
    df = pd.read_csv('rl_ready_sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Identify all SKUs
    skus = df[['Store ID', 'Product ID']].drop_duplicates().values
    print(f"Found {len(skus)} unique SKUs. Starting batch processing...")

    all_results = []
    
    for i, (store_id, prod_id) in enumerate(skus):
        print(f"[{i+1}/{len(skus)}] Processing {store_id} - {prod_id}...")
        sku_result = process_single_sku(df, store_id, prod_id)
        
        if sku_result is not None:
            all_results.append(sku_result)
        else:
            print(f"Skipped {store_id}-{prod_id} (Insufficient data)")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_filename = 'last_3_months_demand_forecast_SPO.csv'
        final_df.to_csv(output_filename, index=False)
        print("-" * 30)
        print(f"Batch processing complete.")
        print(f"Total plan generated for {len(all_results)} SKUs.")
        print(f"Saved to: {output_filename}")
        print("-" * 30)
        print(final_df.head())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
