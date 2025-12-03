import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
import joblib
import statsmodels.api as sm

# --- Data Loading and Initial Processing ---
df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Extract time-based features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

# Sort the DataFrame
df = df.sort_values(by='Date').reset_index(drop=True)

# --- Feature Engineering (Lag & Rolling) ---
lag_period = 7
for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand', 'Price']:
    df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)

rolling_window = 7
for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand', 'Price']:
    df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
    df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)

df = df.fillna(0)

# --- Data Preparation ---
features = [col for col in df.columns if col not in ['Date', 'Demand', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
X = df[features]
y = df['Demand']

X = pd.get_dummies(X, columns=['Discount', 'Promotion'])
training_columns = X.columns.tolist()

test_date = df['Date'].max() - pd.DateOffset(months=3)
X_train = X[df['Date'] <= test_date]
y_train = y[df['Date'] <= test_date]
X_test = X[df['Date'] > test_date]
y_test = y[df['Date'] > test_date]

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# --- Sequence Preparation ---
def create_sequences(X, y, sequence_length):
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length])
    return np.array(X_sequences), np.array(y_sequences)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sequence_length_transformer = 7
X_train_transformer, y_train_transformer = create_sequences(X_train_scaled, y_train.values, sequence_length_transformer)
X_test_transformer, y_test_transformer = create_sequences(X_test_scaled, y_test.values, sequence_length_transformer)

# --- Transformer Model Definition ---
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs=inputs, outputs=outputs)

input_shape_transformer = (X_train_transformer.shape[1], X_train_transformer.shape[2])
head_size = 128
num_heads = 2
ff_dim = 2
num_transformer_blocks = 2
mlp_units = [64]
dropout = 0.1
mlp_dropout = 0.1
learning_rate = 0.001

model_transformer = build_transformer_model(
    input_shape_transformer,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout
)

model_transformer.compile(
    loss="mse",
    optimizer=Adam(learning_rate=learning_rate)
)

early_stopping_transformer = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_transformer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

# --- Training Transformer ---
model_transformer.fit(
    X_train_transformer,
    y_train_transformer,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping_transformer, reduce_lr_transformer],
    verbose=0
)

# --- Hybrid Integration (Transformer + XGBoost) ---
transformer_test_predictions_scaled = model_transformer.predict(X_test_transformer, verbose=0)

X_test_aligned_transformer = X_test.iloc[sequence_length_transformer:].copy()
X_test_transformer_xgb = X_test_aligned_transformer.copy()
X_test_transformer_xgb['transformer_predictions_scaled'] = transformer_test_predictions_scaled
y_test_transformer_xgb = y_test.values[sequence_length_transformer:].copy()

model_transformer_xgb = xgb.XGBRegressor(objective='reg:squarederror',
                                         n_estimators=1000,
                                         learning_rate=0.05,
                                         max_depth=7,
                                         min_child_weight=1,
                                         gamma=0.2,
                                         subsample=0.8,
                                         colsample_bytree=0.7,
                                         reg_alpha=0.005,
                                         random_state=42,
                                         n_jobs=-1)

model_transformer_xgb.fit(X_test_transformer_xgb, y_test_transformer_xgb)

final_predictions_transformer_xgb = model_transformer_xgb.predict(X_test_transformer_xgb)

epsilon = 1e-8
y_test_transformer_xgb_safe = y_test_transformer_xgb.copy()
y_test_transformer_xgb_safe[y_test_transformer_xgb_safe == 0] = epsilon
mape_transformer_xgb = mean_absolute_percentage_error(y_test_transformer_xgb_safe, final_predictions_transformer_xgb)
print(f"MAPE for Transformer + XGBoost: {mape_transformer_xgb:.2%}")

# --- Evaluation and Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_transformer_xgb, y=final_predictions_transformer_xgb, alpha=0.6)
plt.title('Transformer + XGBoost: Actual vs. Predicted Demand')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(y_test_transformer_xgb[:100], label='Actual Demand')
plt.plot(final_predictions_transformer_xgb[:100], label='Transformer + XGBoost Predictions')
plt.title('Transformer + XGBoost: Actual vs. Predicted Demand (First 100 Test Points)')
plt.xlabel('Time Step')
plt.ylabel('Demand Forecast')
plt.legend()
plt.grid(True)
plt.show()

errors = y_test_transformer_xgb - final_predictions_transformer_xgb
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.title('Transformer + XGBoost: Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(final_predictions_transformer_xgb, errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Transformer + XGBoost: Residuals vs. Predicted Values')
plt.xlabel('Predicted Demand')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
sm.qqplot(errors, line='s')
plt.title('Transformer + XGBoost: Q-Q Plot of Residuals')
plt.show()

# --- Wrapper Class ---
class TransformerPredictor:
    def __init__(self, model, scaler, training_columns, sequence_length):
        self.model = model
        self.scaler = scaler
        self.training_columns = training_columns
        self.sequence_length = sequence_length

    def preprocess(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)

        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)

        lag_period = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
             if 'Store ID' in df.columns and 'Product ID' in df.columns:
                 df[f'{col}_lag_{lag_period}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag_period)
             else:
                 df[f'{col}_lag_{lag_period}'] = df[col].shift(lag_period)

        rolling_window = 7
        for col in ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price']:
             if 'Store ID' in df.columns and 'Product ID' in df.columns:
                df[f'{col}_rolling_mean_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df.groupby(['Store ID', 'Product ID'])[col].rolling(window=rolling_window).std().reset_index(drop=True)
             else:
                df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(window=rolling_window).mean().reset_index(drop=True)
                df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(window=rolling_window).std().reset_index(drop=True)

        df = df.fillna(0)

        features_to_process = [col for col in df.columns if col not in ['Date', 'Demand Forecast', 'Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']]
        df_processed = pd.get_dummies(df[features_to_process], columns=['Discount', 'Holiday/Promotion'])

        for col in self.training_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[self.training_columns]

        X_scaled = self.scaler.transform(df_processed)

        X_sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
             X_sequences.append(X_scaled[i:(i + self.sequence_length)])

        if not X_sequences:
            return np.array([]), df.iloc[self.sequence_length -1:]
        else:
            return np.array(X_sequences), df.iloc[self.sequence_length -1:].reset_index(drop=True)

    def predict(self, df):
        X_seq, original_df = self.preprocess(df)
        if X_seq.size == 0:
            return np.array([]), original_df

        predictions_scaled = self.model.predict(X_seq)
        return predictions_scaled.flatten(), original_df

transformer_predictor = TransformerPredictor(model_transformer, scaler, training_columns, sequence_length_transformer)

# --- Save Artifacts ---
model_transformer.save("transformer_model.keras")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model_transformer_xgb, f)

with open("training_info.pkl", "wb") as f:
    pickle.dump({
        "training_columns": training_columns,
        "sequence_length": sequence_length_transformer
    }, f)

# --- Generate Forecast CSV ---
test_date = df['Date'].max() - pd.DateOffset(months=3)
test_mask = df['Date'] > test_date
X_test = X[test_mask]

aligned_idx = X_test.iloc[sequence_length_transformer:].index

assert len(aligned_idx) == len(final_predictions_transformer_xgb), \
    f"Index length {len(aligned_idx)} != predictions length {len(final_predictions_transformer_xgb)}"

forecast_df = df.loc[aligned_idx, ['Date', 'Store ID', 'Product ID']].copy()
forecast_df['Forecasted_Demand'] = final_predictions_transformer_xgb
forecast_df = forecast_df.sort_values(['Date', 'Store ID', 'Product ID']).reset_index(drop=True)

output_path = "last_3_months_demand_forecast.csv"
forecast_df.to_csv(output_path, index=False)
print(f"Forecast CSV saved to: {output_path}")

# --- Transform Dataset Parameters (Business Logic) ---
ANNUAL_HOLDING_RATE = 0.25
GROSS_MARGIN_RATE   = 0.30
SHORTAGE_MULTIPLIER = 3.0
ORDER_COST_BASE = {
    "Electronics": 80.0,
    "Clothing":    40.0,
    "Groceries":   50.0,
}
DEFAULT_ORDER_COST = 50.0

SERVICE_LEVEL_MAP = {
    "A": 0.99,
    "B": 0.95,
    "C": 0.90
}

prod_info = (
    df.groupby(["Store ID", "Product ID", "Category"], as_index=False)
      .agg({"Price": "median"})
)

rev = (
    df.assign(Revenue=df["Price"] * df["Units Sold"])
      .groupby(["Store ID", "Product ID"], as_index=False)["Revenue"]
      .sum()
)

rev = rev.sort_values(["Store ID", "Revenue"], ascending=[True, False])
rev["Revenue_Share"] = rev.groupby("Store ID")["Revenue"].transform(lambda x: x / x.sum())
rev["CumShare"] = rev.groupby("Store ID")["Revenue_Share"].cumsum()

def classify_abc(cum_share):
    if cum_share <= 0.80:
        return "A"
    elif cum_share <= 0.95:
        return "B"
    else:
        return "C"

rev["ABC_Class"] = rev["CumShare"].apply(classify_abc)
rev["Service_Level_Target"] = rev["ABC_Class"].map(SERVICE_LEVEL_MAP)

prod_info = prod_info.merge(
    rev[["Store ID", "Product ID", "ABC_Class", "Service_Level_Target"]],
    on=["Store ID", "Product ID"],
    how="left"
)

prod_info["Unit_Cost_Est"] = prod_info["Price"] * (1 - GROSS_MARGIN_RATE)
prod_info["h_per_day"] = prod_info["Unit_Cost_Est"] * ANNUAL_HOLDING_RATE / 365.0
prod_info["p_shortage"] = SHORTAGE_MULTIPLIER * (prod_info["Price"] - prod_info["Unit_Cost_Est"])
prod_info["K_order"] = prod_info["Category"].map(ORDER_COST_BASE).fillna(DEFAULT_ORDER_COST)

forecast_with_costs = forecast_df.merge(
     prod_info[["Store ID", "Product ID",
                "Unit_Cost_Est", "h_per_day",
                "p_shortage", "K_order",
                "Service_Level_Target"]],
     on=["Store ID", "Product ID"],
     how="left"
 )

output_path_params = "last_3_months_demand_forecast_transformed_parameters.csv"
forecast_with_costs.to_csv(output_path_params, index=False)
print(f"Transformed parameters CSV saved to: {output_path_params}")
