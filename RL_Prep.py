import pandas as pd
import numpy as np

# 1. 加载原始数据
df = pd.read_csv("sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ===============================
# 2. 定义业务假设 (Business Assumptions)
# ===============================
ANNUAL_HOLDING_RATE = 0.25   # 年持有成本率 25%
GROSS_MARGIN_RATE   = 0.30   # 毛利率 30%
SHORTAGE_MULTIPLIER = 3.0    # 缺货惩罚系数
ORDER_COST_BASE = {          # 各品类固定订货成本
    "Electronics": 80.0,
    "Clothing":    40.0,
    "Groceries":   50.0,
}
DEFAULT_ORDER_COST = 50.0
LEAD_TIME_DAYS = 2           # 根据数据分析得出的补货提前期

# ==================================================
# 3. 计算商品静态参数 (Cost & ABC Class)
# ==================================================
# 使用中位数价格以平滑促销带来的波动
prod_stats = df.groupby(["Store ID", "Product ID", "Category"], as_index=False).agg({
    "Price": "median",
    "Units Sold": "sum"
})

# >>> ABC 分类计算 <<<
prod_stats["Revenue"] = prod_stats["Price"] * prod_stats["Units Sold"]
prod_stats = prod_stats.sort_values(["Store ID", "Revenue"], ascending=[True, False])
prod_stats["Revenue_Share"] = prod_stats.groupby("Store ID")["Revenue"].transform(lambda x: x / x.sum())
prod_stats["CumShare"] = prod_stats.groupby("Store ID")["Revenue_Share"].cumsum()

def classify_abc(cum_share):
    if cum_share <= 0.80: return "A"
    elif cum_share <= 0.95: return "B"
    else: return "C"

prod_stats["ABC_Class"] = prod_stats["CumShare"].apply(classify_abc)

# >>> 成本参数计算 <<<
prod_stats["Unit_Cost"] = prod_stats["Price"] * (1 - GROSS_MARGIN_RATE)
prod_stats["Holding_Cost_Daily"] = prod_stats["Unit_Cost"] * ANNUAL_HOLDING_RATE / 365.0
prod_stats["Stockout_Penalty"] = SHORTAGE_MULTIPLIER * (prod_stats["Price"] - prod_stats["Unit_Cost"])
prod_stats["Order_Cost_Fixed"] = prod_stats["Category"].map(ORDER_COST_BASE).fillna(DEFAULT_ORDER_COST)
prod_stats["Lead_Time"] = LEAD_TIME_DAYS

# 提取需要合并回原表的列
static_params = prod_stats[[
    "Store ID", "Product ID", 
    "ABC_Class", "Unit_Cost", "Holding_Cost_Daily", 
    "Stockout_Penalty", "Order_Cost_Fixed", "Lead_Time"
]]

# ==================================================
# 4. 合并回时间序列数据 (Merge)
# ==================================================
rl_df = df.merge(static_params, on=["Store ID", "Product ID"], how="left")

# ==================================================
# 5. 划分训练集/测试集 (Train/Test Split)
# ==================================================
# 设定最后3个月为测试集
max_date = rl_df["Date"].max()
cutoff_date = max_date - pd.DateOffset(months=3)

rl_df["Split"] = np.where(rl_df["Date"] > cutoff_date, "Test", "Train")

# ==================================================
# 6. 格式整理与保存
# ==================================================
rl_df = rl_df.sort_values(["Store ID", "Product ID", "Date"])

output_filename = "rl_ready_sales_data.csv"
rl_df.to_csv(output_filename, index=False)

print(f"✅ 转换完成！文件已保存为: {output_filename}")
print(f"数据时间范围: {rl_df['Date'].min().date()} 至 {rl_df['Date'].max().date()}")
print(f"测试集划分点: {cutoff_date.date()} 之后的数据")
print(f"训练集样本数: {len(rl_df[rl_df['Split']=='Train'])}")
print(f"测试集样本数: {len(rl_df[rl_df['Split']=='Test'])}")