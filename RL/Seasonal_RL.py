import pandas as pd
import numpy as np
import random

# ==============================
# 1. 读取数据 & 基本预处理
# ==============================

df = pd.read_csv("rl_ready_sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store ID", "Product ID", "Date"])

# ------------------------------
# 季节性 & 促销特征 编码
# ------------------------------
# 这里直接使用数据中的 Seasonality（如 Winter/Spring/...）和 Promotion (0/1)
SEASON_COL = "Seasonality"
PROMO_COL = "Promotion"

# 把 Seasonality 统一编码为 0,1,2,3,...
season_values = sorted(df[SEASON_COL].dropna().unique())
season_to_idx = {s: i for i, s in enumerate(season_values)}
N_SEASON_STATES = len(season_values)

# 促销就用 0/1 两个状态
N_PROMO_STATES = 2  # 0: 非促销, 1: 促销日

# ==============================
# 2. RL 超参数配置
# ==============================

ALPHA = 0.1          # 学习率
GAMMA = 0.95         # 折现因子（稍微看重长期一些）
EPSILON_START = 0.3  # 探索初始值
EPSILON_END = 0.05   # 探索最终值（逐步衰减）
N_EPOCHS = 100       # 训练轮数

N_INV_BINS = 15      # 库存位置分箱数
N_ACTIONS = 5        # 动作档位：0,0.5D,1D,1.5D,2D

# 状态总数 = 库存 bins × Seasonality × Promotion
N_STATES = N_INV_BINS * N_SEASON_STATES * N_PROMO_STATES

# ==============================
# 3. 辅助函数：状态 & 动作
# ==============================

def get_state_index(inv_pos, avg_demand, season_value, promo_value):
    """
    把 (库存位置, 季节, 促销) 映射成一个一维的状态索引，用于 Q-Table.
    """
    # 库存位置分箱
    # 0.2 * avg_demand 对应一个 bin，大致覆盖 0 ~ 3*avg_demand
    bin_size = 0.2 * avg_demand
    if bin_size <= 0:
        bin_size = 1.0
    inv_bin = int(inv_pos / bin_size)
    inv_bin = max(0, min(inv_bin, N_INV_BINS - 1))
    
    # Seasonality -> index
    season_idx = season_to_idx.get(season_value, 0)
    
    # Promotion -> 0/1
    promo_idx = 1 if promo_value else 0
    
    # 三维 (inv_bin, season_idx, promo_idx) 拉平成一维索引
    state_idx = (
        inv_bin
        + N_INV_BINS * season_idx
        + N_INV_BINS * N_SEASON_STATES * promo_idx
    )
    return int(state_idx)

def get_action_qty(action_idx, avg_demand):
    """
    动作定义：订货量 = idx * 0.5 * avg_demand
    使用 ceil 保证对慢动销品不会全部被截断为 0.
    """
    if avg_demand <= 0:
        avg_demand = 1.0
    raw_qty = action_idx * 0.5 * avg_demand
    return int(np.ceil(raw_qty))

def epsilon_by_epoch(epoch, total_epochs):
    """
    简单线性衰减的 ε-greedy 探索率
    """
    if total_epochs <= 1:
        return EPSILON_END
    frac = epoch / (total_epochs - 1)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * frac

# ==============================
# 4. 主循环：逐 SKU 训练 & 测试
# ==============================

all_decisions = []

skus = df[["Store ID", "Product ID"]].drop_duplicates().values
print(f"开始 RL(含季节性+促销特征) 训练与评估 {len(skus)} 个 SKU ...")

for store_id, prod_id in skus:
    sku_data = df[(df["Store ID"] == store_id) & (df["Product ID"] == prod_id)].copy()
    train_data = sku_data[sku_data["Split"] == "Train"].reset_index(drop=True)
    test_data = sku_data[sku_data["Split"] == "Test"].reset_index(drop=True)
    
    if train_data.empty or test_data.empty:
        # 没有完整的 Train/Test 的 SKU 直接跳过
        continue
    
    # 环境参数
    avg_demand = train_data["Demand"].mean()
    if avg_demand <= 0:
        avg_demand = 1.0
    
    price = sku_data["Price"].median()
    unit_cost = sku_data["Unit_Cost"].iloc[0]
    holding_cost = sku_data["Holding_Cost_Daily"].iloc[0]
    stockout_cost = sku_data["Stockout_Penalty"].iloc[0]
    fixed_order_cost = sku_data["Order_Cost_Fixed"].iloc[0]
    lead_time = int(sku_data["Lead_Time"].iloc[0])
    
    # 初始化该 SKU 的 Q-Table
    q_table = np.zeros((N_STATES, N_ACTIONS))
    
    # -------------- 训练阶段 --------------
    for epoch in range(N_EPOCHS):
        # ε 随 epoch 线性衰减
        epsilon = epsilon_by_epoch(epoch, N_EPOCHS)
        
        # 初始库存：稍微随机一下，增加探索的状态多样性
        curr_inv = avg_demand * random.uniform(1.0, 5.0)
        pipeline = []  # [(剩余天数, 数量), ...]
        
        # 注意：我们遍历到 len(train_data) - 1，是为了方便取 next_state 用 i+1 行
        for i in range(len(train_data) - 1):
            row = train_data.iloc[i]
            next_row = train_data.iloc[i + 1]
            
            # 当前 exogenous 特征
            season_now = row[SEASON_COL]
            promo_now = row[PROMO_COL]
            
            inv_pos = curr_inv + sum(qty for days, qty in pipeline)
            state_idx = get_state_index(inv_pos, avg_demand, season_now, promo_now)
            
            # ε-greedy 选动作
            if random.random() < epsilon:
                action_idx = random.randint(0, N_ACTIONS - 1)
            else:
                action_idx = int(np.argmax(q_table[state_idx]))
            
            order_qty = get_action_qty(action_idx, avg_demand)
            
            # ---- 环境一步推进（与原代码保持一致的顺序）----
            # 1) 先处理管道到货
            arriving = 0
            new_pipeline = []
            for days, qty in pipeline:
                if days <= 1:
                    arriving += qty
                else:
                    new_pipeline.append((days - 1, qty))
            pipeline = new_pipeline
            curr_inv += arriving
            
            # 2) 把今天下的单放入管道
            if order_qty > 0 and lead_time > 0:
                pipeline.append((lead_time, order_qty))
            elif order_qty > 0 and lead_time <= 0:
                # lead_time <= 0 视作当天就能到
                curr_inv += order_qty
            
            # 3) 发生需求 & 销售
            demand = row["Demand"]
            sold = min(curr_inv, demand)
            curr_inv -= sold
            missed = demand - sold
            
            # 4) Reward（与财务口径一致）
            reward = (
                sold * price
                - curr_inv * holding_cost
                - missed * stockout_cost
                - (fixed_order_cost if order_qty > 0 else 0)
                - order_qty * unit_cost
            )
            
            # 5) 计算 next_state（使用下一天的 Seasonality/Promotion）
            next_inv_pos = curr_inv + sum(qty for days, qty in pipeline)
            season_next = next_row[SEASON_COL]
            promo_next = next_row[PROMO_COL]
            next_state_idx = get_state_index(next_inv_pos, avg_demand, season_next, promo_next)
            
            # 6) Q-learning 更新
            best_next_action = int(np.argmax(q_table[next_state_idx]))
            td_target = reward + GAMMA * q_table[next_state_idx, best_next_action]
            td_error = td_target - q_table[state_idx, action_idx]
            q_table[state_idx, action_idx] += ALPHA * td_error
    
    # -------------- 测试 & 评估阶段 --------------
    curr_inv = avg_demand * 3.0
    pipeline = []
    
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        
        date = row["Date"]
        season_now = row[SEASON_COL]
        promo_now = row[PROMO_COL]
        
        inv_pos = curr_inv + sum(qty for days, qty in pipeline)
        state_idx = get_state_index(inv_pos, avg_demand, season_now, promo_now)
        
        # 直接用贪心策略
        action_idx = int(np.argmax(q_table[state_idx]))
        order_qty = get_action_qty(action_idx, avg_demand)
        
        # ---- 环境推进（顺序与训练一致）----
        arriving = 0
        new_pipeline = []
        for days, qty in pipeline:
            if days <= 1:
                arriving += qty
            else:
                new_pipeline.append((days - 1, qty))
        pipeline = new_pipeline
        curr_inv += arriving
        
        if order_qty > 0 and lead_time > 0:
            pipeline.append((lead_time, order_qty))
        elif order_qty > 0 and lead_time <= 0:
            curr_inv += order_qty
        
        demand = row["Demand"]
        sold = min(curr_inv, demand)
        curr_inv -= sold
        missed = demand - sold
        
        # ------ 财务计算（与原版/WW 对齐）------
        revenue = sold * price
        cogs = sold * unit_cost
        cash_purchase = order_qty * unit_cost
        opex_holding = curr_inv * holding_cost
        opex_order = fixed_order_cost if order_qty > 0 else 0
        cost_stockout = missed * stockout_cost
        
        total_cost_flow = cash_purchase + opex_holding + opex_order + cost_stockout
        daily_net_profit = revenue - total_cost_flow
        
        all_decisions.append(
            {
                "Date": date,
                "Store ID": store_id,
                "Product ID": prod_id,
                "Demand": demand,
                "Sales_Qty": sold,
                "Missed_Qty": missed,
                "Ordered_Qty": order_qty,
                "Inventory_Level": curr_inv,
                "Revenue": revenue,
                "COGS": cogs,
                "Cash_Purchase": cash_purchase,
                "OpEx_Holding": opex_holding,
                "OpEx_Order": opex_order,
                "Cost_Stockout": cost_stockout,
                "Daily_Net_Profit": daily_net_profit,
                # 附加一些方便后分析的字段（可选）
                "Seasonality": season_now,
                "Promotion": promo_now,
            }
        )

# ==============================
# 5. 生成明细 & 财务汇总
# ==============================

decisions_df = pd.DataFrame(all_decisions)

if decisions_df.empty:
    print("警告：没有生成任何决策记录（可能 Train/Test 切分有问题或筛选条件太严格）。")
else:
    print("正在生成 RL(含季节性+促销) 策略的财务汇总表...")

    rl_summary = (
        decisions_df.groupby(["Store ID", "Product ID"])
        .agg(
            Total_Days=("Date", "count"),
            Total_Revenue=("Revenue", "sum"),
            Total_COGS=("COGS", "sum"),
            Total_Purchase_Spend=("Cash_Purchase", "sum"),
            Total_Holding_Cost=("OpEx_Holding", "sum"),
            Total_Order_Fixed_Cost=("OpEx_Order", "sum"),
            Total_Stockout_Penalty=("Cost_Stockout", "sum"),
            Total_Net_Profit=("Daily_Net_Profit", "sum"),
            Total_Demand=("Demand", "sum"),
            Total_Sales=("Sales_Qty", "sum"),
            Avg_Inventory=("Inventory_Level", "mean"),
        )
        .reset_index()
    )

    # 计算 KPI
    rl_summary["Gross_Profit"] = rl_summary["Total_Revenue"] - rl_summary["Total_COGS"]
    rl_summary["Gross_Margin_%"] = (
        rl_summary["Gross_Profit"] / rl_summary["Total_Revenue"].replace(0, 1) * 100.0
    )

    total_invest = (
        rl_summary["Total_Purchase_Spend"]
        + rl_summary["Total_Holding_Cost"]
        + rl_summary["Total_Order_Fixed_Cost"]
    )
    rl_summary["ROI_%"] = (
        rl_summary["Total_Net_Profit"] / total_invest.replace(0, 1) * 100.0
    )

    rl_summary["Service_Level_%"] = (
        rl_summary["Total_Sales"] / rl_summary["Total_Demand"].replace(0, 1) * 100.0
    )

    cols_round = [
        "Gross_Margin_%",
        "ROI_%",
        "Service_Level_%",
        "Total_Revenue",
        "Total_Net_Profit",
    ]
    rl_summary[cols_round] = rl_summary[cols_round].round(2)

    # 保存文件（文件名稍改一下以示区别）
    decisions_df.to_csv("rl_algorithm_daily_simulation_season_promo.csv", index=False)
    rl_summary.to_csv(
        "rl_financial_performance_summary_season_promo.csv", index=False
    )

    print("完成！已生成两份文件：")
    print("1. rl_algorithm_daily_simulation_season_promo.csv (RL 每日明细，含季节/促销)")
    print("2. rl_financial_performance_summary_season_promo.csv (RL 财务汇总)")
    print("\n--- RL 财务绩效表预览 ---")
    print(
        rl_summary[
            ["Store ID", "Product ID", "Total_Net_Profit", "ROI_%", "Service_Level_%"]
        ].head()
    )