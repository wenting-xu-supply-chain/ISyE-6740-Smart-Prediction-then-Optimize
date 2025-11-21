import pandas as pd
import numpy as np
import random

# 1. 读取数据 (这是包含真实销量/需求的历史数据)
df = pd.read_csv("rl_ready_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store ID', 'Product ID', 'Date'])

# 2. RL 参数
ALPHA = 0.1       
GAMMA = 0.90      
EPSILON = 0.1     
N_STATE_BINS = 15 
N_ACTIONS = 5     
N_EPOCHS = 200     

all_decisions = []
skus = df[['Store ID', 'Product ID']].drop_duplicates().values

print(f"开始 RL 模拟 (使用 Test 集的真实需求进行绩效结算)...")

for store_id, prod_id in skus:
    # 提取数据
    sku_data = df[(df['Store ID'] == store_id) & (df['Product ID'] == prod_id)].copy()
    
    # 训练集 (用于学习策略)
    train_data = sku_data[sku_data['Split'] == 'Train'].reset_index(drop=True)
    
    # 测试集 (用于评估绩效 - 这里包含了这三个月真实的 Units Sold)
    test_data = sku_data[sku_data['Split'] == 'Test'].reset_index(drop=True)
    
    # 如果测试集为空，跳过
    if test_data.empty:
        continue

    # 提取参数
    # 注意：Demand 用于计算环境状态，Evaluation 阶段会用每一天的真实 Demand
    avg_demand = train_data['Demand'].mean()
    if avg_demand == 0: avg_demand = 1
    
    price = sku_data['Price'].median()
    unit_cost = sku_data['Unit_Cost'].iloc[0]
    holding_cost = sku_data['Holding_Cost_Daily'].iloc[0]
    stockout_cost = sku_data['Stockout_Penalty'].iloc[0]
    fixed_order_cost = sku_data['Order_Cost_Fixed'].iloc[0]
    lead_time = int(sku_data['Lead_Time'].iloc[0])
    
    # Q-Table 初始化
    q_table = np.zeros((N_STATE_BINS, N_ACTIONS))
    
    def get_state(inv_pos):
        bin_idx = int(inv_pos / (0.2 * avg_demand))
        return max(0, min(bin_idx, N_STATE_BINS - 1))
    
    def get_action_qty(idx):
        return int(idx * 0.5 * avg_demand)

    # --- 训练阶段 (Learning Phase) ---
    for epoch in range(N_EPOCHS):
        curr_inv = avg_demand * 3
        pipeline = [] 
        
        for i in range(len(train_data) - 1):
            # ... (训练逻辑保持不变，目的是让 Q 表收敛) ...
            inv_pos = curr_inv + sum([x[1] for x in pipeline])
            state = get_state(inv_pos)
            
            if random.random() < EPSILON: action_idx = random.randint(0, N_ACTIONS - 1)
            else: action_idx = np.argmax(q_table[state])
            order_qty = get_action_qty(action_idx)
            
            arriving = 0
            new_pipeline = []
            for days, qty in pipeline:
                if days <= 1: arriving += qty
                else: new_pipeline.append((days-1, qty))
            pipeline = new_pipeline
            curr_inv += arriving
            if order_qty > 0: pipeline.append((lead_time, order_qty))
            
            # 这里使用的是训练集的真实需求来更新 Q 值
            demand = train_data.iloc[i]['Demand']
            sold = min(curr_inv, demand)
            curr_inv -= sold
            missed = demand - sold
            
            reward = (sold * price) - (curr_inv * holding_cost) - (missed * stockout_cost) - (fixed_order_cost if order_qty > 0 else 0) - (order_qty * unit_cost)
            
            next_inv_pos = curr_inv + sum([x[1] for x in pipeline])
            next_state = get_state(next_inv_pos)
            best_next = np.argmax(q_table[next_state])
            q_table[state][action_idx] += ALPHA * (reward + GAMMA * q_table[next_state][best_next] - q_table[state][action_idx])

    # --- 测试与绩效评估阶段 (Evaluation Phase) ---
    # 这里我们要模拟：如果当时使用了 RL 策略，在这三个月里我们会花多少钱，赚多少钱。
    curr_inv = avg_demand * 3
    pipeline = []
    
    for i in range(len(test_data)):
        date = test_data.iloc[i]['Date']
        
        # 1. 获取当天的真实需求 (Ground Truth / Actual Units Sold)
        # 这就是你提到的 "Units Sold"，即通过历史数据得知的当天真实情况
        actual_demand = test_data.iloc[i]['Demand']
        
        # 2. RL 做出决策 (基于当前库存状态)
        inv_pos = curr_inv + sum([x[1] for x in pipeline])
        state = get_state(inv_pos)
        action_idx = np.argmax(q_table[state])
        order_qty = get_action_qty(action_idx)
        
        # 3. 环境推演 (库存变化)
        arriving = 0
        new_pipeline = []
        for days, qty in pipeline:
            if days <= 1: arriving += qty
            else: new_pipeline.append((days-1, qty))
        pipeline = new_pipeline
        curr_inv += arriving
        
        if order_qty > 0: pipeline.append((lead_time, order_qty))
        
        # 4. 结算：用 "真实需求" vs "当前库存" 计算销量
        sold = min(curr_inv, actual_demand)
        curr_inv -= sold
        
        # 5. >>> 财务绩效计算 (基于真实发生的数据) <<<
        missed = actual_demand - sold
        
        # 营收 (Revenue): 实际卖出的量 * 价格
        revenue = sold * price
        
        # 销货成本 (COGS): 实际卖出的量 * 成本
        cogs = sold * unit_cost
        
        # 现金采购 (Cash Purchase): 只要订货就要付钱
        cash_purchase = order_qty * unit_cost
        
        # 运营费用
        opex_holding = curr_inv * holding_cost
        opex_order = fixed_order_cost if order_qty > 0 else 0
        
        # 缺货惩罚 (隐性成本)
        cost_stockout = missed * stockout_cost
        
        # 净利润 (Net Profit): 
        # 营收 - 采购现金流 - 持有费 - 订货费 - 缺货惩罚
        # (这个公式与 WW 代码完全一致)
        total_cost_flow = cash_purchase + opex_holding + opex_order + cost_stockout
        daily_net_profit = revenue - total_cost_flow
        
        all_decisions.append({
            'Date': date,
            'Store ID': store_id,
            'Product ID': prod_id,
            'Actual_Demand': actual_demand,  # 明确标注这是真实需求
            'Sales_Qty': sold,
            'Missed_Qty': missed,
            'Ordered_Qty': order_qty,
            'Inventory_Level': curr_inv,
            'Revenue': revenue,
            'COGS': cogs,
            'Cash_Purchase': cash_purchase,
            'OpEx_Holding': opex_holding,
            'OpEx_Order': opex_order,
            'Cost_Stockout': cost_stockout,
            'Daily_Net_Profit': daily_net_profit
        })

# 3. 汇总结果
decisions_df = pd.DataFrame(all_decisions)

print("正在生成 RL 财务报表...")

rl_summary = decisions_df.groupby(['Store ID', 'Product ID']).agg(
    Total_Days=('Date', 'count'),
    # 这里的 Total_Demand 就是 Test 集所有 actual_demand 的总和
    Total_Demand=('Actual_Demand', 'sum'), 
    Total_Sales=('Sales_Qty', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    Total_COGS=('COGS', 'sum'),
    Total_Purchase_Spend=('Cash_Purchase', 'sum'),
    Total_Holding_Cost=('OpEx_Holding', 'sum'),
    Total_Order_Fixed_Cost=('OpEx_Order', 'sum'),
    Total_Stockout_Penalty=('Cost_Stockout', 'sum'),
    Total_Net_Profit=('Daily_Net_Profit', 'sum'),
    Avg_Inventory=('Inventory_Level', 'mean')
).reset_index()

# 计算 KPI
rl_summary['Gross_Profit'] = rl_summary['Total_Revenue'] - rl_summary['Total_COGS']
rl_summary['Gross_Margin_%'] = (rl_summary['Gross_Profit'] / rl_summary['Total_Revenue'].replace(0, 1)) * 100

total_invest = rl_summary['Total_Purchase_Spend'] + rl_summary['Total_Holding_Cost'] + rl_summary['Total_Order_Fixed_Cost']
rl_summary['ROI_%'] = (rl_summary['Total_Net_Profit'] / total_invest.replace(0, 1)) * 100

rl_summary['Service_Level_%'] = (rl_summary['Total_Sales'] / rl_summary['Total_Demand'].replace(0, 1)) * 100

# 格式化
cols_round = ['Gross_Margin_%', 'ROI_%', 'Service_Level_%', 'Total_Revenue', 'Total_Net_Profit']
rl_summary[cols_round] = rl_summary[cols_round].round(2)

# 保存
decisions_df.to_csv("rl_algorithm_daily_simulation.csv", index=False)
rl_summary.to_csv("rl_financial_performance_summary.csv", index=False)

print("完成！RL 策略评估已基于 Test 集的真实需求 (Actual Demand) 计算完毕。")
print("结果文件：rl_financial_performance_summary.csv")

print("\n--- 预览 (基于真实需求的评估) ---")
print(rl_summary[['Store ID', 'Product ID', 'Total_Demand', 'Total_Sales', 'Total_Net_Profit', 'ROI_%']].head())