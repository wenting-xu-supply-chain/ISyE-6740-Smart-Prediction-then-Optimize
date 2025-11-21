import pandas as pd
import numpy as np

# ==========================================
# 1. Wagner-Whitin 核心算法 (规划层 - 只看预测)
# ==========================================
def run_wagner_whitin(demands, dates, K, h):
    """
    输入: 预测的需求 (forecasted demands)
    输出: 最优的到货计划 {期望到货日期: 数量}
    """
    n = len(demands)
    if n == 0: return {}
    
    # Z[t] 是满足 0 到 t 需求的最小成本
    Z = [float('inf')] * (n + 1)
    Z[0] = 0
    last_order_start = [0] * (n + 1)
    
    # 动态规划
    for t in range(1, n + 1):
        for j in range(1, t + 1):
            setup_cost = K
            holding_cost = 0
            
            # 假设货物在 j 时刻(dates[j-1]) 到货，覆盖 j 到 t 的需求
            arrival_date = dates[j-1]
            for k in range(j, t + 1):
                qty = demands[k-1] # 这里用的是预测量
                days_held = (dates[k-1] - arrival_date).days
                holding_cost += h * qty * days_held
            
            total_cost = Z[j-1] + setup_cost + holding_cost
            
            if total_cost < Z[t]:
                Z[t] = total_cost
                last_order_start[t] = j
                
    # 回溯生成“期望到货计划”
    arrival_plan = {} 
    curr_t = n
    while curr_t > 0:
        start_idx = last_order_start[curr_t] - 1
        end_idx = curr_t - 1
        
        batch_qty = sum(demands[start_idx : end_idx + 1])
        arrival_date = dates[start_idx]
        
        if batch_qty > 0:
            arrival_plan[arrival_date] = batch_qty
            
        curr_t = start_idx
        
    return arrival_plan

# ==========================================
# 2. 主流程 (模拟执行层 - 用真实数据结算)
# ==========================================
def main():
    print("正在读取数据...")
    
    # 1. 真实历史数据 (Ground Truth)
    # 这里的 Demand 列是真实的 Units Sold
    df_sales = pd.read_csv("rl_ready_sales_data.csv")
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    
    # 2. 预测数据 (Forecast)
    # 这里的 Forecasted_Demand 是算法看到的“猜测值”
    df_forecast = pd.read_csv("last_3_months_demand_forecast_transformed_parameters.csv")
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

    # 筛选 Test 集
    if 'Split' in df_sales.columns:
        test_sales = df_sales[df_sales['Split'] == 'Test'].copy()
    else:
        test_sales = df_sales.copy()

    # 按时间排序
    test_sales = test_sales.sort_values(['Store ID', 'Product ID', 'Date'])

    skus = test_sales[['Store ID', 'Product ID']].drop_duplicates().values
    print(f"开始 WW 策略评估 (共 {len(skus)} 个 SKU)...")
    print("注意: 策略规划使用 Forecast，财务结算使用 Actual Demand (真实销量)。")
    
    daily_records = []

    for store_id, prod_id in skus:
        # === 数据准备 ===
        # 真实数据 (用于环境模拟和算钱)
        sku_actuals = test_sales[(test_sales['Store ID'] == store_id) & 
                                 (test_sales['Product ID'] == prod_id)].sort_values('Date').reset_index(drop=True)
        
        # 预测数据 (用于 WW 规划)
        sku_forecast = df_forecast[(df_forecast['Store ID'] == store_id) & 
                                   (df_forecast['Product ID'] == prod_id)].sort_values('Date')
        
        if sku_actuals.empty: continue

        # 提取成本参数 (假设参数一致)
        price = sku_actuals['Price'].median()
        unit_cost = sku_actuals['Unit_Cost'].iloc[0]
        holding_cost_daily = sku_actuals['Holding_Cost_Daily'].iloc[0]
        stockout_cost = sku_actuals['Stockout_Penalty'].iloc[0]
        fixed_order_cost = sku_actuals['Order_Cost_Fixed'].iloc[0]
        lead_time = int(sku_actuals['Lead_Time'].iloc[0])
        
        # === 步骤 1: 运行 WW 算法 (大脑) ===
        # 必须确保预测数据的日期覆盖了测试集日期
        start_date = sku_actuals['Date'].min()
        end_date = sku_actuals['Date'].max()
        
        # 找出对应时间段的预测
        relevant_forecast = sku_forecast[(sku_forecast['Date'] >= start_date) & 
                                         (sku_forecast['Date'] <= end_date)].copy()
        
        # 如果预测缺失，用 0 补全 (防止报错)
        if len(relevant_forecast) < len(sku_actuals):
            relevant_forecast = pd.merge(sku_actuals[['Date']], relevant_forecast, on='Date', how='left').fillna(0)
        
        # 提取预测序列
        forecast_demands = relevant_forecast['Forecasted_Demand'].tolist()
        forecast_dates = relevant_forecast['Date'].tolist()
        
        # WW 计算出“理想的到货计划”
        target_arrival_plan = run_wagner_whitin(forecast_demands, forecast_dates, fixed_order_cost, holding_cost_daily)
        
        # === 步骤 2: 模拟真实执行 (钱包) ===
        # 这里的逻辑必须模拟真实的时间流逝
        
        # 初始库存 (与 RL 保持一致)
        avg_demand_est = sku_actuals['Demand'].mean() if sku_actuals['Demand'].mean() > 0 else 1
        curr_inv = avg_demand_est * 3
        pipeline = [] # 在途库存 [(剩余天数, 数量)]
        
        for i in range(len(sku_actuals)):
            today = sku_actuals.iloc[i]['Date']
            
            # >>> 关键点: 这里取的是真实的 Demand <<<
            real_actual_demand = sku_actuals.iloc[i]['Demand'] 
            
            # 1. 决策：今天要下单吗？
            # WW 算的是“期望到货日”。如果 LeadTime = 2，今天(Day 0)下单，Day 2 到货。
            # 所以我们检查：Day 2 是否在 WW 的到货计划里？
            arrival_due_date = today + pd.Timedelta(days=lead_time)
            
            order_qty = 0
            if arrival_due_date in target_arrival_plan:
                order_qty = target_arrival_plan[arrival_due_date]
            
            # 2. 物流：更新在途库存
            arriving = 0
            new_pipeline = []
            for days_left, qty in pipeline:
                if days_left <= 1:
                    arriving += qty
                else:
                    new_pipeline.append((days_left - 1, qty))
            pipeline = new_pipeline
            curr_inv += arriving
            
            # 如果今天下单，加入管道
            if order_qty > 0:
                pipeline.append((lead_time, order_qty))
            
            # 3. 销售：发生真实交易 (用真实需求结算)
            sold = min(curr_inv, real_actual_demand)
            curr_inv -= sold
            
            # 4. 财务结算 (Financial Evaluation)
            missed = real_actual_demand - sold
            
            # [收入] 卖了多少收多少钱
            revenue = sold * price
            
            # [销货成本] 
            cogs = sold * unit_cost
            
            # [现金支出] 只要订货就得付钱 (不管卖没卖掉)
            cash_purchase = order_qty * unit_cost
            
            # [运营成本]
            opex_holding = curr_inv * holding_cost_daily
            opex_order = fixed_order_cost if order_qty > 0 else 0
            
            # [隐性惩罚]
            cost_stockout = missed * stockout_cost
            
            # [净利润] = 收入 - (采购支出 + 持有 + 订货 + 缺货)
            # 这是标准的 Cash Flow Profit 计算，严厉惩罚库存积压
            total_cost_flow = cash_purchase + opex_holding + opex_order + cost_stockout
            daily_net_profit = revenue - total_cost_flow
            
            daily_records.append({
                'Date': today,
                'Store ID': store_id,
                'Product ID': prod_id,
                'Actual_Demand': real_actual_demand, # 记录真实需求
                'Forecasted_Demand': forecast_demands[i] if i < len(forecast_demands) else 0, # 记录当时预测了多少(方便对比误差)
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

    # ==========================================
    # 3. 汇总 (与 RL 输出格式对齐)
    # ==========================================
    ww_df = pd.DataFrame(daily_records)
    
    print("正在生成 WW 财务报表...")
    
    ww_summary = ww_df.groupby(['Store ID', 'Product ID']).agg(
        Total_Days=('Date', 'count'),
        Total_Demand=('Actual_Demand', 'sum'), # 用真实需求汇总
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
    ww_summary['Gross_Profit'] = ww_summary['Total_Revenue'] - ww_summary['Total_COGS']
    ww_summary['Gross_Margin_%'] = (ww_summary['Gross_Profit'] / ww_summary['Total_Revenue'].replace(0, 1)) * 100
    
    total_invest = ww_summary['Total_Purchase_Spend'] + ww_summary['Total_Holding_Cost'] + ww_summary['Total_Order_Fixed_Cost']
    ww_summary['ROI_%'] = (ww_summary['Total_Net_Profit'] / total_invest.replace(0, 1)) * 100
    
    ww_summary['Service_Level_%'] = (ww_summary['Total_Sales'] / ww_summary['Total_Demand'].replace(0, 1)) * 100
    
    # 格式化
    cols_round = ['Gross_Margin_%', 'ROI_%', 'Service_Level_%', 'Total_Revenue', 'Total_Net_Profit']
    ww_summary[cols_round] = ww_summary[cols_round].round(2)
    
    # 保存
    ww_df.to_csv("ww_algorithm_daily_simulation.csv", index=False)
    ww_summary.to_csv("ww_financial_performance_summary.csv", index=False)
    
    print("-" * 30)
    print("WW 算法评估完成 (基于真实需求)！")
    print(f"结果文件: ww_financial_performance_summary.csv")
    print("-" * 30)
    print(ww_summary[['Store ID', 'Product ID', 'Total_Demand', 'Total_Net_Profit', 'ROI_%', 'Service_Level_%']].head())

if __name__ == "__main__":
    main()