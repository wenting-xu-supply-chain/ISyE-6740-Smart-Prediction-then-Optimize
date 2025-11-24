import pandas as pd
import numpy as np

def main():
    print("正在启动 SPO 策略绩效评估 (修正版)...")
    
    # 1. 读取数据
    spo_file = 'last_3_months_demand_forecast_SPO.csv'
    sales_file = 'rl_ready_sales_data_with_ww.csv'
    
    print(f"读取策略文件: {spo_file}")
    spo_df = pd.read_csv(spo_file)
    
    print(f"读取环境数据: {sales_file}")
    sales_df = pd.read_csv(sales_file)
    
    # 2. 数据预处理
    spo_df['Date'] = pd.to_datetime(spo_df['Date'])
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    # 重命名 Q 以免混淆
    if 'Q' in spo_df.columns:
        spo_df = spo_df.rename(columns={'Q': 'SPO_Order_Qty'})
    spo_df['SPO_Order_Qty'] = spo_df['SPO_Order_Qty'].fillna(0)
    
    # 3. 数据合并 (Merge)
    print("正在合并数据...")
    # 仅保留双方都有的时间段 (即最后3个月 Test 集)
    merged_df = pd.merge(
        sales_df, 
        spo_df[['Date', 'Store ID', 'Product ID', 'SPO_Order_Qty']], 
        on=['Date', 'Store ID', 'Product ID'], 
        how='inner'
    )
    
    merged_df = merged_df.sort_values(['Store ID', 'Product ID', 'Date'])
    
    skus = merged_df[['Store ID', 'Product ID']].drop_duplicates().values
    print(f"共匹配到 {len(skus)} 个 SKU，开始模拟 (使用 'Units Sold' 作为真实需求)...")
    
    daily_records = []
    
    # 4. 循环模拟
    for store_id, prod_id in skus:
        sku_data = merged_df[(merged_df['Store ID'] == store_id) & 
                             (merged_df['Product ID'] == prod_id)].copy()
        
        if sku_data.empty: continue
        
        # --- 初始化参数 ---
        # 这里的 Demand 均值仅用于计算初始库存，为防止除以0，加个保险
        # 注意：这里还是用 row['Units Sold'] 的均值来估算更准，因为 Demand 列有误
        avg_sales = sku_data['Demand'].mean()
        if avg_sales == 0: avg_sales = 1
        curr_inv = avg_sales * 3
        
        pipeline = [] # 在途库存
        
        # 提取成本参数 (假设该 SKU 参数稳定，取第一行)
        first_row = sku_data.iloc[0]
        price = sku_data['Price'].median() 
        unit_cost = first_row['Unit_Cost']
        holding_cost_daily = first_row['Holding_Cost_Daily']
        stockout_cost = first_row['Stockout_Penalty']
        fixed_order_cost = first_row['Order_Cost_Fixed']
        lead_time = int(first_row['Lead_Time'])
        
        # --- 按天推演 ---
        for i in range(len(sku_data)):
            row = sku_data.iloc[i]
            date = row['Date']
            
            # 1. 获取 SPO 订货决策
            # 取整处理 (假设订货必须是整数)
            order_qty = int(round(max(0, row['SPO_Order_Qty'])))
            
            # 2. >>> 修正点：使用 Units Sold 作为真实需求 <<<
            actual_demand = row['Units Sold']
            
            # 3. 物流：更新在途
            arriving = 0
            new_pipeline = []
            for days_left, qty in pipeline:
                if days_left <= 1:
                    arriving += qty
                else:
                    new_pipeline.append((days_left - 1, qty))
            pipeline = new_pipeline
            curr_inv += arriving
            
            # 下单
            if order_qty > 0:
                pipeline.append((lead_time, order_qty))
            
            # 4. 销售
            sold = min(curr_inv, actual_demand)
            curr_inv -= sold
            missed = actual_demand - sold
            
            # 5. 财务结算 (Financial Calculation)
            # [收入]
            revenue = sold * price
            
            # [销货成本 COGS]
            cogs = sold * unit_cost
            
            # [现金采购支出] (Cash Outflow)
            cash_purchase = order_qty * unit_cost
            
            # [运营成本]
            opex_holding = curr_inv * holding_cost_daily
            opex_order = fixed_order_cost if order_qty > 0 else 0
            
            # [隐性惩罚]
            cost_stockout = missed * stockout_cost
            
            # [净利润] (Cash Flow Profit)
            # 公式：营收 - (采购 + 持有 + 订货 + 缺货惩罚)
            total_cost_flow = cash_purchase + opex_holding + opex_order + cost_stockout
            daily_net_profit = revenue - total_cost_flow
            
            daily_records.append({
                'Date': date,
                'Store ID': store_id,
                'Product ID': prod_id,
                'Actual_Demand': actual_demand, # 这里记录的是 Units Sold
                'SPO_Order_Qty': row['SPO_Order_Qty'],
                'Ordered_Qty': order_qty,
                'Sales_Qty': sold,
                'Missed_Qty': missed,
                'Inventory_Level': curr_inv,
                'Revenue': revenue,
                'COGS': cogs,
                'Cash_Purchase': cash_purchase,
                'OpEx_Holding': opex_holding,
                'OpEx_Order': opex_order,
                'Cost_Stockout': cost_stockout,
                'Daily_Net_Profit': daily_net_profit
            })

    # 5. 汇总结果
    print("正在生成 SPO 财务汇总表...")
    spo_sim_df = pd.DataFrame(daily_records)
    
    spo_summary = spo_sim_df.groupby(['Store ID', 'Product ID']).agg(
        Total_Days=('Date', 'count'),
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
    spo_summary['Gross_Profit'] = spo_summary['Total_Revenue'] - spo_summary['Total_COGS']
    spo_summary['Gross_Margin_%'] = (spo_summary['Gross_Profit'] / spo_summary['Total_Revenue'].replace(0, 1)) * 100
    
    total_invest = spo_summary['Total_Purchase_Spend'] + spo_summary['Total_Holding_Cost'] + spo_summary['Total_Order_Fixed_Cost']
    spo_summary['ROI_%'] = (spo_summary['Total_Net_Profit'] / total_invest.replace(0, 1)) * 100
    
    spo_summary['Service_Level_%'] = (spo_summary['Total_Sales'] / spo_summary['Total_Demand'].replace(0, 1)) * 100
    
    # 格式化
    cols_round = ['Gross_Margin_%', 'ROI_%', 'Service_Level_%', 'Total_Revenue', 'Total_Net_Profit']
    spo_summary[cols_round] = spo_summary[cols_round].round(2)
    
    # 6. 保存
    spo_sim_df.to_csv("spo_algorithm_daily_simulation.csv", index=False)
    spo_summary.to_csv("spo_financial_performance_summary.csv", index=False)
    
    print("-" * 30)
    print("计算完成！")
    print("真实需求数据源: 'Units Sold' column from rl_ready_sales_data_with_ww.csv")
    print("结果文件: spo_financial_performance_summary.csv")
    print("-" * 30)
    print("预览:")
    print(spo_summary[['Store ID', 'Product ID', 'Total_Net_Profit', 'ROI_%', 'Service_Level_%']].head())

if __name__ == "__main__":
    main()