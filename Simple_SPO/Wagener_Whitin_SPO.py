import pandas as pd
import numpy as np

def run_wagner_whitin(demands, dates, K, h):
    """
    Wagner-Whitin 算法实现
    输入:
        demands: 需求列表
        dates: 日期列表 (pandas Timestamp)
        K: 固定订货成本 (Setup Cost)
        h: 单个商品的日持有成本 (Holding Cost per unit per day)
    输出:
        order_plan: 字典 {到货日期: 补货数量}
    """
    n = len(demands)
    if n == 0:
        return {}

    # Z[t] 表示满足从第 0 天到第 t-1 天需求的最小成本
    # Z 的索引是 1-based，Z[0] = 0
    Z = [float('inf')] * (n + 1)
    Z[0] = 0
    last_order_start = [0] * (n + 1) # 记录最优决策的回溯点

    # accumulated_H[j] 用于缓存从 j 时刻开始订货一直覆盖到当前 t 的持有成本
    accumulated_H = []

    for t in range(1, n + 1):
        # 当前处理的是索引 t-1 的需求
        current_demand_idx = t - 1
        current_demand = demands[current_demand_idx]
        current_date = dates[current_demand_idx]

        # 为新的可能的起始点（即当前天）初始化持有成本缓存
        accumulated_H.append(0.0)

        # 遍历所有可能的上一次订货点 j (从 0 到 t-1)
        for j in range(t):
            # 如果在 j 时刻下单来满足 t-1 的需求，计算新增的持有成本
            days_held = (current_date - dates[j]).days
            accumulated_H[j] += h * current_demand * days_held
            
            # 计算总成本: Z[j] (前 j 个需求的成本) + K (本次订货费) + 持有成本
            total_cost = Z[j] + K + accumulated_H[j]

            if total_cost < Z[t]:
                Z[t] = total_cost
                last_order_start[t] = j + 1 

    # 回溯生成最优计划
    arrival_plan = {} # {Arrival_Date: Qty}
    curr_t = n
    while curr_t > 0:
        start_idx = last_order_start[curr_t] - 1
        end_idx = curr_t - 1
        
        # 计算该批次的总需求
        batch_qty = sum(demands[start_idx : end_idx + 1])
        arrival_date = dates[start_idx]
        
        if batch_qty > 0:
            if arrival_date in arrival_plan:
                arrival_plan[arrival_date] += batch_qty
            else:
                arrival_plan[arrival_date] = batch_qty
            
        curr_t = start_idx
        
    return arrival_plan

def main():
    print("正在读取数据...")
    # 读取原始数据
    try:
        df = pd.read_csv("rl_ready_sales_data.csv")
    except FileNotFoundError:
        print("错误: 找不到文件 'rl_ready_sales_data.csv'。请确保文件在当前目录下。")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 初始化新列，默认值为 0
    df['WW_Order_Quantity'] = 0.0
    
    # 筛选出训练集的数据进行计算
    train_mask = df['Split'] == 'Train'
    groups = df[train_mask][['Store ID', 'Product ID']].drop_duplicates().values
    
    print(f"开始计算，共需处理 {len(groups)} 个 SKU 组合...")
    
    updates = [] # 存储待更新的数据 [(index, value), ...]
    
    for i, (store_id, prod_id) in enumerate(groups):
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1} / {len(groups)}...")

        # 获取该 SKU 的所有历史数据（按日期排序）
        # 注意：这里我们只用 'Train' 数据来生成决策
        mask = (df['Store ID'] == store_id) & (df['Product ID'] == prod_id) & train_mask
        group_df = df[mask].sort_values('Date')
        
        if group_df.empty:
            continue
            
        # 提取参数
        # 假设同一 SKU 的成本参数是固定的，取第一行即可
        K = group_df['Order_Cost_Fixed'].iloc[0]
        h = group_df['Holding_Cost_Daily'].iloc[0]
        lead_time = int(group_df['Lead_Time'].iloc[0])
        
        demands = group_df['Demand'].tolist()
        dates = group_df['Date'].tolist()
        
        # 运行 Wagner-Whitin 算法
        arrival_plan = run_wagner_whitin(demands, dates, K, h)
        
        # 将“到货计划”转换为“订货计划”
        # 订货日期 = 到货日期 - 提前期 (Lead Time)
        # 我们需要找到这个订货日期在原 DataFrame 中的索引
        
        # 创建一个 日期 -> DataFrame索引 的映射，加速查找
        date_to_idx = pd.Series(group_df.index.values, index=group_df['Date']).to_dict()
        
        for arr_date, qty in arrival_plan.items():
            order_date = arr_date - pd.Timedelta(days=lead_time)
            
            # 只有当计算出的订货日期在我们的数据范围内时，才记录
            if order_date in date_to_idx:
                idx = date_to_idx[order_date]
                updates.append((idx, qty))
    
    print("计算完成，正在合并数据...")
    
    # 批量更新 DataFrame
    for idx, qty in updates:
        df.at[idx, 'WW_Order_Quantity'] = qty
        
    # 保存结果
    output_file = "rl_ready_sales_data_with_ww.csv"
    df.to_csv(output_file, index=False)
    print(f"成功！文件已保存为: {output_file}")
    
    # 打印前几行预览
    print("\n数据预览 (Train 集):")
    print(df[train_mask][['Date', 'Store ID', 'Product ID', 'Demand', 'WW_Order_Quantity']].head(10))

if __name__ == "__main__":
    main()