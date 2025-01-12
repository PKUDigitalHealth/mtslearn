# import pandas as pd
# from visualization import plot_distribution, plot_multivariate_distribution, plot_correlation_heatmap

# if __name__ == "__main__":
#     # 1. 定义数据文件路径和图片保存路径
#     data_path = "/home/xiedonglin/mtslearn/xdl/6MWT_mts_version2.0.xlsx"
#     save_path="output/age_distribution.png"

#     # 2. 加载数据
#     try:
#         df = pd.read_excel(data_path)
#         print("Data loaded successfully.")
#         print(f"Data shape: {df.shape}")
#         print(f"Columns: {list(df.columns)}")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         exit(1)

#     # # 3. 单变量分布图测试
#     # column_to_plot = "Age"  # 替换为你的实际列名
#     # if column_to_plot in df.columns:
#     #     print(f"Plotting distribution for column: {column_to_plot}")
#     #     plot_distribution(df, column=column_to_plot, kind="hist", bins=20, save_path=save_path)
#     # else:
#     #     print(f"Column '{column_to_plot}' not found in the dataset.")

#     # # 4. 多变量分布图测试
#     # columns_to_plot = ["Age", "Height(cm)"]  # 替换为你的实际列名
#     # if all(col in df.columns for col in columns_to_plot):
#     #     print(f"Plotting multivariate distribution for columns: {columns_to_plot}")
#     #     plot_multivariate_distribution(df, columns=columns_to_plot, kind="kde", save_path=save_path)
#     # else:
#     #     print(f"Columns {columns_to_plot} not found in the dataset.")

#     # 5. 相关性热力图测试
#     correlation_matrix = plot_correlation_heatmap(
#         data=df,
#         columns=["Gender", "Age", "Height(cm)", "Weight(kg)", "BMI"],  # 指定分析的列
#         save_path="output/selected_columns_heatmap.png",  # 保存热力图
#         excel_path="output/selected_columns_correlation.xlsx"  # 保存相关矩阵
#     )

#     # # 默认分析所有数值型列
#     # correlation_matrix = plot_correlation_heatmap(
#     #     data=df,
#     #     save_path="output/all_columns_heatmap.png",
#     #     excel_path="output/all_columns_correlation.xlsx"
#     # )


import pandas as pd
from visualization import compare_groups

if __name__ == "__main__":
    # 1. 定义数据路径
    non_chd_path = "/home/xiedonglin/mtslearn/xdl/non_CHD.xlsx"
    chd_path = "/home/xiedonglin/mtslearn/xdl/CHD.xlsx"

    # 2. 加载数据
    try:
        non_chd_df = pd.read_excel(non_chd_path)
        chd_df = pd.read_excel(chd_path)
        print("数据加载成功。")
    except Exception as e:
        print(f"数据加载失败：{e}")
        exit(1)

    # 3. 分析的列名
    columns_to_compare = ["BMI", "LVEF", "Heartage"]

    # 4. 进行群体比较分析
    results_df = compare_groups(
        group1_data=non_chd_df,
        group2_data=chd_df,
        columns=columns_to_compare,
        test_type='auto',  # 自动选择检验类型
        visualize=True,    # 生成可视化图表
        save_path="output/group_comparison"  # 保存图片路径
    )

    # 5. 显示并保存分析结果
    print("分析结果：")
    print(results_df)

    # 保存结果到 Excel 文件
    results_df.to_excel("output/group_comparison_results.xlsx", index=False)
    print("结果已保存到 output/group_comparison_results.xlsx")
