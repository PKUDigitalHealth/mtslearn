import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

def plot_distribution(data, column, kind='hist', bins=30, kde=True, figsize=(10, 6), save_path=None):
    """
    绘制单变量分布图（直方图、箱线图、核密度图），支持保存为图片。
    
    Parameters:
    - data (pd.DataFrame): 数据集
    - column (str): 需要绘制分布的列名
    - kind (str): 可选 'hist'（直方图）或 'box'（箱线图）
    - bins (int): 如果是直方图，指定箱子的数量
    - kde (bool): 是否叠加核密度估计
    - figsize (tuple): 图形大小
    - save_path (str): 保存图像的路径（包括文件名），如 'output/distribution.png'
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    
    plt.figure(figsize=figsize)
    
    if kind == 'hist':
        sns.histplot(data[column].dropna(), bins=bins, kde=kde, color='skyblue')
        plt.title(f"Distribution of {column}", fontsize=16)
    elif kind == 'box':
        sns.boxplot(y=data[column].dropna(), color='skyblue')
        plt.title(f"Box Plot of {column}", fontsize=16)
    else:
        raise ValueError("Invalid kind. Choose 'hist' or 'box'.")
    
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=800)
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multivariate_distribution(data, columns, kind='kde', figsize=(10, 6), save_path=None):
    """
    绘制多变量分布图（联合分布图、核密度图），支持保存为图片。
    
    Parameters:
    - data (pd.DataFrame): 数据集
    - columns (list): 需要绘制分布的两列名
    - kind (str): 可选 'scatter' 或 'kde'（核密度）
    - figsize (tuple): 图形大小
    - save_path (str): 保存图像的路径（包括文件名）
    """
    if len(columns) != 2 or any(col not in data.columns for col in columns):
        raise ValueError("Please provide exactly two valid column names.")
    
    plt.figure(figsize=figsize)
    if kind == 'scatter':
        sns.scatterplot(x=data[columns[0]], y=data[columns[1]], color='skyblue')
        plt.title(f"Scatter Plot of {columns[0]} vs {columns[1]}", fontsize=16)
    elif kind == 'kde':
        sns.kdeplot(x=data[columns[0]], y=data[columns[1]], fill=True, cmap="Blues")
        plt.title(f"KDE Plot of {columns[0]} vs {columns[1]}", fontsize=16)
    else:
        raise ValueError("Invalid kind. Choose 'scatter' or 'kde'.")
    
    plt.xlabel(columns[0], fontsize=14)
    plt.ylabel(columns[1], fontsize=14)
    plt.grid(alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=800)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(data, 
                             columns=None, 
                             figsize=(12, 8), 
                             cmap='coolwarm', 
                             save_path=None, 
                             excel_path=None):
    """
    绘制相关性热力图并将相关系数矩阵保存为 Excel 文件。

    Parameters:
    - data (pd.DataFrame): 数据集
    - columns (list): 需要分析的列名列表（默认分析所有数值型列）
    - figsize (tuple): 图形大小
    - cmap (str): 热力图的颜色映射
    - save_path (str): 保存热力图的路径（包括文件名），如 'output/heatmap.png'
    - excel_path (str): 保存相关系数矩阵的 Excel 路径，如 'output/correlation_matrix.xlsx'
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # 选择用户指定的列，或默认选择所有数值型列
    if columns:
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in the dataset.")
        data_to_analyze = data[columns]
    else:
        data_to_analyze = data.select_dtypes(include=["float64", "int64"])

    # 计算相关性矩阵
    correlation_matrix = data_to_analyze.corr()

    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap, square=True, cbar=True, linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=16)

    # 保存热力图
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=800)
        print(f"Heatmap saved to {save_path}")

    plt.show()

    # 保存相关性矩阵到 Excel
    if excel_path:
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        try:
            correlation_matrix.to_excel(excel_path)
            print(f"Correlation matrix saved to {excel_path}")
        except Exception as e:
            print(f"Failed to save correlation matrix to Excel: {e}")

    return correlation_matrix

def compare_groups(group1_data, group2_data, 
                   columns=None, 
                   test_type='auto', 
                   visualize=True, 
                   save_path=None):
    """
    比较两组数据的差异（t检验、Mann-Whitney U检验、卡方检验），并可视化结果。

    Parameters:
    - group1_data (pd.DataFrame): 第一组数据
    - group2_data (pd.DataFrame): 第二组数据
    - columns (list or str): 需要分析的列（默认分析所有共有列）
    - test_type (str): 'auto'（默认自动选择检验），'t-test'，'mannwhitney'，'chi2'
    - visualize (bool): 是否绘制可视化图（小提琴图或条形图）
    - save_path (str): 图片保存路径（如 'output/group_comparison.png'）

    Returns:
    - results_df (pd.DataFrame): 检验结果（包括P值、统计量）
    """
    # 1. 数据检查
    if not isinstance(group1_data, pd.DataFrame) or not isinstance(group2_data, pd.DataFrame):
        raise ValueError("Both group1_data and group2_data must be pandas DataFrames.")

    # 2. 确定要分析的列
    shared_columns = list(set(group1_data.columns) & set(group2_data.columns))
    if columns is None:
        columns = shared_columns
    elif isinstance(columns, str):
        columns = [columns]
    else:
        missing_cols = [col for col in columns if col not in shared_columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in both datasets.")

    # 3. 初始化结果存储
    results = []

    # 4. 遍历每一列进行检验
    for col in columns:
        data1 = group1_data[col].dropna()
        data2 = group2_data[col].dropna()

        # 数值型数据检验
        if pd.api.types.is_numeric_dtype(data1):
            if test_type == 't-test' or (test_type == 'auto' and len(data1) > 30 and len(data2) > 30):
                stat, p_value = ttest_ind(data1, data2)
                test_used = 't-test'
            else:
                stat, p_value = mannwhitneyu(data1, data2)
                test_used = 'Mann-Whitney U'
            
            # 可视化：小提琴图
            if visualize:
                combined_data = pd.DataFrame({
                    col: list(data1) + list(data2),
                    'Group': ['Group 1'] * len(data1) + ['Group 2'] * len(data2)
                })
                plt.figure(figsize=(8, 6))
                sns.violinplot(x='Group', y=col, data=combined_data)
                plt.title(f"{col} Comparison ({test_used}, p={p_value:.4f})")
                if save_path:
                    plt.savefig(f"{save_path}_{col}.png", dpi=300, bbox_inches='tight')
                plt.show()

        # 分类数据检验
        else:
            contingency_table = pd.crosstab(group1_data[col], group2_data[col])
            stat, p_value, _, _ = chi2_contingency(contingency_table)
            test_used = 'Chi-square'

            # 可视化：条形图
            if visualize:
                combined_data = pd.DataFrame({
                    col: list(group1_data[col]) + list(group2_data[col]),
                    'Group': ['Group 1'] * len(group1_data) + ['Group 2'] * len(group2_data)
                })
                plt.figure(figsize=(8, 6))
                sns.countplot(x=col, hue='Group', data=combined_data)
                plt.title(f"{col} Comparison ({test_used}, p={p_value:.4f})")
                if save_path:
                    plt.savefig(f"{save_path}_{col}.png", dpi=300, bbox_inches='tight')
                plt.show()

        # 结果记录
        results.append({
            'Column': col,
            'Test': test_used,
            'Statistic': stat,
            'P-Value': p_value
        })

    # 5. 结果输出
    results_df = pd.DataFrame(results)
    return results_df

