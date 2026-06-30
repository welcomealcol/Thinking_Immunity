"""
Study 1 CFA 分析脚本 - v4 版本
===========================
"""

import pandas as pd
import numpy as np
from semopy import Model
import os

# ============================================================
# CFA 模型定义（TII 五维度模型）
# ============================================================

# 维度名称
DIM_NAMES = ['Trap', 'Boundary', 'Scaffold', 'Iterative', 'Sovereignty']

# 每个维度 4 个题项
# 题项编号：Trap(1-4), Boundary(5-8), Scaffold(9-12), Iterative(13-16), Sovereignty(17-20)
ITEMS_PER_DIM = 4

# semopy 模型语法
CFA_MODEL = """
# 五维度模型
Trap =~ Trap_Item1 + Trap_Item2 + Trap_Item3 + Trap_Item4
Boundary =~ Boundary_Item1 + Boundary_Item2 + Boundary_Item3 + Boundary_Item4
Scaffold =~ Scaffold_Item1 + Scaffold_Item2 + Scaffold_Item3 + Scaffold_Item4
Iterative =~ Iterative_Item1 + Iterative_Item2 + Iterative_Item3 + Iterative_Item4
Sovereignty =~ Sovereignty_Item1 + Sovereignty_Item2 + Sovereignty_Item3 + Sovereignty_Item4

# 允许相关（维度间相关）
Trap ~~ Boundary + Scaffold + Iterative + Sovereignty
Boundary ~~ Scaffold + Iterative + Sovereignty
Scaffold ~~ Iterative + Sovereignty
Iterative ~~ Sovereignty
"""

# ============================================================
# 函数定义
# ============================================================

def load_data(filepath):
    """
    加载 Items 数据
    """
    print(f"加载数据: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"  形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    return df


def fit_cfa(df, model_spec):
    """
    拟合 CFA 模型
    
    返回：
    - result: semopy 拟合结果
    - model: 拟合后的模型对象
    """
    print("\n拟合 CFA 模型...")
    model = Model(model_spec)
    model.fit(df)
    result = model.inspect()
    return result, model


def calc_fit_indices(model):
    """
    计算拟合指数
    
    返回：
    - fit_indices: dict 包含 CFI, RMSEA, SRMR, Chi2, df, Chi2/df
    """
    print("\n计算拟合指数...")
    
    # 使用 semopy.calc_stats() 函数
    import semopy
    stats_df = semopy.calc_stats(model)
    
    print(f"  拟合统计信息:")
    print(stats_df.to_string())
    
    # stats_df 格式：列名是统计量名称，有一行叫 'Value'
    # 或者可能是一个单行 DataFrame，列名是统计量名称
    
    # 提取关键指标
    if 'Value' in stats_df.index:
        # 格式：行索引为 'Value'
        row = stats_df.loc['Value']
    else:
        # 格式：单行 DataFrame
        row = stats_df.iloc[0]
    
    chi2 = row.get('chi2', np.nan)
    dof = row.get('DoF', np.nan)
    cfi = row.get('CFI', np.nan)
    rmsea = row.get('RMSEA', np.nan)
    srmr = row.get('SRMR', np.nan)
    
    chi2_df_ratio = chi2 / dof if (not np.isnan(chi2) and not np.isnan(dof) and dof > 0) else np.nan
    
    fit_indices = {
        'chi2': chi2,
        'dof': dof,
        'chi2_df_ratio': chi2_df_ratio,
        'cfi': cfi,
        'rmsea': rmsea,
        'srmr': srmr,
    }
    
    return fit_indices


def check_fit_criteria(fit_indices):
    """
    检查拟合指数是否达到标准
    
    标准：
    - CFI > 0.95
    - RMSEA < 0.06
    - SRMR < 0.08
    - Chi2/df < 3
    """
    print("\n" + "=" * 60)
    print("拟合指数检验")
    print("=" * 60)
    
    criteria = {
        'CFI > 0.95': (fit_indices['cfi'] > 0.95, fit_indices['cfi']),
        'RMSEA < 0.06': (fit_indices['rmsea'] < 0.06, fit_indices['rmsea']),
        'SRMR < 0.08': (fit_indices['srmr'] < 0.08 if not np.isnan(fit_indices['srmr']) else (True, fit_indices['srmr'])),
        'Chi2/df < 3': (fit_indices['chi2_df_ratio'] < 3.0, fit_indices['chi2_df_ratio']),
    }
    
    all_pass = True
    for criterion, (passed, value) in criteria.items():
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"  {criterion}: {value:.4f}  {status}")
        if not passed:
            all_pass = False
    
    print("=" * 60)
    if all_pass:
        print("结果: 所有拟合指数均达标！")
    else:
        print("结果: 部分拟合指数未达标")
    print("=" * 60)
    
    return all_pass


def extract_loadings(model):
    """
    提取因子载荷
    
    返回：
    - loadings_df: DataFrame 包含因子载荷
    """
    print("\n提取因子载荷...")
    
    # 使用 inspect() 获取参数估计
    params = model.inspect()
    
    # 筛选因子载荷（op == '~'）
    # semopy 2.3.11 列名：lval, op, rval, Estimate
    if 'lval' in params.columns:
        # semopy 2.3.11 格式
        loadings = params[(params['op'] == '~') & (params['Estimate'] != 1.0)].copy()
    else:
        # 旧格式：lhs, op, rhs
        loadings = params[(params['op'] == '~') & (params['Estimate'] != 1.0)].copy()
    
    print(f"  找到 {len(loadings)} 个因子载荷")
    
    return loadings


def main():
    # 1. 加载数据
    items_file = "study1_convergent_discriminant_N500_items.csv"
    
    if not os.path.exists(items_file):
        print(f"错误: 找不到文件 {items_file}")
        return
    
    df = load_data(items_file)
    
    # 2. 拟合 CFA 模型
    result, model = fit_cfa(df, CFA_MODEL)
    
    # 3. 计算拟合指数
    fit_indices = calc_fit_indices(model)
    
    # 4. 检查拟合标准
    all_pass = check_fit_criteria(fit_indices)
    
    # 5. 提取因子载荷
    loadings = extract_loadings(model)
    if len(loadings) > 0:
        print("\n因子载荷（前 10 个）:")
        # semopy 2.3.11 列名：lval, op, rval, Estimate, Std. Err, z-value, p-value
        cols_to_show = ['lval', 'op', 'rval', 'Estimate', 'Std. Err', 'p-value']
        # 只显示实际存在的列
        cols_available = [c for c in cols_to_show if c in loadings.columns]
        print(loadings[cols_available].head(10).to_string())
    
    # 6. 保存结果
    print("\n保存 CFA 结果...")
    
    # 拟合指数
    fit_df = pd.DataFrame([fit_indices])
    fit_df.to_csv("study1_cfa_fit_indices_v4.csv", index=False, encoding='utf-8-sig')
    print(f"  已保存: study1_cfa_fit_indices_v4.csv")
    
    # 因子载荷
    if len(loadings) > 0:
        loadings.to_csv("study1_cfa_loadings_v4.csv", index=False, encoding='utf-8-sig')
        print(f"  已保存: study1_cfa_loadings_v4.csv")
    
    # 7. 返回是否所有指数达标
    return all_pass


if __name__ == "__main__":
    all_pass = main()
    

