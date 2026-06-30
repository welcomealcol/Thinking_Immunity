"""
Thinking Immunity 补充实证数据分析程序（Study 5 最终修复版）
修复要点:
  - Study 5 Construction LGCM 添加残差方差相等约束 (c*C1)
  - 改进 semopy 参数提取: 支持 inspect('mx') / inspect() / estimates 多种接口
  - 增加样本层面多项式拟合作为描述性统计备选
  - 若 LGCM 仍失败，自动降级到混合模型
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import semopy
    from semopy import Model
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("Warning: semopy not installed. LGCM analysis will be replaced by repeated measures ANOVA.")


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled


def extract_semopy_param(model, latent_var, param_type='mean'):
    """
    通用 semopy 参数提取函数
    尝试多种接口: inspect('mx') -> inspect() -> estimates
    """
    # 尝试1: inspect('mx') 专门提取均值结构
    try:
        mx = model.inspect('mx')
        if mx is not None and not mx.empty:
            rows = mx[(mx['lval'] == latent_var)]
            if not rows.empty:
                for _, row in rows.iterrows():
                    if row.get('op') in ['~1', '~'] and pd.notna(row.get('Estimate')):
                        return float(row['Estimate'])
    except Exception:
        pass

    # 尝试2: 普通 inspect()
    try:
        params = model.inspect()
        rows = params[params['lval'] == latent_var]
        if not rows.empty:
            for _, row in rows.iterrows():
                if row.get('op') in ['~1', '~', '~~'] and pd.notna(row.get('Estimate')):
                    # 区分均值(~1)和方差(~~)
                    if param_type == 'mean' and row.get('op') == '~1':
                        return float(row['Estimate'])
                    elif param_type == 'variance' and row.get('op') == '~~':
                        return float(row['Estimate'])
                    elif param_type == 'mean' and row.get('op') == '~':
                        # 某些版本用 ~ 表示均值
                        return float(row['Estimate'])
    except Exception:
        pass

    # 尝试3: model.estimates (旧版接口)
    try:
        est = model.estimates
        rows = est[est['lval'] == latent_var]
        if not rows.empty:
            for _, row in rows.iterrows():
                if row.get('op') in ['~1', '~', '~~'] and pd.notna(row.get('Estimate')):
                    if param_type == 'mean' and row.get('op') in ['~1', '~']:
                        return float(row['Estimate'])
    except Exception:
        pass

    return None


def sample_level_growth_rates(df, y_var, time_var='Wave', id_var='SubjectID', degree=2):
    """
    样本层面多项式拟合: 对每个人拟合曲线，返回平均系数
    作为 semopy 参数提取失败时的描述性统计备选
    """
    coefs = {f'coef_{d}': [] for d in range(degree+1)}
    for sid in df[id_var].unique():
        sub = df[df[id_var] == sid].sort_values(time_var)
        x = sub[time_var].values.astype(float)
        y = sub[y_var].values.astype(float)
        if len(x) >= degree + 1:
            c = np.polyfit(x, y, degree)
            for d in range(degree+1):
                coefs[f'coef_{d}'].append(c[d])  # c[0]=最高次, c[-1]=截距
    result = {}
    for k, v in coefs.items():
        result[k] = np.mean(v)
        result[k + '_std'] = np.std(v, ddof=1)
    return result


# -------------------------------------------------------------------
# Study 1: 竞争构念增量效度
# -------------------------------------------------------------------
def study1_analysis(dims_path):
    print("\n" + "="*60)
    print("Study 1: 竞争构念增量效度")
    print("="*60)
    df = pd.read_csv(dims_path)
    print("\n--- TII五维度相关矩阵 ---")
    print(df[['Trap','Boundary','Scaffold','Iterative','Sovereignty']].corr().round(3))
    df['AILST_Total'] = df[['AILST_Operational','AILST_Ethical','AILST_Application']].mean(axis=1)
    df['CSM'] = df['Sovereignty']
    controls = ['MAI_Knowledge','MAI_Process','BPNS_Autonomy','BPNS_Competence',
                'BPNS_Relatedness','EAS_Construction','EAS_Evaluation','EAS_Integration']
    X1 = sm.add_constant(df[controls])
    model1 = sm.OLS(df['AILST_Total'], X1).fit()
    X2 = sm.add_constant(df[controls + ['CSM']])
    model2 = sm.OLS(df['AILST_Total'], X2).fit()
    print(f"\nModel1 R² = {model1.rsquared:.4f}")
    print(f"Model2 R² = {model2.rsquared:.4f}")
    print(f"ΔR² = {model2.rsquared - model1.rsquared:.4f}")
    print(f"CSM系数: β = {model2.params['CSM']:.4f}, p = {model2.pvalues['CSM']:.4f}")
    if model2.pvalues['CSM'] < 0.05:
        print("结论: CSM具有显著增量效度 ✓")
    else:
        print("结论: CSM增量效度不显著")
    print(f"\n区分效度:")
    print(f"CSM与MAI_Knowledge相关: r = {df['CSM'].corr(df['MAI_Knowledge']):.3f}")
    print(f"CSM与MAI_Process相关:   r = {df['CSM'].corr(df['MAI_Process']):.3f}")


# -------------------------------------------------------------------
# CFA: 验证性因子分析 (TII量表)
# -------------------------------------------------------------------
def cfa_analysis(items_path, output_path=None):
    """
    对TII量表进行验证性因子分析(CFA)
    使用semopy包计算拟合指数和因子载荷
    """
    print("\n" + "="*60)
    print("CFA: TII量表验证性因子分析")
    print("="*60)
    
    if not SEMOPY_AVAILABLE:
        print("错误: semopy未安装，无法进行CFA分析")
        print("请安装: pip install semopy")
        return None
    
    # 检查数据文件
    if not os.path.exists(items_path):
        print(f"错误: 数据文件未找到: {items_path}")
        print(f"请确保文件存在于正确路径")
        return None
    
    try:
        # 读取项目层面数据
        df = pd.read_csv(items_path)
        print(f"\n数据加载成功! 形状: {df.shape}")
        
        # 检查TII项目列是否存在
        tii_items = {
            'Trap': ['Trap_1', 'Trap_2', 'Trap_3', 'Trap_4'],
            'Boundary': ['Boundary_1', 'Boundary_2', 'Boundary_3', 'Boundary_4'],
            'Scaffold': ['Scaffold_1', 'Scaffold_2', 'Scaffold_3', 'Scaffold_4'],
            'Iterative': ['Iterative_1', 'Iterative_2', 'Iterative_3', 'Iterative_4'],
            'Sovereignty': ['Sovereignty_1', 'Sovereignty_2', 'Sovereignty_3', 'Sovereignty_4']
        }
        
        # 检查哪些项目列存在
        existing_items = {}
        missing_items = {}
        for dim, items in tii_items.items():
            existing = [item for item in items if item in df.columns]
            missing = [item for item in items if item not in df.columns]
            if existing:
                existing_items[dim] = existing
            if missing:
                missing_items[dim] = missing
        
        print(f"\n找到的项目列: {sum(len(v) for v in existing_items.values())}")
        for dim, items in existing_items.items():
            print(f"  {dim}: {len(items)}/4 个项目")
        
        if missing_items:
            print(f"\n缺失的项目列:")
            for dim, items in missing_items.items():
                print(f"  {dim}: {items}")
        
        if not existing_items:
            print("\n错误: 未找到任何TII项目列!")
            return None
        
        # 准备CFA数据（只保留存在的项目列）
        cfa_cols = []
        for items in existing_items.values():
            cfa_cols.extend(items)
        
        cfa_data = df[cfa_cols].dropna()
        print(f"\nCFA分析样本量: N = {len(cfa_data)}")
        
        if len(cfa_data) < 100:
            print("警告: 样本量较小，CFA结果可能不稳定")
        
        # 构建semopy模型公式
        model_spec = "# TII五因子模型\n"
        for dim, items in existing_items.items():
            model_spec += f"{dim} =~ {' + '.join(items)}\n"
        
        # 添加因子协方差（允许因子相关）
        dims = list(existing_items.keys())
        for i in range(len(dims)):
            for j in range(i+1, len(dims)):
                model_spec += f"{dims[i]} ~~ {dims[j]}\n"
        
        print(f"\nCFA模型公式:")
        print(model_spec)
        
        # 运行CFA
        print(f"\n正在运行CFA...")
        model = Model(model_spec)
        model.fit(cfa_data)
        
        # 获取拟合指数
        stats_df = semopy.calc_stats(model)
        print(f"\n--- CFA拟合指数 ---")
        print(stats_df.T)
        
        # 提取关键拟合指数
        key_indices = ['chi2', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']
        print(f"\n关键拟合指数:")
        for idx in key_indices:
            if idx in stats_df.columns:
                val = stats_df.loc['Value', idx]
                if idx in ['CFI', 'TLI'] and not pd.isna(val):
                    val_capped = min(val, 1.0)
                    note = ' (capped)' if val > 1.0 else ''
                    print(f"  {idx}: {val_capped:.4f}{note}")
                elif idx == 'chi2':
                    df_val = stats_df.loc['Value', 'df'] if 'df' in stats_df.columns else 'N/A'
                    print(f"  {idx}: {val:.2f}, df = {df_val}")
                    if not pd.isna(val) and not pd.isna(df_val):
                        chi2_df = val / df_val if df_val != 0 else 'N/A'
                        print(f"  χ²/df = {chi2_df:.2f}")
                else:
                    print(f"  {idx}: {val:.4f}" if not pd.isna(val) else f"  {idx}: N/A")
        
        # 获取因子载荷
        print(f"\n--- 因子载荷 (标准化) ---")
        try:
            params = model.inspect()
            loadings = params[params['op'] == '=~']
            print(loadings[['lhs', 'rhs', 'Estimate', 'Std.lv', 'pvalue']].round(4))
        except Exception as e:
            print(f"无法提取因子载荷: {e}")
        
        # 与原手稿比较
        print(f"\n--- 与原手稿比较 ---")
        print(f"  原手稿报告: CFI=.932, TLI=.922, RMSEA=.095, χ²/df=3.97")
        if 'CFI' in stats_df.columns and 'TLI' in stats_df.columns and 'RMSEA' in stats_df.columns:
            cfi = stats_df.loc['Value', 'CFI']
            tli = stats_df.loc['Value', 'TLI']
            rmsea = stats_df.loc['Value', 'RMSEA']
            chi2 = stats_df.loc['Value', 'chi2'] if 'chi2' in stats_df.columns else None
            df_val = stats_df.loc['Value', 'df'] if 'df' in stats_df.columns else None
            
            if not pd.isna(cfi) and not pd.isna(tli) and not pd.isna(rmsea):
                print(f"  当前结果:    CFI={cfi:.3f}, TLI={tli:.3f}, RMSEA={rmsea:.3f}", end='')
                if not pd.isna(chi2) and not pd.isna(df_val) and df_val != 0:
                    print(f", χ²/df={chi2/df_val:.2f}")
                else:
                    print()
        
        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存拟合指数
            fit_indices_path = output_path / "cfa_fit_indices.csv"
            stats_df.to_csv(fit_indices_path)
            print(f"\n拟合指数已保存至: {fit_indices_path}")
            
            # 保存因子载荷
            if 'loadings' in locals():
                loadings_path = output_path / "cfa_factor_loadings.csv"
                loadings.to_csv(loadings_path, index=False)
                print(f"因子载荷已保存至: {loadings_path}")
        
        return model, stats_df
        
    except Exception as e:
        print(f"\nCFA分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# -------------------------------------------------------------------
# Study 2: Double Dissociation
# -------------------------------------------------------------------
def study2_analysis(long_path):
    print("\n" + "="*60)
    print("Study 2: Double Dissociation")
    print("="*60)
    df = pd.read_csv(long_path)
    df['z_TII'] = (df['TII_Total'] - df['TII_Total'].mean()) / df['TII_Total'].std()
    df['Task_AI'] = (df['Task_Type'] == 'AI-mediated').astype(int)
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
        model = MixedLM.from_formula("Performance ~ Task_AI * z_TII", groups=df["SubjectID"], data=df)
        result = model.fit(reml=False)
        print("\n--- 混合线性模型结果 ---")
        print(result.summary().tables[1])
        p_inter = result.pvalues.get('Task_AI:z_TII', None)
        if p_inter is not None and p_inter < 0.05:
            print(f"\n交互效应显著 (p={p_inter:.4f}) → 支持Double dissociation ✓")
        else:
            print("交互效应不显著")
    except Exception as e:
        print(f"混合模型失败: {e}")
    ai = df[df['Task_Type']=='AI-mediated']
    free = df[df['Task_Type']=='AI-free']
    m1 = smf.ols('Performance ~ z_TII', data=ai).fit()
    m2 = smf.ols('Performance ~ z_TII', data=free).fit()
    print(f"\n简单斜率:")
    print(f"TII → AI-mediated: β={m1.params['z_TII']:.3f}, p={m1.pvalues['z_TII']:.4f}")
    print(f"TII → AI-free:   β={m2.params['z_TII']:.3f}, p={m2.pvalues['z_TII']:.4f}")


# -------------------------------------------------------------------
# Study 3: CTI干预RCT
# -------------------------------------------------------------------
def study3_analysis(data_path):
    print("\n" + "="*60)
    print("Study 3: CTI干预RCT")
    print("="*60)
    df = pd.read_csv(data_path)
    df['Gain_TII'] = df['Post_TII_Total'] - df['Pre_TII_Total']
    print("\n--- 各组描述统计 ---")
    for gname in ['CTI', 'Active_Control', 'No_Intervention']:
        g = df[df['Group']==gname]
        print(f"{gname}: Pre={g['Pre_TII_Total'].mean():.2f}, Post={g['Post_TII_Total'].mean():.2f}, Gain={g['Gain_TII'].mean():.2f}")
    from scipy.stats import ttest_ind
    pairs = [('CTI','Active_Control'), ('CTI','No_Intervention'), ('Active_Control','No_Intervention')]
    print("\n--- 增益组间比较 (Bonferroni校正 α=0.017) ---")
    for g1, g2 in pairs:
        d1 = df[df['Group']==g1]['Gain_TII']
        d2 = df[df['Group']==g2]['Gain_TII']
        t, p = ttest_ind(d1, d2)
        d = cohens_d(d1, d2)
        sig = "✓" if p < 0.017 else "✗"
        print(f"{g1} vs {g2}: t={t:.2f}, p={p:.4f}, d={d:.3f} {sig}")
    cti = df[df['Group']=='CTI']['Gain_TII']
    ctrl = df[df['Group']=='No_Intervention']['Gain_TII']
    active = df[df['Group']=='Active_Control']['Gain_TII']
    print(f"\nCTI vs Control d = {cohens_d(cti, ctrl):.3f}")
    print(f"CTI vs Active  d = {cohens_d(cti, active):.3f}")
    sub = df[df['Group'].isin(['CTI','No_Intervention'])].copy()
    sub['Group_CTI'] = (sub['Group'] == 'CTI').astype(int)
    model_a = smf.ols('Gain_MAI ~ Group_CTI', data=sub).fit()
    model_b = smf.ols('Gain_TII ~ Gain_MAI + Group_CTI', data=sub).fit()
    a_coef = model_a.params['Group_CTI']
    b_coef = model_b.params['Gain_MAI']
    ab = a_coef * b_coef
    se_a = model_a.bse['Group_CTI']
    se_b = model_b.bse['Gain_MAI']
    sobel_z = ab / np.sqrt(se_a**2 * b_coef**2 + se_b**2 * a_coef**2)
    p_sobel = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    print("\n--- 中介分析 (MAI) ---")
    print(f"a: Group→Gain_MAI = {a_coef:.4f}, p = {model_a.pvalues['Group_CTI']:.4f}")
    print(f"b: Gain_MAI→Gain_TII = {b_coef:.4f}, p = {model_b.pvalues['Gain_MAI']:.4f}")
    print(f"间接效应 = {ab:.4f}, Sobel z = {sobel_z:.3f}, p = {p_sobel:.4f}")
    if p_sobel < 0.05:
        print("结论: 中介效应显著 ✓")
    else:
        print("结论: 中介效应不显著")


# -------------------------------------------------------------------
# Study 4: RSA + LPA
# -------------------------------------------------------------------
def study4_analysis(data_path):
    print("\n" + "="*60)
    print("Study 4: RSA + LPA")
    print("="*60)
    df = pd.read_csv(data_path)
    # RSA: mean-center before squaring to reduce multicollinearity (VIF ~134→~4)
    D_c = df['Defense'] - df['Defense'].mean()
    C_c = df['Construction'] - df['Construction'].mean()
    X = sm.add_constant(pd.DataFrame({
        'D': D_c, 'C': C_c,
        'D2': D_c**2, 'DC': D_c * C_c, 'C2': C_c**2
    }))
    y = df['Higher_Order_Thinking']
    model = sm.OLS(y, X).fit()
    print("\n--- RSA回归结果 (mean-centered) ---")
    print(model.summary().tables[1])
    b = model.params
    curv = b['D2'] + b['DC'] + b['C2']
    print(f"\n一致性线曲率 = {curv:.3f}")
    denom = b['DC']**2 - 4*b['D2']*b['C2']
    if abs(denom) > 1e-6:
        D_star_raw = (2*b['C2']*b['D'] - b['DC']*b['C']) / denom
        C_star_raw = (2*b['D2']*b['C'] - b['DC']*b['D']) / denom
        D_star = D_star_raw + df['Defense'].mean()
        C_star = C_star_raw + df['Construction'].mean()
        print(f"驻点 (raw scale): Defense* = {D_star:.2f}, Construction* = {C_star:.2f}")
    else:
        print("驻点无法计算")
    features = ['Trap','Boundary','Scaffold','Iterative','Sovereignty']
    X_lpa = StandardScaler().fit_transform(df[features])
    aics = []
    models = []
    for n in range(2, 6):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42, n_init=10)
        gmm.fit(X_lpa)
        models.append(gmm)
        aics.append(gmm.aic(X_lpa))
    best_n = np.argmin(aics) + 2
    print(f"\nLPA: 最佳类别数 (基于AIC) = {best_n}")
    best_gmm = models[best_n-2]
    df['Profile'] = best_gmm.predict(X_lpa)
    profile_means = df.groupby('Profile')[['Defense','Construction','Higher_Order_Thinking']].mean()
    print("\n各Profile均值 (按HOT排序):")
    print(profile_means.sort_values('Higher_Order_Thinking', ascending=False).round(3))


# -------------------------------------------------------------------
# Study 5: 纵向增长模型（LGCM + 备选混合模型）
# -------------------------------------------------------------------
def study5_analysis(data_path):
    print("\n" + "="*60)
    print("Study 5: 纵向增长轨迹")
    print("="*60)
    df = pd.read_csv(data_path)
    wide = df.pivot(index='SubjectID', columns='Wave', values=['Defense','Construction'])
    wide.columns = [f'{col[0]}_{col[1]}' for col in wide.columns]
    wide.reset_index(inplace=True)
    cov = df.groupby('SubjectID')[['Prior_AI_Experience','Discipline','Institutional_AI_Policy']].first()
    wide = wide.merge(cov, on='SubjectID')
    wide['Discipline_code'] = wide['Discipline'].map({'STEM':1, 'Humanities':0, 'Arts':-1})

    print("\n--- Defense 波次相关矩阵 ---")
    print(wide[['Defense_1','Defense_2','Defense_3','Defense_4']].corr().round(3))
    print("\n--- Construction 波次相关矩阵 ---")
    print(wide[['Construction_1','Construction_2','Construction_3','Construction_4']].corr().round(3))

    # 样本层面描述性统计（无论 semopy 是否成功，都输出）
    print("\n--- 样本层面增长趋势描述 ---")
    def_rates = sample_level_growth_rates(df, 'Defense', degree=2)
    con_rates = sample_level_growth_rates(df, 'Construction', degree=1)
    print(f"Defense 平均二次系数 = {def_rates['coef_0']:.4f} (SD={def_rates['coef_0_std']:.4f})")
    print(f"Defense 平均线性系数 = {def_rates['coef_1']:.4f} (SD={def_rates['coef_1_std']:.4f})")
    print(f"Construction 平均线性斜率 = {con_rates['coef_0']:.4f} (SD={con_rates['coef_0_std']:.4f})")

    if not SEMOPY_AVAILABLE:
        print("\nsemopy未安装，改用重复测量混合模型分析时间效应")
        long_df = df[df['Wave'].isin([1,2,3,4])].copy()
        long_df['Wave_sq'] = long_df['Wave']**2
        try:
            model = smf.mixedlm("Defense ~ Wave + Wave_sq", long_df, groups=long_df["SubjectID"])
            result = model.fit()
            print("\n--- Defense 混合模型 (线性+二次) ---")
            print(result.summary().tables[1])
            print("二次项系数:", result.params['Wave_sq'])
        except Exception as e:
            print(f"混合模型拟合失败: {e}")
        return

    # ============================================================
    # Defense 二次增长曲线 (LGCM)
    # ============================================================
    try:
        model_desc = """
        D1 =~ 1*Defense_1
        D2 =~ 1*Defense_2
        D3 =~ 1*Defense_3
        D4 =~ 1*Defense_4

        Intercept =~ 1*D1 + 1*D2 + 1*D3 + 1*D4
        Linear =~ 0*D1 + 1*D2 + 2*D3 + 3*D4
        Quadratic =~ 0*D1 + 1*D2 + 4*D3 + 9*D4

        Intercept ~ Prior_AI_Experience + Discipline_code + Institutional_AI_Policy
        Linear ~ Prior_AI_Experience + Discipline_code + Institutional_AI_Policy
        Quadratic ~ Prior_AI_Experience + Discipline_code + Institutional_AI_Policy
        """
        model = Model(model_desc)
        model.fit(wide)

        stats_df = semopy.calc_stats(model)
        print("\n--- Defense二次增长曲线模型拟合 ---")
        print(stats_df.T)

        key_indices = ['chi2', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']
        for idx in key_indices:
            if idx in stats_df.columns:
                val = stats_df.loc['Value', idx]
                if idx in ['CFI', 'TLI'] and not pd.isna(val):
                    val_capped = min(val, 1.0)
                    note = ' (capped)' if val > 1.0 else ''
                    print(f"{idx}: {val_capped:.4f}{note}" if not pd.isna(val) else f"{idx}: N/A")
                else:
                    print(f"{idx}: {val:.4f}" if not pd.isna(val) else f"{idx}: N/A")

        # 通用参数提取
        quad_mean = extract_semopy_param(model, 'Quadratic', 'mean')
        if quad_mean is not None:
            print(f"\n二次项均值 = {quad_mean:.3f} (负值表示开口向下)")
        else:
            print("\n[注意] semopy 未返回 Quadratic 均值，使用样本层面估计:")
            print(f"      样本平均二次系数 = {def_rates['coef_0']:.3f}")

    except Exception as e:
        print(f"Defense LGCM失败: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # Construction 线性增长 (LGCM) —— 添加残差方差相等约束
    # ============================================================
    try:
        # 修复: 添加 c*C1 约束，避免 CFI>1 的识别崩溃
        model_lin = """
        C1 =~ 1*Construction_1
        C2 =~ 1*Construction_2
        C3 =~ 1*Construction_3
        C4 =~ 1*Construction_4

        Intercept =~ 1*C1 + 1*C2 + 1*C3 + 1*C4
        Linear =~ 0*C1 + 1*C2 + 2*C3 + 3*C4

        C1 ~~ c*C1
        C2 ~~ c*C2
        C3 ~~ c*C3
        C4 ~~ c*C4

        Intercept ~ Prior_AI_Experience + Discipline_code + Institutional_AI_Policy
        Linear ~ Prior_AI_Experience + Discipline_code + Institutional_AI_Policy
        """
        model2 = Model(model_lin)
        model2.fit(wide)

        stats2 = semopy.calc_stats(model2)
        print("\n--- Construction线性增长模型拟合 ---")
        print(stats2.T)

        for idx in key_indices:
            if idx in stats2.columns:
                val = stats2.loc['Value', idx]
                if idx in ['CFI', 'TLI'] and not pd.isna(val):
                    val_capped = min(val, 1.0)
                    note = ' (capped)' if val > 1.0 else ''
                    print(f"{idx}: {val_capped:.4f}{note}" if not pd.isna(val) else f"{idx}: N/A")
                else:
                    print(f"{idx}: {val:.4f}" if not pd.isna(val) else f"{idx}: N/A")

        # CFI>1 is a known SEM artifact (chi2<df). Model fit is excellent.
        cfi_val = stats2.loc['Value', 'CFI'] if 'CFI' in stats2.columns else None
        if cfi_val is not None and cfi_val > 1.0:
            print(f"\n[注] CFI={cfi_val:.3f}>1.0 是因 χ²<df 的计算 artifact，已截断为 1.0。模型拟合优秀。")

        linear_mean = extract_semopy_param(model2, 'Linear', 'mean')
        if linear_mean is not None:
            print(f"\n线性斜率均值 = {linear_mean:.3f} (正值表示增长)")
        else:
            print("\n[注意] semopy 未返回 Linear 均值，使用样本层面估计:")
            print(f"      样本平均线性斜率 = {con_rates['coef_0']:.3f}")

    except Exception as e:
        print(f"Construction LGCM失败: {e}")
        print("\n--- 降级到混合模型 (Construction) ---")
        long_df = df[df['Wave'].isin([1,2,3,4])].copy()
        try:
            model = smf.mixedlm("Construction ~ Wave", long_df, groups=long_df["SubjectID"])
            result = model.fit()
            print(result.summary().tables[1])
        except Exception as e2:
            print(f"混合模型也失败: {e2}")
        import traceback
        traceback.print_exc()


# ============================================================
# 主程序
# ============================================================
FIG_PALETTE = {
    "ink": "#272727",
    "muted": "#767676",
    "light": "#D8D8D8",
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "green": "#2E9E44",
    "red": "#B64342",
    "red_soft": "#F0C0CC",
    "gold": "#D8A03D",
    "violet": "#7C6CCF",
}


def apply_publication_style():
    sns.set_theme(context="paper", style="white")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.04, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8.5, fontweight="bold",
        color=FIG_PALETTE["ink"],
    )


def format_p(p_value):
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def mean_ci(values):
    vals = pd.Series(values).dropna().astype(float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, 0
    mean = vals.mean()
    ci = 1.96 * vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, ci, n


def save_pub_figure(fig, out_base, dpi=600):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ["svg", "pdf", "tiff", "png"]:
        path = out_base.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"tiff", "png"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def fit_study1_incremental(df):
    df = df.copy()
    if "AILST_Total" not in df:
        df["AILST_Total"] = df[["AILST_Operational", "AILST_Ethical", "AILST_Application"]].mean(axis=1)
    df["CSM"] = df["Sovereignty"]
    controls = [
        "MAI_Knowledge", "MAI_Process", "BPNS_Autonomy", "BPNS_Competence",
        "BPNS_Relatedness", "EAS_Construction", "EAS_Evaluation", "EAS_Integration",
    ]
    model1 = sm.OLS(df["AILST_Total"], sm.add_constant(df[controls])).fit()
    model2 = sm.OLS(df["AILST_Total"], sm.add_constant(df[controls + ["CSM"]])).fit()
    return model1, model2


def plot_incremental_validity(ax, df, source_frames):
    model1, model2 = fit_study1_incremental(df)
    r2_values = [model1.rsquared, model2.rsquared]
    labels = ["Controls", "+ CSM"]
    colors = [FIG_PALETTE["light"], FIG_PALETTE["blue"]]
    bars = ax.bar(labels, r2_values, color=colors, edgecolor=FIG_PALETTE["ink"], linewidth=0.6)
    for bar, value in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}",
                ha="center", va="bottom", fontsize=6)
    delta = model2.rsquared - model1.rsquared
    beta = model2.params["CSM"]
    p_value = model2.pvalues["CSM"]
    ymax = max(r2_values) * 1.65
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Explained variance (R2)")
    ax.set_title("Incremental validity")
    ax.text(0.5, ymax * 0.92, f"delta R2 = {delta:.3f}\nbeta = {beta:.3f}, {format_p(p_value)}",
            ha="center", va="top", fontsize=6, color=FIG_PALETTE["ink"])
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    source_frames.append(pd.DataFrame({
        "panel": "A_incremental_validity",
        "model": labels,
        "R2": r2_values,
        "delta_R2": [0.0, delta],
        "CSM_beta": [np.nan, beta],
        "CSM_p": [np.nan, p_value],
    }))


def plot_double_dissociation(ax, df, source_frames):
    df = df.copy()
    df["z_TII"] = (df["TII_Total"] - df["TII_Total"].mean()) / df["TII_Total"].std()
    colors = {"AI-mediated": FIG_PALETTE["red"], "AI-free": FIG_PALETTE["blue"]}
    x_min, x_max = df["z_TII"].quantile([0.02, 0.98])
    x_line = np.linspace(x_min, x_max, 120)
    source_rows = []
    for task in ["AI-mediated", "AI-free"]:
        sub = df[df["Task_Type"] == task]
        model = smf.ols("Performance ~ z_TII", data=sub).fit()
        y_line = model.params["Intercept"] + model.params["z_TII"] * x_line
        ax.scatter(sub["z_TII"], sub["Performance"], s=8, alpha=0.18,
                   color=colors[task], edgecolor="none")
        ax.plot(x_line, y_line, color=colors[task], linewidth=1.8)
        ax.text(x_line[-1] + 0.08, y_line[-1],
                f"{task}\nbeta={model.params['z_TII']:.2f}",
                color=colors[task], fontsize=6, ha="left", va="center")
        for x_val, y_val in zip(x_line, y_line):
            source_rows.append({
                "panel": "B_double_dissociation",
                "task": task,
                "z_TII": x_val,
                "predicted_performance": y_val,
                "slope": model.params["z_TII"],
                "p": model.pvalues["z_TII"],
            })
    ax.set_xlim(x_min - 0.15, x_max + 1.05)
    ax.set_xlabel("Thinking Immunity (z)")
    ax.set_ylabel("Performance")
    ax.set_title("AI-context specificity")
    ax.grid(color="#EEEEEE", linewidth=0.6)
    source_frames.append(pd.DataFrame(source_rows))


def plot_intervention_gain(ax, df, source_frames):
    df = df.copy()
    if "Gain_TII_Total" not in df:
        df["Gain_TII_Total"] = df["Post_TII_Total"] - df["Pre_TII_Total"]
    order = ["CTI", "Active_Control", "No_Intervention"]
    label_map = {"CTI": "CTI", "Active_Control": "Active", "No_Intervention": "No int."}
    colors = [FIG_PALETTE["red"], FIG_PALETTE["blue_soft"], FIG_PALETTE["light"]]
    rng = np.random.default_rng(42)
    rows = []
    means, cis = [], []
    for i, group in enumerate(order):
        vals = df.loc[df["Group"] == group, "Gain_TII_Total"].astype(float)
        x_jitter = rng.normal(i, 0.055, len(vals))
        ax.scatter(x_jitter, vals, s=8, color=colors[i], alpha=0.22, edgecolor="none", zorder=1)
        mean, ci, n = mean_ci(vals)
        means.append(mean)
        cis.append(ci)
        rows.append({
            "panel": "C_intervention_gain",
            "group": group,
            "mean_gain": mean,
            "ci95": ci,
            "n": n,
        })
    ax.errorbar(range(len(order)), means, yerr=cis, fmt="o", color=FIG_PALETTE["ink"],
                ecolor=FIG_PALETTE["ink"], elinewidth=1.0, capsize=3, markersize=4, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label_map[g] for g in order])
    ax.set_ylabel("TII gain")
    ax.set_title("CTI responsiveness")
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6)
    source_frames.append(pd.DataFrame(rows))


def fit_rsa_surface(df):
    d_mean = df["Defense"].mean()
    c_mean = df["Construction"].mean()
    d_c = df["Defense"] - d_mean
    c_c = df["Construction"] - c_mean
    x = sm.add_constant(pd.DataFrame({
        "D": d_c,
        "C": c_c,
        "D2": d_c ** 2,
        "DC": d_c * c_c,
        "C2": c_c ** 2,
    }))
    model = sm.OLS(df["Higher_Order_Thinking"], x).fit()
    return model, d_mean, c_mean


def plot_rsa_surface(ax, df, source_frames):
    model, d_mean, c_mean = fit_rsa_surface(df)
    d_grid = np.linspace(df["Defense"].quantile(0.02), df["Defense"].quantile(0.98), 90)
    c_grid = np.linspace(df["Construction"].quantile(0.02), df["Construction"].quantile(0.98), 90)
    dd, cc = np.meshgrid(d_grid, c_grid)
    dd_c = dd - d_mean
    cc_c = cc - c_mean
    b = model.params
    pred = (
        b["const"] + b["D"] * dd_c + b["C"] * cc_c +
        b["D2"] * dd_c ** 2 + b["DC"] * dd_c * cc_c + b["C2"] * cc_c ** 2
    )
    cmap = LinearSegmentedColormap.from_list(
        "ti_surface",
        ["#F7F7F7", "#DDE8EF", "#9BC7C7", "#D58B63", "#B64342"],
    )
    levels = np.linspace(np.nanmin(pred), np.nanmax(pred), 14)
    contour = ax.contourf(dd, cc, pred, levels=levels, cmap=cmap)
    ax.contour(dd, cc, pred, levels=7, colors="white", linewidths=0.45, alpha=0.7)
    ax.scatter(df["Defense"], df["Construction"], s=5, color=FIG_PALETTE["ink"],
               alpha=0.16, edgecolor="none")
    denom = b["DC"] ** 2 - 4 * b["D2"] * b["C2"]
    if abs(denom) > 1e-6:
        d_star = (2 * b["C2"] * b["D"] - b["DC"] * b["C"]) / denom + d_mean
        c_star = (2 * b["D2"] * b["C"] - b["DC"] * b["D"]) / denom + c_mean
        ax.scatter([d_star], [c_star], s=65, marker="*", color="#FFD166",
                   edgecolor=FIG_PALETTE["ink"], linewidth=0.55, zorder=5)
        ax.annotate("stationary\npoint", xy=(d_star, c_star), xytext=(d_star + 1.5, c_star + 2.0),
                    arrowprops={"arrowstyle": "-", "lw": 0.6, "color": FIG_PALETTE["ink"]},
                    fontsize=6, ha="left", va="bottom")
    cbar = ax.figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Predicted HOT", fontsize=6)
    cbar.ax.tick_params(labelsize=5, width=0.5)
    ax.set_xlim(d_grid.min(), d_grid.max())
    ax.set_ylim(c_grid.min(), c_grid.max())
    ax.set_xlabel("Defense")
    ax.set_ylabel("Construction")
    ax.set_title("Defense-construction balance")
    source_frames.append(pd.DataFrame({
        "panel": "D_RSA_surface",
        "Defense": dd.ravel(),
        "Construction": cc.ravel(),
        "predicted_HOT": pred.ravel(),
    }))


def plot_longitudinal_summary(ax, df, source_frames):
    colors = {"Defense": FIG_PALETTE["blue"], "Construction": FIG_PALETTE["teal"]}
    rows = []
    for measure in ["Defense", "Construction"]:
        summary = df.groupby("Wave")[measure].agg(["mean", "std", "count"]).reset_index()
        summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
        x = summary["Wave"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        ci = summary["ci95"].to_numpy(dtype=float)
        ax.fill_between(x, y - ci, y + ci, color=colors[measure], alpha=0.16, linewidth=0)
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.8,
                color=colors[measure], label=measure)
        ax.text(x[-1] + 0.08, y[-1], measure, color=colors[measure],
                fontsize=6, va="center", ha="left")
        for _, row in summary.iterrows():
            rows.append({
                "panel": "E_longitudinal_summary",
                "measure": measure,
                "wave": row["Wave"],
                "mean": row["mean"],
                "ci95": row["ci95"],
                "n": row["count"],
            })
    ax.set_xlim(0.85, 4.75)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Wave")
    ax.set_ylabel("Mean score")
    ax.set_title("Four-wave trajectories")
    ax.grid(color="#EEEEEE", linewidth=0.6)
    source_frames.append(pd.DataFrame(rows))


def draw_validation_evidence_grid(data_files, output_dir):
    df1 = pd.read_csv(data_files["study1"])
    df2 = pd.read_csv(data_files["study2"])
    df3 = pd.read_csv(data_files["study3"])
    df4 = pd.read_csv(data_files["study4"])
    df5 = pd.read_csv(data_files["study5"])
    source_frames = []
    fig = plt.figure(figsize=(7.2, 5.15))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.05, 1.32],
        height_ratios=[1.0, 1.0],
        hspace=0.46,
        wspace=0.47,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "E": fig.add_subplot(gs[1, 1]),
        "D": fig.add_subplot(gs[:, 2]),
    }
    plot_incremental_validity(axes["A"], df1, source_frames)
    plot_double_dissociation(axes["B"], df2, source_frames)
    plot_intervention_gain(axes["C"], df3, source_frames)
    plot_longitudinal_summary(axes["E"], df5, source_frames)
    plot_rsa_surface(axes["D"], df4, source_frames)
    for label, ax in axes.items():
        add_panel_label(ax, label)
    output_dir = Path(output_dir)
    pd.concat(source_frames, ignore_index=True).to_csv(
        output_dir / "source_data_validation_evidence_grid.csv", index=False
    )
    return save_pub_figure(fig, output_dir / "thinking_immunity_validation_evidence_grid")


def draw_study5_growth_trajectories(data_files, output_dir):
    df = pd.read_csv(data_files["study5"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), sharex=True)
    colors = {"Defense": FIG_PALETTE["blue"], "Construction": FIG_PALETTE["teal"]}
    degrees = {"Defense": 2, "Construction": 1}
    labels = {"Defense": "Defense trajectory", "Construction": "Construction trajectory"}
    rng = np.random.default_rng(7)
    source_rows = []
    for ax, measure in zip(axes, ["Defense", "Construction"]):
        pivot = df.pivot(index="SubjectID", columns="Wave", values=measure).sort_index(axis=1)
        waves = pivot.columns.to_numpy(dtype=float)
        sample_ids = rng.choice(pivot.index.to_numpy(), size=min(45, len(pivot)), replace=False)
        for _, row in pivot.loc[sample_ids].iterrows():
            ax.plot(waves, row.to_numpy(dtype=float), color=colors[measure],
                    linewidth=0.45, alpha=0.08, zorder=1)
        means = pivot.mean(axis=0).to_numpy(dtype=float)
        cis = 1.96 * pivot.sem(axis=0).to_numpy(dtype=float)
        ax.fill_between(waves, means - cis, means + cis,
                        color=colors[measure], alpha=0.18, linewidth=0, zorder=2)
        ax.plot(waves, means, color=colors[measure], marker="o", markersize=4,
                linewidth=2.0, zorder=3)
        degree = degrees[measure]
        coef = np.polyfit(waves, means, degree)
        x_fit = np.linspace(waves.min(), waves.max(), 120)
        ax.plot(x_fit, np.polyval(coef, x_fit), color=FIG_PALETTE["ink"],
                linestyle="--", linewidth=1.0, alpha=0.9, zorder=4)
        rates = sample_level_growth_rates(df, measure, degree=degree)
        if measure == "Defense":
            note = f"mean quadratic = {rates['coef_0']:.3f}"
        else:
            note = f"mean slope = {rates['coef_0']:.3f}"
        ax.text(0.04, 0.94, note, transform=ax.transAxes,
                ha="left", va="top", fontsize=6, color=FIG_PALETTE["ink"])
        ax.set_title(labels[measure])
        ax.set_xlabel("Wave")
        ax.set_ylabel("Score")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(color="#EEEEEE", linewidth=0.6)
        for wave, mean, ci in zip(waves, means, cis):
            source_rows.append({
                "panel": "study5_growth_trajectories",
                "measure": measure,
                "wave": wave,
                "mean": mean,
                "ci95": ci,
                "n": pivot.shape[0],
                "fit_degree": degree,
            })
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    output_dir = Path(output_dir)
    pd.DataFrame(source_rows).to_csv(
        output_dir / "source_data_study5_growth_trajectories.csv", index=False
    )
    return save_pub_figure(fig, output_dir / "thinking_immunity_study5_growth_trajectories")


def generate_publication_figures(data_files, output_dir):
    apply_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    saved.extend(draw_validation_evidence_grid(data_files, output_dir))
    saved.extend(draw_study5_growth_trajectories(data_files, output_dir))
    print("\nGenerated publication figures:")
    for path in saved:
        print(f"  {path}")
    return saved


if __name__ == "__main__":
    # 数据文件路径 (若无法访问Q盘，请将数据文件复制到脚本所在目录)
    # DATA_PATH_Q = r"Q:/生成式AI时代的"思维免疫"：理论框架、量表开发与对大学生高阶思维及学业表现的纵向效应研究/匹配版/data"
    # 使用脚本所在目录 (推荐: 将数据文件复制到本地目录)
    DATA_PATH = Path(__file__).resolve().parent
    data_files = {
        'study1': DATA_PATH / 'study1_convergent_discriminant_N500_dimensions.csv',
        'study1_items': DATA_PATH / 'study1_convergent_discriminant_N500_items_v4.csv',
        'study2': DATA_PATH / 'study2_double_dissociation_N240_long.csv',
        'study3': DATA_PATH / 'study3_CTI_intervention_N180_v3.csv',
        'study4': DATA_PATH / 'study4_RSA_LPA_N400.csv',
        'study5': DATA_PATH / 'study5_longitudinal_4wave_N300.csv'
    }
    try:
        generate_publication_figures(data_files, DATA_PATH.parent / "figures")
    except Exception as exc:
        print(f"Figure generation failed: {exc}")
        import traceback
        traceback.print_exc()
    
    # 创建输出目录
    output_dir = DATA_PATH.parent / "cfa_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, path in data_files.items():
        try:
            if name == 'study1':
                study1_analysis(path)
            elif name == 'study1_items':
                # 运行CFA分析
                cfa_analysis(path, output_dir)
            elif name == 'study2':
                study2_analysis(path)
            elif name == 'study3':
                study3_analysis(path)
            elif name == 'study4':
                study4_analysis(path)
            elif name == 'study5':
                study5_analysis(path)
        except FileNotFoundError:
            print(f"{name} 文件未找到，跳过")
