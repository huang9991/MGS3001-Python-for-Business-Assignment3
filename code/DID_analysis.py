import random

import pandas as pd
import numpy as np
import statsmodels.api as sm

from typing import Dict, List, Optional, Tuple
import warnings
from data_load_and_analysis import Config

warnings.filterwarnings('ignore')


class DIDAnalyzer:
    """
    双重差分(DID)分析器
    参考：Li et al. (2024) "Generative AI and user voluntary knowledge contribution on Stack Overflow"
    """

    def __init__(self, config: Config):
        self.config = config
        self.chatgpt_release_date = pd.Timestamp(config.CUTOFF_DATE1)

    def _analyze_single_outcome(self,
                                df: pd.DataFrame,
                                treatment_subreddits: List[str],
                                control_subreddits: List[str],
                                outcome: str,
                                outcome_label: str) -> Dict:
        """
        单个结果变量的统一DID分析框架
        """
        print(f"\n{'=' * 50}")
        print(f"分析: {outcome_label}")
        print(f"{'=' * 50}")

        # 1. 数据准备
        df_did = self._prepare_did_data(df, treatment_subreddits, control_subreddits)

        # 2. 描述性统计
        self._did_descriptive_stats(df_did, outcome, outcome_label)

        # 3. 主DID回归
        did_result = self._run_did_regression(df_did, outcome)

        # 4. PSM-DID稳健性检验
        psm_did_result = self._psm_did(df_did, outcome)

        # 5. 平行趋势检验（事件研究法）
        event_study = self._event_study(df_did, outcome)

        # 6. 安慰剂检验
        placebo_result = self._placebo_test(df_did, outcome)

        # 汇总结果
        result = {
            'hypothesis': outcome_label,
            'did_coefficient': did_result['coefficient'],
            'did_pvalue': did_result['pvalue'],
            'did_se': did_result['se'],
            'did_ci_lower': did_result['ci_lower'],
            'did_ci_upper': did_result['ci_upper'],
            'psm_did_coefficient': psm_did_result['coefficient'],
            'psm_did_pvalue': psm_did_result['pvalue'],
            'parallel_trend_passed': event_study['parallel_trend_passed'],
            'placebo_pvalue': placebo_result['pvalue'],
            'r_squared': did_result['r_squared']
        }

        # 打印摘要
        self._print_did_summary(result)
        return result

    def did_analysis_h1_psychoeducation(self, df: pd.DataFrame, treatment_subreddits: List[str],
                                        control_subreddits: List[str]) -> Dict:
        """H1: 心理教育内容比例变化"""
        return self._analyze_single_outcome(df, treatment_subreddits, control_subreddits,
                                            'psychoedu_score', 'H1: 心理教育内容得分')

    def did_analysis_h2_term_density(self, df: pd.DataFrame, treatment_subreddits: List[str],
                                     control_subreddits: List[str]) -> Dict:
        """H2: 心理学术语密度变化"""
        return self._analyze_single_outcome(df, treatment_subreddits, control_subreddits,
                                            'psych_term_density', 'H2: 心理学术语密度')

    def did_analysis_h3_support_score(self, df: pd.DataFrame, treatment_subreddits: List[str],
                                      control_subreddits: List[str]) -> Dict:
        """H3: 情感支持性变化"""
        return self._analyze_single_outcome(df, treatment_subreddits, control_subreddits,
                                            'support_score', 'H3: 情感支持得分')

    def unified_did_analysis(self, df: pd.DataFrame, treatment_subreddits: List[str], control_subreddits: List[str],
                             outcomes: List[str] = None) -> Dict:
        """
        统一DID分析框架
        """
        if outcomes is None:
            outcomes = ['psychoedu_score', 'psych_term_density', 'support_score']

        outcome_labels = {
            'psychoedu_score': 'H1: Psychoeducation score',
            'psych_term_density': 'H2: Psychological term density (Per thousand words)',
            'support_score': 'H3: Support score'
        }

        results = {}
        for outcome in outcomes:
            results[outcome] = self._analyze_single_outcome(
                df, treatment_subreddits, control_subreddits,
                outcome, outcome_labels[outcome]
            )
        return results

    def _prepare_did_data(self,
                          df: pd.DataFrame,
                          treatment_subreddits: List[str],
                          control_subreddits: List[str],
                          pre_window: int = 365,
                          post_window: int = 365) -> pd.DataFrame:
        """
        准备DID分析数据集
        """
        # 深拷贝避免修改原数据
        df_did = df.copy()

        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df_did['date']):
            df_did['date'] = pd.to_datetime(df_did['date'])

        # 创建处理组指示变量
        df_did['treated'] = df_did['subreddit'].isin(treatment_subreddits).astype(int)

        # 创建时间指示变量（ChatGPT发布后=1）
        df_did['post'] = (df_did['date'] >= self.chatgpt_release_date).astype(int)

        # 创建DID交互项
        df_did['did_interact'] = df_did['treated'] * df_did['post']

        # 限制时间窗口
        start_date = self.chatgpt_release_date - pd.Timedelta(days=pre_window)
        end_date = self.chatgpt_release_date + pd.Timedelta(days=post_window)
        df_did = df_did[(df_did['date'] >= start_date) & (df_did['date'] <= end_date)]

        # 创建相对时间变量（以发布日为中心） 将日历时间标准化为"距离事件的时间"
        df_did['relative_time'] = (df_did['date'] - self.chatgpt_release_date).dt.days

        # 创建时间固定效应变量（月份）
        df_did['month'] = df_did['date'].dt.to_period('M')

        # 创建相对周（用于事件研究）
        df_did['relative_week'] = (df_did['relative_time'] / 7).round().astype(int)

        # 删除对照组中的处理组数据（如果存在重叠）
        control_in_treatment = df_did['subreddit'].isin(treatment_subreddits)
        treatment_in_control = df_did['subreddit'].isin(control_subreddits)

        # 只保留明确分组的观测
        df_did = df_did[control_in_treatment | treatment_in_control]

        # 确保treated标签正确
        df_did.loc[df_did['subreddit'].isin(control_subreddits), 'treated'] = 0
        df_did.loc[df_did['subreddit'].isin(treatment_subreddits), 'treated'] = 1

        # 重新计算交互项（确保标签正确）
        df_did['did_interact'] = df_did['treated'] * df_did['post']

        return df_did

    def _run_did_regression(self, df_did: pd.DataFrame, outcome: str) -> Dict:
        """
        运行标准DID回归
        Y = β₀ + β₁*Treated + β₂*Post + β₃*(Treated×Post) + ε
        outcome
        β₃是我们关注的DID估计量
        """
        # 检查是否有足够的观测
        n_treatment_pre = len(df_did[(df_did['treated'] == 1) & (df_did['post'] == 0)])
        n_treatment_post = len(df_did[(df_did['treated'] == 1) & (df_did['post'] == 1)])
        n_control_pre = len(df_did[(df_did['treated'] == 0) & (df_did['post'] == 0)])
        n_control_post = len(df_did[(df_did['treated'] == 0) & (df_did['post'] == 1)])

        if any(n == 0 for n in [n_treatment_pre, n_treatment_post, n_control_pre, n_control_post]):
            raise ValueError(
                f"Insufficient data for DID analysis:\n"
                f"  Treatment Pre: {n_treatment_pre}\n"
                f"  Treatment Post: {n_treatment_post}\n"
                f"  Control Pre: {n_control_pre}\n"
                f"  Control Post: {n_control_post}"
            )

        # 基础DID模型
        X = df_did[['treated', 'post', 'did_interact']]
        X = sm.add_constant(X)
        y = df_did[outcome]

        model = sm.OLS(y, X).fit(cov_type='HC1')  # 异方差稳健标准误

        # 提取结果
        coef = model.params['did_interact']
        se = model.bse['did_interact']
        pvalue = model.pvalues['did_interact']
        ci = model.conf_int().loc['did_interact']

        # 计算DID的直观解释 treated 是处理组标识，指示一个观测是否属于"受到ChatGPT影响"的组。即 心理健康相关子版块
        mean_before_treat = df_did[(df_did['treated'] == 1) & (df_did['post'] == 0)][outcome].mean()
        mean_after_treat = df_did[(df_did['treated'] == 1) & (df_did['post'] == 1)][outcome].mean()
        mean_before_control = df_did[(df_did['treated'] == 0) & (df_did['post'] == 0)][outcome].mean()
        mean_after_control = df_did[(df_did['treated'] == 0) & (df_did['post'] == 1)][outcome].mean()

        print(f"\nDID回归结果 (结果变量: {outcome}):")
        print(f"  处理组变化: {mean_after_treat - mean_before_treat:+.6f}")
        print(f"  对照组变化: {mean_after_control - mean_before_control:+.6f}")
        print(f"  DID估计量: {coef:+.6f}")
        print(f"  标准误: {se:.6f}")
        print(f"  p值: {pvalue:.6f}")
        print(f"  95% CI: [{ci[0]:.6}, {ci[1]:.6f}]")
        print(f"  R²: {model.rsquared:.6f}")

        return {
            'coefficient': coef,
            'se': se,
            'pvalue': pvalue,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'r_squared': model.rsquared,
            'model': model,
            'sample_sizes': {
                'treatment_pre': n_treatment_pre,
                'treatment_post': n_treatment_post,
                'control_pre': n_control_pre,
                'control_post': n_control_post
            }
        }

    def _run_did_with_controls(self,
                               df_did: pd.DataFrame,
                               outcome: str,
                               controls: List[str]) -> Dict:
        """
        带控制变量的DID回归
        Y = β₀ + β₁*Treated + β₂*Post + β₃*(Treated×Post) + γ*X + ε
        """
        # 筛选存在的控制变量
        available_controls = [c for c in controls if c in df_did.columns]

        if not available_controls:
            return {'coefficient': np.nan, 'pvalue': np.nan, 'adj_r_squared': np.nan}

        # 创建设计矩阵
        X_vars = ['treated', 'post', 'did_interact'] + available_controls
        X = df_did[X_vars].dropna()
        X = sm.add_constant(X)

        # 确保y与X对齐
        y = df_did.loc[X.index, outcome]

        model = sm.OLS(y, X).fit(cov_type='HC1')

        coef = model.params['did_interact']
        pvalue = model.pvalues['did_interact']

        print(f"\n带控制变量的DID结果:")
        print(f"  控制变量: {available_controls}")
        print(f"  DID系数: {coef:+.6f} (p={pvalue:.6f})")
        print(f"  Adj R²: {model.rsquared_adj:.6f}")

        return {
            'coefficient': coef,
            'pvalue': pvalue,
            'adj_r_squared': model.rsquared_adj,
            'controls': available_controls
        }

    def _event_study(self, df_did: pd.DataFrame, outcome: str,
                     n_lags: int = 8, n_leads: int = 8) -> Dict:
        """
        事件研究法检验平行趋势

        模型: Y = Σβₖ*Dₖ + γTreated + δControls + ε
        其中Dₖ是相对时间的虚拟变量
        """
        # 创建相对时间虚拟变量
        df_es = df_did.copy()

        # 生成时间窗口
        time_windows = list(range(-n_lags, n_leads + 1))
        time_windows.remove(-1)  # 基准期设为-1期（发布前一期）

        for t in time_windows:
            col_name = f'time_{t}' if t >= 0 else f'time_m{abs(t)}'
            # 创建处理组×时间交互项
            df_es[col_name] = (df_es['treated'] == 1) & (df_es['relative_week'] == t)
            df_es[col_name] = df_es[col_name].astype(int)

        # 删除基准期
        time_cols = [f'time_{t}' if t >= 0 else f'time_m{abs(t)}' for t in time_windows]

        # 回归
        X = df_es[time_cols + ['treated']]
        X = sm.add_constant(X)
        y = df_es[outcome]

        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')

            # 提取处理前各期的系数
            pre_treatment_coefs = {}
            for t in range(-n_lags, 0):
                if t != -1:  # 排除基准期
                    col_name = f'time_m{abs(t)}'
                    pre_treatment_coefs[t] = model.params.get(col_name, np.nan)

            # 简单平行趋势检验：处理前系数是否联合显著
            pre_cols = [c for c in time_cols if 'm' in c]
            pre_coefs = [model.params.get(c, 0) for c in pre_cols if c in model.params.index]

            # 判断平行趋势（视觉检查 + 联合检验）
            parallel_trend_passed = True
            if len(pre_coefs) > 1:
                # 检查系数是否接近0
                avg_pre_coef = np.mean(np.abs(pre_coefs))
                if avg_pre_coef > 0.1 * abs(model.params.get('did_interact', 0)):
                    parallel_trend_passed = False

            return {
                'parallel_trend_passed': parallel_trend_passed,
                'coefficients': pre_treatment_coefs,
                'model': model
            }

        except Exception as e:
            print(f"事件研究估计失败: {e}")
            return {
                'parallel_trend_passed': None,
                'coefficients': {},
                'error': str(e)
            }

    def _placebo_test(self, df_did: pd.DataFrame, outcome: str) -> Dict:
        """
        安慰剂检验：假设处理发生在实际时间之前
        """
        # 将"伪处理时间"设为实际处理前180天
        placebo_date = self.chatgpt_release_date - pd.Timedelta(days=180)

        df_placebo = df_did[
            df_did['date'] < self.chatgpt_release_date
            ].copy()

        if len(df_placebo) < 100:
            return {'coefficient': np.nan, 'pvalue': np.nan, 'passed': None}

        # 创建伪时间变量
        df_placebo['post_placebo'] = (df_placebo['date'] >= placebo_date).astype(int)
        df_placebo['did_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

        # 回归
        X = df_placebo[['treated', 'post_placebo', 'did_placebo']]
        X = sm.add_constant(X)
        y = df_placebo[outcome]

        model = sm.OLS(y, X).fit()

        coef = model.params['did_placebo']
        pvalue = model.pvalues['did_placebo']

        # 安慰剂检验通过的标准：伪处理效应不显著
        passed = pvalue > 0.05

        print(f"\n安慰剂检验 (伪处理时间: {placebo_date.date()}):")
        print(f"  伪DID系数: {coef:+.6f} (p={pvalue:.6f})")
        print(f"  检验结果: {'通过' if passed else '未通过'} (p值应>0.05)")

        return {
            'coefficient': coef,
            'pvalue': pvalue,
            'passed': passed,
            'placebo_date': placebo_date
        }

    def _quantile_did(self, df_did: pd.DataFrame, outcome: str,
                      quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict:
        """
        分位数DID：检查处理效应在不同分布位置的变化
        """
        import statsmodels.formula.api as smf

        results = {}

        for q in quantiles:
            try:
                model = smf.quantreg(
                    f'{outcome} ~ treated + post + did_interact',
                    data=df_did
                ).fit(q=q)

                coef = model.params['did_interact']
                pvalue = model.pvalues['did_interact']

                results[f'q{int(q * 100)}'] = {
                    'coefficient': coef,
                    'pvalue': pvalue
                }

                print(f"  分位数 {q:.2f}: DID系数 = {coef:+.6f} (p={pvalue:.6f})")

            except Exception as e:
                print(f"  分位数 {q:.2f}: 估计失败 - {e}")
                results[f'q{int(q * 100)}'] = {'coefficient': np.nan, 'pvalue': np.nan}

        return results

    def _psm_did(self, df_did: pd.DataFrame, outcome: str) -> Dict:
        """
        PSM-DID：先进行倾向得分匹配，再进行DID
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        # 计算倾向得分
        features = ['text_length']
        if 'score' in df_did.columns:
            features.append('score')
        if 'num_comments' in df_did.columns:
            features.append('num_comments')

        X_ps = df_did[features].fillna(df_did[features].mean())
        y_ps = df_did['treated']

        try:
            # 逻辑回归估计倾向得分
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_ps, y_ps)
            propensity_scores = lr.predict_proba(X_ps)[:, 1]

            df_psm = df_did.copy()
            df_psm['pscore'] = propensity_scores

            # 最近邻匹配（1:1）
            treated = df_psm[df_psm['treated'] == 1]
            control = df_psm[df_psm['treated'] == 0]

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control[['pscore']])
            distances, indices = nn.kneighbors(treated[['pscore']])

            # 构建匹配后的数据集
            matched_control = control.iloc[indices.flatten()]
            df_matched = pd.concat([treated, matched_control])

            # 在匹配样本上进行DID
            did_result = self._run_did_regression(df_matched, outcome)

            return {
                'coefficient': did_result['coefficient'],
                'pvalue': did_result['pvalue'],
                'n_matched': len(df_matched)
            }

        except Exception as e:
            print(f"PSM-DID失败: {e}")
            return {'coefficient': np.nan, 'pvalue': np.nan, 'n_matched': 0}

    def _alternative_control_test(self,
                                  df: pd.DataFrame,
                                  treatment_subreddits: List[str],
                                  control_subreddits: List[str],
                                  outcome: str) -> Dict:
        """
        替代对照组稳健性检验：随机抽取多个对照组子集
        """
        if len(control_subreddits) < 3:
            return {'robust': None}

        n_samples = min(10, len(control_subreddits))
        results = []

        for i in range(n_samples):
            # 随机抽取一半对照组
            sample_control = np.random.choice(
                control_subreddits,
                size=len(control_subreddits) // 2,
                replace=False
            ).tolist()

            df_sample = self._prepare_did_data(
                df, treatment_subreddits, sample_control
            )

            try:
                did_result = self._run_did_regression(df_sample, outcome)
                results.append({
                    'coefficient': did_result['coefficient'],
                    'pvalue': did_result['pvalue']
                })
            except:
                continue

        if results:
            coefs = [r['coefficient'] for r in results]
            pvalues = [r['pvalue'] for r in results]

            return {
                'robust': True,
                'mean_coefficient': np.mean(coefs),
                'std_coefficient': np.std(coefs),
                'significant_ratio': np.mean([p < 0.05 for p in pvalues]),
                'n_tests': len(results)
            }

        return {'robust': None}

    def _did_descriptive_stats(self, df_did: pd.DataFrame, outcome: str, label: str):
        """DID描述性统计"""
        # 计算四组的平均值
        stats_table = df_did.groupby(['treated', 'post'])[outcome].agg(['mean', 'std', 'count']).round(4)

        print(f"\nDID描述性统计 ({label}):")
        print(f"{'=' * 50}")
        print(f"{'组别':<15} {'平均值':<10} {'标准差':<10} {'样本量':<10}")
        print(f"{'-' * 50}")

        group_labels = {
            (0, 0): '对照组-处理前',
            (0, 1): '对照组-处理后',
            (1, 0): '处理组-处理前',
            (1, 1): '处理组-处理后'
        }

        for (treated, post), row in stats_table.iterrows():
            group_name = group_labels.get((treated, post), f'T={treated},P={post}')
            print(f"{group_name:<15} {row['mean']:<10.6f} {row['std']:<10.6f} {int(row['count']):<10}")

        # 计算原始DID
        if all((t, p) in stats_table.index for t in [0, 1] for p in [0, 1]):
            treat_change = (stats_table.loc[(1, 1), 'mean'] -
                            stats_table.loc[(1, 0), 'mean'])
            control_change = (stats_table.loc[(0, 1), 'mean'] -
                              stats_table.loc[(0, 0), 'mean'])
            raw_did = treat_change - control_change

            print(f"{'-' * 50}")
            print(f"处理组变化: {treat_change:+.6f}")
            print(f"对照组变化: {control_change:+.6f}")
            print(f"原始DID: {raw_did:+.6f}")

    def _print_did_summary(self, result: Dict):
        """打印DID结果摘要"""
        print(f"\n{'=' * 50}")
        print(f"DID结果摘要: {result['hypothesis']}")
        print(f"{'=' * 50}")
        print(f"DID系数: {result['did_coefficient']:+.6f}")
        print(f"标准误: {result['did_se']:.6f}")
        print(f"95% CI: [{result['did_ci_lower']:.6f}, {result['did_ci_upper']:.6f}]")
        print(f"p值: {result['did_pvalue']:.6f}")
        print(
            f"显著性: {'***' if result['did_pvalue'] < 0.001 else '**' if result['did_pvalue'] < 0.01 else '*' if result['did_pvalue'] < 0.05 else 'n.s.'}")
        print(f"平行趋势: {'通过' if result['parallel_trend_passed'] else '未通过/未检验'}")
        print(f"R²: {result['r_squared']:.6f}")
        if 'adj_r_squared' in result:
            print(f"Adj R²: {result['adj_r_squared']:.6f}")


# did_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path

class DIDVisualizer:
    """DID分析可视化（核心功能）"""

    def __init__(self, config):
        self.config = config
        self.chatgpt_release_date = pd.Timestamp(config.CUTOFF_DATE1)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_event_study(self, df, outcome, treatment_subs, control_subs,
                         n_lags=6, n_leads=6, save_path=None):
        """事件研究图：展示处理前后各期的DID系数和置信区间
        核心作用：验证平行趋势假设 (Parallel Trend Assumption)

        平行趋势假设是 DID 分析的灵魂。它要求：在政策发生之前，处理组和对照组的结果变量应该沿着相同的趋势发展。
        """
        df_es = self._prepare_data(df, treatment_subs, control_subs, outcome)

        print("数据形状:", df_es.shape)

        coef_dict = self._calculate_event_coefs(df_es, outcome, n_lags, n_leads)
        if not coef_dict:
            print("事件研究数据不足")
            return

        # 提取数据
        times = sorted(coef_dict.keys())
        coefs = [coef_dict[t]['coef'] for t in times]
        ci_low = [coef_dict[t]['ci_lower'] for t in times]
        ci_high = [coef_dict[t]['ci_upper'] for t in times]

        fig, ax = plt.subplots(figsize=(12, 6))

        # 系数和置信区间
        ax.plot(times, coefs, 'b-', linewidth=2, marker='o', markersize=5)
        ax.fill_between(times, ci_low, ci_high, alpha=0.2, color='blue')

        # 参考线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, label='ChatGPT release')

        # 标注
        ax.set_xlabel('Relative time(week)', fontdict={'weight': 'bold', 'size': 14})
        ax.set_ylabel('Estimated coefficient', fontdict={'weight': 'bold', 'size': 14})
        ax.set_title(f'Event study: {self._label(outcome)}', fontdict={'weight': 'bold', 'size': 16})
        ax.legend()

        # 平行趋势判断
        pre_coefs = [coef_dict[t]['pvalue'] for t in times if t < 0]
        passed = all(p > 0.05 for p in pre_coefs) if pre_coefs else None
        status = 'Parallel trend through' if passed else 'Parallel trend did not pass'
        ax.text(0.02, 0.98, status, transform=ax.transAxes,
                va='top', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()

    def plot_did_trends(self, df, outcome, treatment_subs, control_subs, save_path=None):
        """DID趋势图：处理组和对照组的时间趋势对比"""
        df_did = self._prepare_data(df, treatment_subs, control_subs, outcome)

        # 周度聚合
        df_did['year_week'] = df_did['date'].dt.to_period('W')
        treated_w = df_did[df_did['treated'] == 1].groupby('year_week')[outcome].agg(['mean', 'std', 'count'])
        control_w = df_did[df_did['treated'] == 0].groupby('year_week')[outcome].agg(['mean', 'std', 'count'])

        treated_w['date'] = treated_w.index.start_time
        control_w['date'] = control_w.index.start_time

        fig, ax = plt.subplots(figsize=(14, 6))

        # 处理组
        ax.plot(treated_w['date'], treated_w['mean'], 'b-', linewidth=2, label='Treatment group(mental health)')
        se = treated_w['std'] / np.sqrt(treated_w['count'])
        ax.fill_between(treated_w['date'], treated_w['mean'] - 1.96 * se,
                        treated_w['mean'] + 1.96 * se, alpha=0.15, color='blue')

        # 对照组
        ax.plot(control_w['date'], control_w['mean'], 'orange', linewidth=2, label='Control group')
        se = control_w['std'] / np.sqrt(control_w['count'])
        ax.fill_between(control_w['date'], control_w['mean'] - 1.96 * se,
                        control_w['mean'] + 1.96 * se, alpha=0.15, color='orange')

        # 断点
        ax.axvline(self.chatgpt_release_date, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='ChatGPT release')

        ax.set_xlabel('Date')
        ax.set_ylabel(self._label(outcome))
        ax.set_title('DID trend analysis: Treatment group vs Control group')
        ax.legend(loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()

    def plot_placebo_test(self, df, outcome, treatment_subs, control_subs,
                          n_placebos=100, save_path=None):
        """安慰剂检验：随机生成假处理时间，检验真实效应的显著性
        跟上面的self._placebo_test区别在于 这里的placebo_effects是通过多次（n_placebos）随机生成一些假处理时间点，然后绘制分布，查看真实效应是否处于placebo_effects分布的边缘
        来判断是否通过安慰剂检验

        """
        # 准备数据
        df_did = DIDAnalyzer(config=self.config)._prepare_did_data(df, treatment_subs, control_subs)
        print(df_did.columns.values)
        # 获取真实效应（通过回归）
        did_result = DIDAnalyzer(config=self.config)._run_did_regression(df_did, outcome)
        true_effect = did_result['coefficient']
        # true_effect = self._calc_true_effect(df, treatment_subs, control_subs, outcome)
        placebo_effects = self._gen_placebo_effects(df, treatment_subs, control_subs,
                                                    outcome, n_placebos)

        fig, ax = plt.subplots(figsize=(10, 5))

        # 安慰剂分布
        ax.hist(placebo_effects, bins=30, alpha=0.7, color='gray', edgecolor='black', density=True)

        # 真实效应
        ax.axvline(true_effect, color='red', linestyle='--', linewidth=2,
                   label=f'True effect: {true_effect:.6f}')

        # p值
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(true_effect))
        ax.text(0.95, 0.95, f'p = {p_value:.6f}\n Placebo nums = {n_placebos}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('DID estimated value')
        ax.set_ylabel('Density')
        ax.set_title(f'Placebo test: {self._label(outcome)}')
        ax.legend(loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()

    def plot_method_comparison(self, did_results, trad_results, save_path=None):
        """方法对比图：DID vs 传统方法"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        hypotheses = ['H1', 'H2', 'H3']

        # 提取效应值
        did_effects = []
        trad_effects = []
        for h in hypotheses:
            d = did_results.get(h, {}).get('did_coefficient', 0)
            t = trad_results.get(h, {})
            if h == 'H1':
                trad_effects.append(t.get('change', 0) / 100)
            elif h == 'H2':
                trad_effects.append(t.get('change', 0))
            else:
                trad_effects.append(t.get('change', 0))
            did_effects.append(d)

        # 子图1：效应大小对比
        x = np.arange(3)
        width = 0.35
        axes[0].bar(x - width / 2, did_effects, width, label='DID', color='#2E86AB', alpha=0.8)
        axes[0].bar(x + width / 2, trad_effects, width, label='Traditional method', color='#F18F01', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(hypotheses)
        axes[0].set_ylabel('Effect size')
        axes[0].set_title('Effect size comparison')
        axes[0].axhline(y=0, color='black', linewidth=0.5)
        axes[0].legend()

        # 子图2：显著性对比
        did_pvals = [did_results.get(h, {}).get('did_pvalue', 1) for h in hypotheses]
        trad_pvals = []
        for h in hypotheses:
            t = trad_results.get(h, {})
            if h == 'H1':
                trad_pvals.append(t.get('chi2_pvalue', 1))
            elif h == 'H2':
                trad_pvals.append(t.get('t_pvalue', 1))
            else:
                trad_pvals.append(t.get('mannwhitney_pvalue', 1))

        did_logp = [-np.log10(p) if p > 0 else 10 for p in did_pvals]
        trad_logp = [-np.log10(p) if p > 0 else 10 for p in trad_pvals]

        axes[1].bar(x - width / 2, did_logp, width, label='DID', color='#2E86AB', alpha=0.8)
        axes[1].bar(x + width / 2, trad_logp, width, label='Traditional method', color='#F18F01', alpha=0.8)
        axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(hypotheses)
        axes[1].set_ylabel('-log10(p value)')
        axes[1].set_title('Significance comparison (higher values indicate greater significance)')
        axes[1].legend()

        plt.suptitle('DID vs Traditional method', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()

    # ===== 辅助方法 =====
    def _prepare_data(self, df, treatment_subs, control_subs, outcome):
        """准备数据"""
        df_out = df[df['subreddit'].isin(treatment_subs + control_subs)].copy()
        if not pd.api.types.is_datetime64_any_dtype(df_out['date']):
            df_out['date'] = pd.to_datetime(df_out['date'])

        df_out['treated'] = df_out['subreddit'].isin(treatment_subs).astype(int)
        df_out['relative_week'] = ((df_out['date'] - self.chatgpt_release_date).dt.days / 7).round().astype(int)
        df_out['post'] = (df_out['date'] >= self.chatgpt_release_date).astype(int)
        return df_out

    def _calculate_event_coefs(self, df_es, outcome, n_lags, n_leads):
        """计算事件研究系数"""
        import statsmodels.formula.api as smf

        time_range = [t for t in range(-n_lags, n_leads + 1) if t != -1] # 创建时间范围

        for t in time_range:
            col = f'time_{t}' if t >= 0 else f'time_m{abs(t)}' # m 标识 负数， 之前的时间
            df_es[col] = ((df_es['treated'] == 1) & (df_es['relative_week'] == t)).astype(int) # 构建交互项，交互项系数表示：处理组在该时间点相对于基准期（t=-1）的差异

        time_cols = [f'time_{t}' if t >= 0 else f'time_m{abs(t)}' for t in time_range]

        formula = f"{outcome} ~ {' + '.join(time_cols)} + treated"
        model = smf.ols(formula, data=df_es).fit()

        result = {}
        for col in time_cols:
            if col in model.params.index:
                if 'time_m' in col:
                    t = -int(col.replace('time_m', ''))
                else:
                    t = int(col.replace('time_', ''))
                result[t] = {
                    'coef': model.params[col],
                    'ci_lower': model.conf_int().loc[col, 0],
                    'ci_upper': model.conf_int().loc[col, 1],
                    'pvalue': model.pvalues[col]
                }
        return result
        # except:
        #     return None

    def _calc_true_effect(self, df, treatment_subs, control_subs, outcome):
        """计算真实DID效应"""
        df_did = self._prepare_data(df, treatment_subs, control_subs, outcome)

        means = df_did.groupby(['treated', 'post'])[outcome].mean()
        tp = means.get((1, 1), 0) - means.get((1, 0), 0)
        cp = means.get((0, 1), 0) - means.get((0, 0), 0)
        return tp - cp

    def _gen_placebo_effects(self, df, treatment_subs, control_subs, outcome, n_placebos):
        """生成安慰剂效应"""
        import statsmodels.api as sm

        effects = []
        for _ in range(n_placebos):
            if random.random() < 0.5:
                days = np.random.randint(-240, -90)
            else:
                days = np.random.randint(90, 240)

            placebo_date = self.chatgpt_release_date + pd.Timedelta(days=days)

            df_p = df[df['subreddit'].isin(treatment_subs + control_subs)].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_p['date']):
                df_p['date'] = pd.to_datetime(df_p['date'])

            df_p['post'] = (df_p['date'] >= placebo_date).astype(int)
            df_p['treated'] = df_p['subreddit'].isin(treatment_subs).astype(int)
            df_p['did'] = df_p['treated'] * df_p['post']

            try:
                X = sm.add_constant(df_p[['treated', 'post', 'did']])
                model = sm.OLS(df_p[outcome], X).fit()
                effects.append(model.params['did'])
            except:
                continue

        return np.array(effects) if effects else np.array([0])

    def _label(self, outcome):
        """标签映射"""
        return {
            'psychoedu_score': 'Psychoeducation score',
            'psych_term_density': 'Psychological term density (Per thousand words)',
            'support_score': 'Support score'
        }.get(outcome, outcome)

    def save_all(self, df, treatment_subs, control_subs, output_dir='./figure'):
        """一键保存所有核心图表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for outcome in ['psychoedu_score', 'psych_term_density', 'support_score']:
            if outcome in df.columns:
                self.plot_event_study(df, outcome, treatment_subs, control_subs,
                                      save_path=output_path / f'event_{outcome}.png')
                self.plot_did_trends(df, outcome, treatment_subs, control_subs,
                                     save_path=output_path / f'trends_{outcome}.png')