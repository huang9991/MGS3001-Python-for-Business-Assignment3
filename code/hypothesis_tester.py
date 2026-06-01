# hypothesis_tester.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import timedelta
from config import Config
warnings.filterwarnings('ignore')


class HypothesisTester:
    """验证三个假设 - 集成DID、RDD、T检验等方法框架"""

    def __init__(self, config: Config):
        self.config = config
        self.did_analyzer = None  # 延迟初始化DID分析器

    def _init_did_analyzer(self):
        """延迟初始化DID分析器，避免循环导入"""
        if self.did_analyzer is None:
            from DID_analysis import DIDAnalyzer
            self.did_analyzer = DIDAnalyzer(self.config)
        return self.did_analyzer

    # ==================== 统一的假设检验入口 ====================

    def test_all_hypotheses(self,
                            df: pd.DataFrame,
                            method: str = 'did',
                            treatment_subreddits: Optional[List[str]] = None,
                            control_subreddits: Optional[List[str]] = None,
                            run_all_methods: bool = False) -> Dict:
        """
        统一的假设检验入口

        Args:
            df: 包含所有数据的DataFrame
            method: 主分析方法 - 'did' 或 'traditional'
            treatment_subreddits: 处理组子版块列表
            control_subreddits: 对照组子版块列表
            run_all_methods: 是否运行所有方法进行比较

        Returns:
            包含所有假设检验结果的字典
        """
        results = {
            'method_used': method,
            'hypotheses': {},
            'comparison': {} if run_all_methods else None
        }

        if method == 'did':
            print("\n" + "=" * 80)
            print("使用 DID (双重差分) 方法进行假设检验")
            print("=" * 80)

            # 使用提供的子版块或默认配置
            treatment_subs = treatment_subreddits or self.config.TREATMENT_SUBREDDITS
            control_subs = control_subreddits or self.config.CONTROL_SUBREDDITS

            did = self._init_did_analyzer()

            # 统一DID分析
            did_results = did.unified_did_analysis(
                df, treatment_subs, control_subs
            )

            # 格式化DID结果
            for outcome, result in did_results.items():
                h_num = self._get_hypothesis_number(outcome)
                results['hypotheses'][h_num] = {
                    'method': 'DID',
                    'hypothesis': result['hypothesis'],
                    'coefficient': result['did_coefficient'],
                    'se': result['did_se'],
                    'pvalue': result['did_pvalue'],
                    'ci_lower': result['did_ci_lower'],
                    'ci_upper': result['did_ci_upper'],
                    'significant': result['did_pvalue'] < 0.05,
                    'parallel_trend_passed': result['parallel_trend_passed'],
                    'r_squared': result['r_squared']
                }

            # 运行所有方法进行比较
            if run_all_methods:
                results['comparison']['traditional'] = self._run_traditional_methods(df)
                results['comparison']['did'] = results['hypotheses'].copy()

        elif method == 'traditional':
            print("\n" + "=" * 80)
            print("使用传统方法进行假设检验")
            print("=" * 80)

            results['hypotheses'] = self._run_traditional_methods(df)

            # 也运行DID作为比较
            if run_all_methods:
                treatment_subs = treatment_subreddits or self.config.TREATMENT_SUBREDDITS
                control_subs = control_subreddits or self.config.CONTROL_SUBREDDITS
                did = self._init_did_analyzer()

                did_results = did.unified_did_analysis(
                    df, treatment_subs, control_subs
                )

                did_formatted = {}
                for outcome, result in did_results.items():
                    h_num = self._get_hypothesis_number(outcome)
                    did_formatted[h_num] = {
                        'method': 'DID',
                        'hypothesis': result['hypothesis'],
                        'coefficient': result['did_coefficient'],
                        'pvalue': result['did_pvalue'],
                        'significant': result['did_pvalue'] < 0.05,
                        'parallel_trend_passed': result['parallel_trend_passed']
                    }

                results['comparison']['did'] = did_formatted

        # 打印对比结果
        if run_all_methods:
            self._print_method_comparison(results['comparison'])

        return results

    def _run_traditional_methods(self, df: pd.DataFrame) -> Dict:
        """运行传统的三个假设检验方法"""
        results = {}

        results['H1'] = self.test_h1_psychoeducation_proportion(
            df, is_considering_effect_time=True
        )

        results['H2'] = self.test_h2_psych_term_density(df)

        results['H3'] = self.test_h3_support_score(df)

        return results

    def _get_hypothesis_number(self, outcome: str) -> str:
        """根据结果变量名获取假设编号"""
        mapping = {
            'is_psychoeducational': 'H1',
            'psychoedu_score': 'H1',
            'psych_term_density': 'H2',
            'support_score': 'H3'
        }
        return mapping.get(outcome, 'Unknown')

    def _print_method_comparison(self, comparison: Dict):
        """打印方法对比结果"""
        print("\n" + "=" * 80)
        print("方法对比结果")
        print("=" * 80)

        print(f"\n{'假设':<6} {'传统方法':<20} {'DID方法':<20} {'结论一致':<10}")
        print("-" * 60)

        for h in ['H1', 'H2', 'H3']:
            trad_result = comparison['traditional'].get(h, {})
            did_result = comparison['did'].get(h, {})

            trad_sig = '显著***' if trad_result.get('significant') else '不显著'
            did_sig = '显著***' if did_result.get('significant') else '不显著'
            consistent = '✓' if (trad_result.get('significant') == did_result.get('significant')) else '✗'

            print(f"{h:<6} {trad_sig:<20} {did_sig:<20} {consistent:<10}")

        print("-" * 60)
        print("注: 推荐以DID结果为主要结论")

    # ==================== 原始假设检验方法（保持向后兼容） ====================

    def test_h1_psychoeducation_proportion(self,
                                           df: pd.DataFrame,
                                           is_considering_effect_time: bool = True,
                                           use_did: bool = False,
                                           treatment_subreddits: Optional[List[str]] = None,
                                           control_subreddits: Optional[List[str]] = None) -> Dict:
        """检验H1：心理教育内容比例变化"""
        print("\n" + "=" * 60)
        print("H1: 心理教育内容比例变化分析")
        print("=" * 60)

        if use_did and treatment_subreddits and control_subreddits:
            print("  使用DID方法分析...")
            return self._h1_did_analysis(df, treatment_subreddits, control_subreddits)

        print("  使用传统方法分析...")

        before = df[df['period'] == 'before']
        after = df[df['period'] == 'after']

        prop_before = before['psychoedu_score'].mean()
        prop_after = after['psychoedu_score'].mean()

        print(f"\n心理教育内容得分:")
        print(f"  ChatGPT发布前: {prop_before:.4f}")
        print(f"  ChatGPT发布后: {prop_after:.4f}")
        print(f"  变化: {prop_after - prop_before:+.4f}")

        # 1. 卡方检验（基于中位数分组）
        combined = pd.concat([before['psychoedu_score'], after['psychoedu_score']])
        median = combined.median()
        before_high = (before['psychoedu_score'] > median).sum()
        before_low = (before['psychoedu_score'] <= median).sum()
        after_high = (after['psychoedu_score'] > median).sum()
        after_low = (after['psychoedu_score'] <= median).sum()
        contingency = np.array([[before_high, before_low], [after_high, after_low]])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

        # 2. t检验
        t_stat, t_p = stats.ttest_ind(after['psychoedu_score'], before['psychoedu_score'], equal_var=False)

        # 3. Mann-Whitney U检验
        u_stat, mw_p = stats.mannwhitneyu(after['psychoedu_score'], before['psychoedu_score'], alternative='two-sided')

        # 4. RDD分析
        rdd_result = self._rdd_analysis(df, 'days_from_cutoff1', outcome='psychoedu_score', running_var2='days_from_cutoff2')

        # Cohen's d
        cohens_d = self._cohens_d(after['psychoedu_score'], before['psychoedu_score'])

        return {
            'hypothesis': 'H1',
            'method': 'Chi-square + t-test + Mann-Whitney U + RDD',
            'before_prop': prop_before,
            'after_prop': prop_after,
            'change': prop_after - prop_before,
            'chi2_pvalue': chi2_p,
            't_pvalue': t_p,
            'mannwhitney_pvalue': mw_p,
            'cohens_d': cohens_d,
            'significant': any(p < 0.05 for p in [chi2_p, t_p, mw_p]),
            'rdd_coefficient': rdd_result['coefficient'],
            'rdd_pvalue': rdd_result['pvalue']
        }

    def test_h2_psych_term_density(self,
                                   df: pd.DataFrame,
                                   use_did: bool = False,
                                   treatment_subreddits: Optional[List[str]] = None,
                                   control_subreddits: Optional[List[str]] = None) -> Dict:
        """检验H2：心理学术语密度变化"""
        print("\n" + "=" * 60)
        print("H2: 心理学术语密度变化分析")
        print("=" * 60)

        if use_did and treatment_subreddits and control_subreddits:
            print("  使用DID方法分析...")
            return self._h2_did_analysis(df, treatment_subreddits, control_subreddits)

        print("  使用传统方法分析...")

        before = df[df['period'] == 'before']
        after = df[df['period'] == 'after']

        density_before = before['psych_term_density'].mean()
        density_after = after['psych_term_density'].mean()

        print(f"\n心理学术语密度（每千字）:")
        print(f"  ChatGPT发布前: {density_before:.4f}")
        print(f"  ChatGPT发布后: {density_after:.4f}")
        print(f"  变化: {density_after - density_before:+.4f}")

        # 1. 卡方检验（基于中位数分组）
        combined = pd.concat([before['psych_term_density'], after['psych_term_density']])
        median = combined.median()
        before_high = (before['psych_term_density'] > median).sum()
        before_low = (before['psych_term_density'] <= median).sum()
        after_high = (after['psych_term_density'] > median).sum()
        after_low = (after['psych_term_density'] <= median).sum()
        contingency = np.array([[before_high, before_low], [after_high, after_low]])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

        # 2. t检验
        t_stat, t_p = stats.ttest_ind(after['psych_term_density'], before['psych_term_density'], equal_var=False)

        # 3. Mann-Whitney U检验
        u_stat, mw_p = stats.mannwhitneyu(after['psych_term_density'], before['psych_term_density'],
                                          alternative='two-sided')

        # 4. RDD分析
        rdd_result = self._rdd_analysis(df, 'days_from_cutoff1', 'psych_term_density', running_var2='days_from_cutoff2')

        # Cohen's d
        cohens_d = self._cohens_d(after['psych_term_density'], before['psych_term_density'])

        # 分位数变化（可选保留）
        quantile_changes = {}
        for q in [0.25, 0.5, 0.75, 0.9]:
            quantile_changes[f'q{int(q * 100)}'] = {
                'before': before['psych_term_density'].quantile(q),
                'after': after['psych_term_density'].quantile(q),
                'change': after['psych_term_density'].quantile(q) - before['psych_term_density'].quantile(q)
            }

        return {
            'hypothesis': 'H2',
            'method': 'Chi-square + t-test + Mann-Whitney U + RDD',
            'before_density': density_before,
            'after_density': density_after,
            'change': density_after - density_before,
            'chi2_pvalue': chi2_p,
            't_pvalue': t_p,
            'mannwhitney_pvalue': mw_p,
            'cohens_d': cohens_d,
            'significant': any(p < 0.05 for p in [chi2_p, t_p, mw_p]),
            'rdd_coefficient': rdd_result['coefficient'],
            'rdd_pvalue': rdd_result['pvalue'],
            'quantile_changes': quantile_changes
        }

    def test_h3_support_score(self,
                              df: pd.DataFrame,
                              use_did: bool = False,
                              treatment_subreddits: Optional[List[str]] = None,
                              control_subreddits: Optional[List[str]] = None) -> Dict:
        """检验H3：情感支持性变化"""
        print("\n" + "=" * 60)
        print("H3: 情感支持性变化分析")
        print("=" * 60)

        if use_did and treatment_subreddits and control_subreddits:
            print("  使用DID方法分析...")
            return self._h3_did_analysis(df, treatment_subreddits, control_subreddits)

        print("  使用传统方法分析...")

        before = df[df['period'] == 'before']
        after = df[df['period'] == 'after']

        support_before = before['support_score'].mean()
        support_after = after['support_score'].mean()

        print(f"\n情感支持得分:")
        print(f"  ChatGPT发布前: {support_before:.4f}")
        print(f"  ChatGPT发布后: {support_after:.4f}")
        print(f"  变化: {support_after - support_before:+.4f}")

        # 1. 卡方检验（基于中位数分组）
        combined = pd.concat([before['support_score'], after['support_score']])
        median = combined.median()
        before_high = (before['support_score'] > median).sum()
        before_low = (before['support_score'] <= median).sum()
        after_high = (after['support_score'] > median).sum()
        after_low = (after['support_score'] <= median).sum()
        contingency = np.array([[before_high, before_low], [after_high, after_low]])
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

        # 2. t检验
        t_stat, t_p = stats.ttest_ind(after['support_score'], before['support_score'], equal_var=False)

        # 3. Mann-Whitney U检验
        u_stat, mw_p = stats.mannwhitneyu(after['support_score'], before['support_score'], alternative='two-sided')

        # 4. RDD分析
        rdd_result = self._rdd_analysis(df, 'days_from_cutoff1', 'support_score', running_var2='days_from_cutoff2')

        # Cohen's d
        cohens_d = self._cohens_d(after['support_score'], before['support_score'])

        # 情感分布（可选保留）
        sentiment_dist = {
            'before_positive_rate': (before['sentiment_positive'] > 0.05).mean(),
            'after_positive_rate': (after['sentiment_positive'] > 0.05).mean(),
            'before_negative_rate': (before['sentiment_negative'] > 0.05).mean(),
            'after_negative_rate': (after['sentiment_negative'] > 0.05).mean()
        }

        return {
            'hypothesis': 'H3',
            'method': 'Chi-square + t-test + Mann-Whitney U + RDD',
            'before_support': support_before,
            'after_support': support_after,
            'change': support_after - support_before,
            'chi2_pvalue': chi2_p,
            't_pvalue': t_p,
            'mannwhitney_pvalue': mw_p,
            'cohens_d': cohens_d,
            'significant': any(p < 0.05 for p in [chi2_p, t_p, mw_p]),
            'rdd_coefficient': rdd_result['coefficient'],
            'rdd_pvalue': rdd_result['pvalue'],
            'sentiment_distribution': sentiment_dist
        }

    def _h1_did_analysis(self, df, treatment_subreddits, control_subreddits) -> Dict:
        """H1的DID分析"""
        did = self._init_did_analyzer()

        result = did.did_analysis_h1_psychoeducation(
            df, treatment_subreddits, control_subreddits
        )

        return {
            'hypothesis': 'H1',
            'method': 'DID (双重差分)',
            'did_coefficient': result['did_coefficient'],
            'did_se': result['did_se'],
            'did_pvalue': result['did_pvalue'],
            'did_ci_lower': result['did_ci_lower'],
            'did_ci_upper': result['did_ci_upper'],
            'significant': result['did_pvalue'] < 0.05,
            'parallel_trend_passed': result['parallel_trend_passed'],
            'r_squared': result['r_squared']
        }


    def _h2_did_analysis(self, df, treatment_subreddits, control_subreddits) -> Dict:
        """H2的DID分析"""
        did = self._init_did_analyzer()

        result = did.did_analysis_h2_term_density(
            df, treatment_subreddits, control_subreddits
        )

        return {
            'hypothesis': 'H2',
            'method': 'DID (双重差分)',
            'did_coefficient': result['did_coefficient'],
            'did_se': result['did_se'],
            'did_pvalue': result['did_pvalue'],
            'significant': result['did_pvalue'] < 0.05,
            'did_with_controls_coef': result.get('did_with_controls_coef'),
            'did_with_controls_pvalue': result.get('did_with_controls_pvalue'),
            'parallel_trend_passed': result['parallel_trend_passed'],
            'r_squared': result['r_squared'],
            'adj_r_squared': result.get('adj_r_squared')
        }


    def _h3_did_analysis(self, df, treatment_subreddits, control_subreddits) -> Dict:
        """H3的DID分析"""
        did = self._init_did_analyzer()

        result = did.did_analysis_h3_support_score(
            df, treatment_subreddits, control_subreddits
        )

        return {
            'hypothesis': 'H3',
            'method': 'DID (双重差分)',
            'did_coefficient': result['did_coefficient'],
            'did_se': result['did_se'],
            'did_pvalue': result['did_pvalue'],
            'significant': result['did_pvalue'] < 0.05,
            'psm_did_coefficient': result.get('psm_did_coefficient'),
            'psm_did_pvalue': result.get('psm_did_pvalue'),
            'parallel_trend_passed': result['parallel_trend_passed'],
            'placebo_pvalue': result.get('placebo_pvalue'),
            'r_squared': result['r_squared']
        }

    # ==================== 稳健性检验框架 ====================
    def run_robustness_checks(self,
                              df: pd.DataFrame,
                              treatment_subreddits: List[str],
                              control_subreddits: List[str],
                              outcomes: Optional[List[str]] = None) -> Dict:
        """
        运行完整的稳健性检验套件

        Args:
            df: 数据DataFrame
            treatment_subreddits: 处理组
            control_subreddits: 对照组
            outcomes: 要检验的结果变量

        Returns:
            稳健性检验结果字典
        """
        if outcomes is None:
            outcomes = ['psychoedu_score', 'psych_term_density', 'support_score']

        print("\n" + "=" * 80)
        print("稳健性检验套件")
        print("=" * 80)

        robustness_results = {}

        # 1. DID作为主方法（如果还在使用传统方法）
        print("\n1. DID分析（主方法）...")
        did = self._init_did_analyzer()
        robustness_results['did_main'] = did.unified_did_analysis(
            df, treatment_subreddits, control_subreddits, outcomes
        )

        # 2. 替代对照组检验
        print("\n2. 替代对照组检验...")
        robustness_results['alternative_controls'] = {}
        for outcome in outcomes:
            alt_result = self._alternative_control_test(
                df, treatment_subreddits, control_subreddits, outcome
            )
            robustness_results['alternative_controls'][outcome] = alt_result

        # 3. 替代时间窗口检验
        print("\n3. 替代时间窗口检验...")
        robustness_results['alternative_windows'] = self._alternative_window_test(
            df, treatment_subreddits, control_subreddits, outcomes
        )

        # 4. 安慰剂检验
        print("\n4. 安慰剂检验...")
        robustness_results['placebo'] = {}
        for outcome in outcomes:
            placebo = self._placebo_test_did(
                df, treatment_subreddits, control_subreddits, outcome
            )
            robustness_results['placebo'][outcome] = placebo

        # 5. 排除极端值检验
        print("\n5. 排除极端值检验...")
        robustness_results['winsorized'] = self._winsorized_test(
            df, treatment_subreddits, control_subreddits, outcomes
        )

        # 打印总结
        self._print_robustness_summary(robustness_results)

        return robustness_results

    def _alternative_control_test(self, df, treatment_subs, control_subs, outcome):
        """替代对照组检验"""
        did = self._init_did_analyzer()
        return did._alternative_control_test(df, treatment_subs, control_subs, outcome)

    def _alternative_window_test(self, df, treatment_subs, control_subs, outcomes):
        """替代时间窗口检验"""
        did = self._init_did_analyzer()
        window_results = {}

        # 测试不同的时间窗口
        windows = [
            (90, 90),  # 窄窗口
            (180, 180),  # 中等窗口
            (365, 365),  # 标准窗口
            (540, 180),  # 不对称窗口
        ]

        for pre, post in windows:
            window_key = f"pre{pre}_post{post}"
            window_results[window_key] = {}

            for outcome in outcomes:
                try:
                    result = did._prepare_did_data(df, treatment_subs, control_subs, pre, post)
                    if len(result) > 100:  # 最少数据要求
                        did_result = did._run_did_regression(result, outcome)
                        window_results[window_key][outcome] = {
                            'coefficient': did_result['coefficient'],
                            'pvalue': did_result['pvalue']
                        }
                except Exception as e:
                    window_results[window_key][outcome] = {
                        'error': str(e)
                    }

        return window_results

    def _placebo_test_did(self, df, treatment_subs, control_subs, outcome):
        """DID安慰剂检验"""
        did = self._init_did_analyzer()

        # 准备DID数据
        df_did = did._prepare_did_data(df, treatment_subs, control_subs)

        return did._placebo_test(df_did, outcome)

    def _winsorized_test(self, df, treatment_subs, control_subs, outcomes):
        """排除极端值后的检验"""
        did = self._init_did_analyzer()
        winsorized_results = {}

        for outcome in outcomes:
            if outcome in df.columns:
                # Winsorize 1%极端值
                df_wins = df.copy()
                lower = df_wins[outcome].quantile(0.01)
                upper = df_wins[outcome].quantile(0.99)
                df_wins[outcome] = df_wins[outcome].clip(lower, upper)

                try:
                    df_did_wins = did._prepare_did_data(
                        df_wins, treatment_subs, control_subs
                    )
                    did_result = did._run_did_regression(df_did_wins, outcome)
                    winsorized_results[outcome] = {
                        'coefficient': did_result['coefficient'],
                        'pvalue': did_result['pvalue'],
                        'lower_bound': lower,
                        'upper_bound': upper
                    }
                except Exception as e:
                    winsorized_results[outcome] = {'error': str(e)}

        return winsorized_results

    def _print_robustness_summary(self, results: Dict):
        """打印稳健性检验总结"""
        print("\n" + "=" * 80)
        print("稳健性检验总结")
        print("=" * 80)

        # 检查主DID结果
        if 'did_main' in results:
            print("\n主DID结果:")
            for outcome, result in results['did_main'].items():
                sig = '显著' if result['did_pvalue'] < 0.05 else '不显著'
                print(
                    f"  {result['hypothesis']}: 系数={result['did_coefficient']:+.4f}, p={result['did_pvalue']:.4f} ({sig})")

        # 检查替代窗口的一致性
        if 'alternative_windows' in results:
            print("\n替代窗口一致性:")
            for window, window_results in results['alternative_windows'].items():
                for outcome, res in window_results.items():
                    if 'pvalue' in res:
                        print(f"  {window} - {outcome}: p={res['pvalue']:.4f}")

        # 检查安慰剂检验
        if 'placebo' in results:
            print("\n安慰剂检验结果:")
            for outcome, placebo in results['placebo'].items():
                if 'pvalue' in placebo:
                    passed = placebo.get('passed', placebo['pvalue'] > 0.05)
                    status = '✓ 通过' if passed else '✗ 未通过'
                    print(f"  {outcome}: p={placebo['pvalue']:.4f} {status}")

    # ==================== 原有的辅助方法（保持不变） ====================

    def _rdd_analysis(self, df: pd.DataFrame, running_var1: str, outcome: str,
                      running_var2: str = None, bandwidth: int = None):
        """断点回归分析（保持原有代码不变）"""
        if bandwidth is None:
            bandwidth = self.config.RDD_BANDWIDTH

        if running_var2 is not None:
            df_rdd = df[((df[running_var1] <= 0) & (df[running_var1] >= -bandwidth)) |
                        ((df[running_var2] >= 0) & (df[running_var2] <= bandwidth))].copy()
            df_rdd['treated'] = (df_rdd[running_var2] >= 0).astype(int)
            df_rdd['running_var_combined'] = df_rdd[running_var1].fillna(0)
            mask_treated = df_rdd['treated'] == 1
            df_rdd.loc[mask_treated, 'running_var_combined'] = df_rdd.loc[mask_treated, running_var2]
            df_rdd['interaction'] = df_rdd['treated'] * df_rdd['running_var_combined']
            X = df_rdd[['treated', 'running_var_combined', 'interaction']]
        else:
            df_rdd = df[abs(df[running_var1]) <= bandwidth].copy()
            df_rdd['treated'] = (df_rdd[running_var1] >= 0).astype(int)
            df_rdd['running_var_centered'] = df_rdd[running_var1]
            df_rdd['interaction'] = df_rdd['treated'] * df_rdd['running_var_centered']
            X = df_rdd[['treated', 'running_var_centered', 'interaction']]

        X = sm.add_constant(X)
        y = df_rdd[outcome]
        model = sm.OLS(y, X).fit()

        return {
            'coefficient': model.params['treated'],
            'pvalue': model.pvalues['treated'],
            'ci_lower': model.conf_int().loc['treated'][0],
            'ci_upper': model.conf_int().loc['treated'][1]
        }

    def _regression_analysis(self, df: pd.DataFrame, outcome: str) -> Dict:
        """多元回归分析（保持原有代码不变）"""
        df_reg = df.copy()
        period_map = {'before': 0, 'middle': 1, 'after': 2}
        df_reg['period_dummy'] = df_reg['period'].map(period_map)

        subreddit_dummies = pd.get_dummies(df_reg['subreddit'], prefix='sub', dtype=int)
        df_reg = pd.concat([df_reg, subreddit_dummies], axis=1)

        predictors = ['period_dummy', 'text_length'] + list(subreddit_dummies.columns)
        X = df_reg[predictors]
        X = sm.add_constant(X)
        y = df_reg[outcome]

        model = sm.OLS(y, X).fit()

        return {
            'period_coefficient': model.params['period_dummy'],
            'period_pvalue': model.pvalues['period_dummy'],
            'r_squared': model.rsquared,
            'aic': model.aic
        }

    def _propensity_score_matching(self, df: pd.DataFrame) -> Dict:
        """倾向性得分匹配（保持原有代码不变）"""
        from sklearn.neighbors import NearestNeighbors

        features = ['text_length', 'score' if 'score' in df.columns else 'ups']
        features = [f for f in features if f in df.columns]

        if not features:
            return {'ate': 0}

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        X = df[features].fillna(df[features].mean())
        y = (df['period'] == 'after').astype(int)

        propensity_scores = rf.fit(X, y).predict_proba(X)[:, 1]

        treated_idx = df[df['period'] == 'after'].index
        control_idx = df[df['period'] == 'before'].index

        if len(treated_idx) == 0 or len(control_idx) == 0:
            return {'ate': 0}

        avg_treated = df.loc[treated_idx, 'support_score'].mean()
        avg_control = df.loc[control_idx, 'support_score'].mean()

        return {'ate': avg_treated - avg_control}

    def _time_trend_analysis(self, df: pd.DataFrame, outcome: str) -> Dict:
        """时间趋势分析（保持原有代码不变）
        在ChatGPT发布前后的线性趋势变化。
        """
        weekly_avg = df.groupby('year_week').agg({
            outcome: 'mean',
            'date': 'first',
            'period': 'first'
        }).reset_index()
        weekly_avg = weekly_avg.sort_values('date')

        before_trend = weekly_avg[weekly_avg['period'] == 'before'][outcome].values
        after_trend = weekly_avg[weekly_avg['period'] == 'after'][outcome].values

        if len(before_trend) > 1:
            before_slope = np.polyfit(range(len(before_trend)), before_trend, 1)[0]
        else:
            before_slope = 0

        if len(after_trend) > 1:
            after_slope = np.polyfit(range(len(after_trend)), after_trend, 1)[0]
        else:
            after_slope = 0

        return {
            'before_slope': before_slope,
            'after_slope': after_slope,
            'slope_change': after_slope - before_slope
        }

    def _cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = (group1.mean() - group2.mean()) / np.sqrt(pooled_var)

        return abs(d)