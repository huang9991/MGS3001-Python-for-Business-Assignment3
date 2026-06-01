# main.py - 重构后的主程序
import os, json
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from hypothesis_tester import HypothesisTester
from DID_analysis import DIDAnalyzer
from ml_enhancer import MLEnhancer
from visualizer import Visualizer
from utils import generate_comprehensive_report
import numpy as np

def main(use_did: bool = True, run_original_methods: bool = True):
    """
    主程序入口
    Args:
        use_did: 是否使用DID方法（推荐）
        run_original_methods: 是否运行原始方法作为稳健性检验
    """

    # 数据特征应该再data_loader里面已经处理好了，todo 所以如果关键词匹配那里需要修改的话，则需要修改data_loader里面的内容
    USE_COLS = [
        'text', 'date', 'subreddit', 'type', 'period',
        'days_from_cutoff1', 'days_from_cutoff2',
        'score', 'ups'
    ] # 原始的特征列

    USE_COLS_FEATURE = [
        'text', 'date', 'subreddit', 'type', 'period',
        'days_from_cutoff1', 'days_from_cutoff2',
        'score', 'ups',
        'psychoedu_score', 'psych_term_density',
        'support_score', 'sentiment_positive', 'sentiment_negative',
        'text_length', 'week', 'year_week'
    ] # 提取了一些人工特征的列

    # 处理组就是原有的心理健康子版块
    treatment_subreddits = config.TREATMENT_SUBREDDITS

    # 对照组需要加载额外数据或从现有数据中筛选
    control_subreddits = config.CONTROL_SUBREDDITS

    output_dir = Path('./output') # 结果保存
    if not os.path.exists('output/visualization_data.csv') or not os.path.exists('output/results_full.json'):
        print("=" * 80)
        print("Reddit心理健康社区分析 - ChatGPT发布前后对比研究")
        print(f"分析方法: {'DID (双重差分) + 传统方法' if use_did else '传统方法'}")
        print("=" * 80)

        # 2. 加载数据
        print("\n【步骤1】加载数据...")

        if not os.path.exists('saved_data/combined_df_treatment.csv'):
            config.SUBREDDITS = [*treatment_subreddits, *control_subreddits]
            loader = DataLoader(config)

            all_data = loader.load_all_data()
            raw_combined_df = pd.concat(all_data.values(), ignore_index=True)
            output_path = Path('saved_data/combined_df_treatment.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_combined_df.to_csv(output_path, index=False)
        else:
            raw_combined_df = pd.read_csv('saved_data/combined_df_treatment.csv', usecols=USE_COLS)
            # 确保日期列格式正确
            raw_combined_df['date'] = pd.to_datetime(raw_combined_df['date'])

        print(f"总数据量: {len(raw_combined_df)} 条记录")
        print(f"时间范围: {raw_combined_df['date'].min()} 到 {raw_combined_df['date'].max()}")

        # 3. 特征提取
        print("\n【步骤2】提取特征...")
        if not os.path.exists('saved_data/combined_df_treatment_with_feature.csv'):
            extractor = FeatureExtractor(config)
            combined_df = extractor.extract_all_features(raw_combined_df)
            combined_df.to_csv('saved_data/combined_df_treatment_with_feature.csv', index=False)
        else:
            combined_df = pd.read_csv('saved_data/combined_df_treatment_with_feature.csv', usecols=USE_COLS_FEATURE)
            # 确保日期列格式正确
            combined_df['date'] = pd.to_datetime(combined_df['date'])

        print(f"特征提取完成，数据形状: {combined_df.shape}")

        # ========== 4. 假设检验 ==========
        print("\n【步骤3】假设检验...")

        results = {
            'main_analysis': {},  # 主分析方法结果
            'robustness_checks': {},  # 稳健性检验结果
            'original_methods': {}  # 原始方法结果
        }

        # 4A. 主分析：统一DID框架（推荐），要加入控制组咯
        if use_did:
            print("\n--- 主分析：统一DID框架 ---")
            did_analyzer = DIDAnalyzer(config)
            # 检查对照组数据是否已加载 fixme 这里 combined_df 没有包含对照组，所以这里需要加载不同的df
            # DID 有两种做法，一种是不同板块中都计算心理内容相关的指标得分来进行稳定性检验；
            # 另一种是深入计算其他板块的主题指标得分判断是否发生变化，但这种需要额外的定义和计算指标，并且不同主题得分的尺度不同。目前使用第一种吧
            combined_df_control = pd.read_csv('saved_data/combined_df_control_with_feature.csv', usecols=USE_COLS_FEATURE)
            combined_df = pd.concat([combined_df, combined_df_control], ignore_index=True) # todo 这里是将之前加载的心理板块与加载的非心理板块合并
            # 这里可以添加加载对照组数据的逻辑
            # 或者使用已有的非心理健康子版块作为替代

            # 统一DID分析
            unified_did_results = did_analyzer.unified_did_analysis(
                combined_df,
                treatment_subreddits,
                control_subreddits,
                outcomes=['psychoedu_score', 'psych_term_density', 'support_score']
            )
            results['main_analysis']['unified_did'] = unified_did_results


            # 也可以分别运行每个假设的DID分析，todo 这里和上面一样的内容，只不过分开计算DID？
            h1_did = did_analyzer.did_analysis_h1_psychoeducation(
                combined_df, treatment_subreddits, control_subreddits
            )
            h2_did = did_analyzer.did_analysis_h2_term_density(
                combined_df, treatment_subreddits, control_subreddits
            )
            h3_did = did_analyzer.did_analysis_h3_support_score(
                combined_df, treatment_subreddits, control_subreddits
            )

            results['main_analysis']['individual_did'] = {
                'H1': h1_did,
                'H2': h2_did,
                'H3': h3_did
            }

        # 4B. 传统方法作为稳健性检验/对比基准
        if run_original_methods:
            print("\n--- 传统方法分析（作为对比基准） ---")
            tester = HypothesisTester(config)

            results['original_methods']['H1'] = tester.test_h1_psychoeducation_proportion(
                combined_df, is_considering_effect_time=True  #
            )
            results['original_methods']['H2'] = tester.test_h2_psych_term_density(combined_df)
            results['original_methods']['H3'] = tester.test_h3_support_score(combined_df)
            # 保存原始结果用于对比
            for h in ['H1', 'H2', 'H3']:
                if h in results['original_methods']:
                    sig = results['original_methods'][h].get('significant', False)
                    print(f"  {h}: {'显著' if sig else '不显著'}")

        # # 5. 机器学习增强分析（可选）
        # print("\n【步骤4】机器学习增强分析...")
        # try:
        #     ml_enhancer = MLEnhancer()
        #     ml_results = ml_enhancer.topic_modeling_analysis(combined_df)
        #     results['ml_analysis'] = ml_results
        # except Exception as e:
        #     print(f"ML分析失败: {e}")
        #     ml_results = {}

        # 5. 准备可视化数据
        print("\n【步骤5】准备可视化数据...")
        visualization_df = prepare_visualization_data(combined_df, results, config, use_did=use_did)
        print(f"可视化数据准备完成，数据形状: {visualization_df.shape}")

        # 6. 保存结果（在可视化之前），这样方便调试
        print("\n【步骤6】保存分析结果...")

        save_results(results, visualization_df, output_dir)
    else:
        combined_df_treatment = pd.read_csv('saved_data/combined_df_treatment_with_feature.csv', usecols=USE_COLS_FEATURE)
        if use_did:
            combined_df_control = pd.read_csv('saved_data/combined_df_control_with_feature.csv',
                                              usecols=USE_COLS_FEATURE)
            combined_df = pd.concat([combined_df_treatment, combined_df_control],
                                    ignore_index=True)  # todo 这里是将之前加载的心理板块与加载的非心理板块合并

        visualization_df = pd.read_csv('output/visualization_data.csv')
        results = json.load(open('output/results_full.json', encoding='utf-8'))

    print("\n【步骤7】生成可视化图表...")
    visualizer = Visualizer(config)
    # 原有的时间序列图（保留）
    visualizer.plot_time_series(
        visualization_df,
        ['psychoedu_score', 'psych_term_density', 'support_score'],
        save_path=output_dir / 'time_series.png'
    )

    # 如果使用了DID，添加DID特有图表
    if use_did:
        from DID_analysis import DIDVisualizer
        did_viz = DIDVisualizer(config)

        # 创建DID图表输出目录
        did_plot_dir = output_dir / 'did_plots'
        did_plot_dir.mkdir(exist_ok=True)

        # 绘制事件研究图（三个结果变量）
        for outcome in ['psychoedu_score', 'psych_term_density', 'support_score']:
            did_viz.plot_event_study(
                visualization_df,  # 使用正确的DataFrame变量名
                outcome,
                treatment_subreddits,  # 使用正确的处理组变量
                control_subreddits,    # 使用正确的对照组变量
                save_path=did_plot_dir / f'event_study_{outcome}.png'
            )

        # 绘制DID趋势图（三个结果变量）
        for outcome in ['psychoedu_score', 'psych_term_density', 'support_score']:
            did_viz.plot_did_trends(
                visualization_df,
                outcome,
                treatment_subreddits,
                control_subreddits,
                save_path=did_plot_dir / f'did_trends_{outcome}.png'
            )

        # 绘制安慰剂检验图（三个结果变量）
        for outcome in ['psychoedu_score', 'psych_term_density', 'support_score']:
            did_viz.plot_placebo_test(
                visualization_df,
                outcome,
                treatment_subreddits,
                control_subreddits,
                save_path=did_plot_dir / f'placebo_{outcome}.png'
            )

        # 绘制方法对比图（如果有传统方法结果）
        if run_original_methods and 'individual_did' in results['main_analysis']:
            # 准备DID结果（取individual_did中的简化版本）
            did_results_for_comparison = {}
            for h, result in results['main_analysis']['individual_did'].items():
                did_results_for_comparison[h] = {
                    'did_coefficient': result.get('did_coefficient', 0),
                    'did_pvalue': result.get('did_pvalue', 1),
                    'significant': result.get('significant', False),
                    'parallel_trend_passed': result.get('parallel_trend_passed', None)
                }

            did_viz.plot_method_comparison(
                did_results_for_comparison,
                results['original_methods'],
                save_path=did_plot_dir / 'method_comparison.png'
            )

    print("\n" + "=" * 80)
    print("分析完成！结果已保存到 output 目录")
    print("=" * 80)

    return results

def convert_to_serializable(obj):
    """将 numpy 类型转为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_results(results, visualization_df, output_dir):
    """保存分析结果和可视化数据"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    ## addition

    # 保存完整 results
    with open('output/results_full.json', 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)

    # 1. 保存可视化数据为CSV
    visualization_df.to_csv(output_dir / 'visualization_data.csv', index=False)
    print(f"✓ 可视化数据已保存到 {output_dir / 'visualization_data.csv'}")

    # 2. 生成综合分析报告
    report_path = output_dir / 'comprehensive_report.txt'
    generate_comprehensive_report(results, report_path, config)
    print(f"✓ 综合分析报告已保存到 {report_path}")

    # 3. 同时保存一份JSON格式的结果摘要（便于程序读取）
    results_summary = {
        'generated_time': datetime.now().isoformat(),
        'use_did': results.get('use_did', True),
        'run_original_methods': results.get('run_original_methods', True),
    }

    # 添加DID结果
    if 'main_analysis' in results and 'individual_did' in results['main_analysis']:
        results_summary['did_results'] = {}
        for h, result in results['main_analysis']['individual_did'].items():
            results_summary['did_results'][h] = {
                'did_coefficient': float(result.get('did_coefficient', 0)),
                'p_value': float(result.get('did_pvalue', 1)),
                'significant': bool(result.get('significant', False))
            }

    # 添加传统方法结果
    if 'original_methods' in results:
        results_summary['traditional_results'] = {}
        for h, result in results['original_methods'].items():
            results_summary['traditional_results'][h] = {
                'p_value': float(result.get('p_value', 1)) if 'p_value' in result else None,
                'significant': bool(result.get('significant', False))
            }

    with open(output_dir / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON结果摘要已保存到 {output_dir / 'results_summary.json'}")


def prepare_visualization_data(original_df, results, config, use_did=True):
    """准备可视化所需的数据，将分析结果合并到原始数据框中"""
    viz_df = original_df.copy()

    # 添加处理组标记
    treatment_subreddits = config.TREATMENT_SUBREDDITS
    control_subreddits = config.CONTROL_SUBREDDITS

    viz_df['is_treatment'] = viz_df['subreddit'].isin(treatment_subreddits)
    viz_df['is_control'] = viz_df['subreddit'].isin(control_subreddits)
    viz_df['is_after'] = viz_df['period'] == 'after' # 其实好像也不需要改成
    viz_df['treatment_effect_indicator'] = viz_df['is_treatment'] & viz_df['is_after'] # False    8710226
        # True     1719897
    if use_did and 'main_analysis' in results:
        individual_did = results['main_analysis'].get('individual_did', {})
        outcomes = ['psychoedu_score', 'psych_term_density', 'support_score']
        for i, outcome in enumerate(outcomes):
            effect_key = f'H{i + 1}'
            if effect_key in individual_did: # H1, H2, H3
                did_coef = individual_did[effect_key].get('did_coefficient', 0)
                viz_df[f'{outcome}_did_effect'] = did_coef * viz_df['treatment_effect_indicator']

    return viz_df

# 主分析入口
if __name__ == "__main__":
    # 可选参数
    USE_DID = True  # 是否使用DID作为主分析方法
    RUN_ORIGINAL = True  # 是否运行原始方法作为对比

    # 1. 初始化配置
    config = Config()
    results = main(use_did=USE_DID, run_original_methods=RUN_ORIGINAL)