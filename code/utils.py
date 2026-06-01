# utils.py - 工具函数
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def generate_comprehensive_report(results: Dict, output_path: Path, config: Any):
    """生成综合分析报告，包含DID和传统方法的对比"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Reddit心理健康社区综合分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ChatGPT发布日期: {config.CUTOFF_DATE1}\n")
        f.write("=" * 80 + "\n\n")

        # ========== 1. 主分析方法结果 ==========
        if 'main_analysis' in results and results['main_analysis']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("第一部分：主分析结果（DID双重差分法）\n")
            f.write("=" * 80 + "\n\n")

            if 'unified_did' in results['main_analysis']:
                unified = results['main_analysis']['unified_did']

                f.write("表1: 统一DID分析结果\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'假设':<20} {'DID系数':<12} {'标准误':<10} {'p值':<10} {'显著性':<10}\n")
                f.write("-" * 70 + "\n")

                for outcome, result in unified.items():
                    sig = '***' if result['did_pvalue'] < 0.001 else '**' if result['did_pvalue'] < 0.01 else '*' if \
                    result['did_pvalue'] < 0.05 else 'n.s.'
                    f.write(f"{result['hypothesis']:<20} {result['did_coefficient']:+.4f}     "
                            f"{result['did_se']:.4f}     {result['did_pvalue']:.4f}     {sig:<10}\n")

                f.write("-" * 70 + "\n")
                f.write("注: *** p<0.001, ** p<0.01, * p<0.05, n.s.=不显著\n\n")

                # 平行趋势检验结果
                f.write("\n平行趋势检验:\n")
                f.write("-" * 40 + "\n")
                for outcome, result in unified.items():
                    status = '✓ 通过' if result['parallel_trend_passed'] else '✗ 未通过'
                    f.write(f"  {result['hypothesis']}: {status}\n")

        # ========== 2. 原始方法结果（对比基准） ==========
        if 'original_methods' in results and results['original_methods']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("第二部分：传统方法结果（作为对比基准）\n")
            f.write("=" * 80 + "\n\n")

            for hypothesis, result in results['original_methods'].items():
                f.write(f"\n{hypothesis} 检验结果:\n")
                f.write("-" * 40 + "\n")

                if hypothesis == 'H1':
                    f.write(f"  方法: 卡方检验 + 断点回归(RDD)\n")
                    f.write(f"  发布前比例: {result.get('before_prop', 'N/A')}\n")
                    f.write(f"  发布后比例: {result.get('after_prop', 'N/A')}\n")
                    f.write(f"  卡方检验p值: {result.get('chi2_pvalue', 'N/A')}\n")
                    f.write(f"  RDD系数: {result.get('rdd_coefficient', 'N/A')}\n")
                    f.write(f"  RDD p值: {result.get('rdd_pvalue', 'N/A')}\n")

                elif hypothesis == 'H2':
                    f.write(f"  方法: t检验 + 多元回归\n")
                    f.write(f"  发布前密度: {result.get('before_density', 'N/A')}\n")
                    f.write(f"  发布后密度: {result.get('after_density', 'N/A')}\n")
                    f.write(f"  t检验p值: {result.get('t_pvalue', 'N/A')}\n")
                    f.write(f"  Cohen's d: {result.get('cohens_d', 'N/A')}\n")

                elif hypothesis == 'H3':
                    f.write(f"  方法: Mann-Whitney U检验 + PSM\n")
                    f.write(f"  发布前得分: {result.get('before_support', 'N/A')}\n")
                    f.write(f"  发布后得分: {result.get('after_support', 'N/A')}\n")
                    f.write(f"  MWU p值: {result.get('mannwhitney_pvalue', 'N/A')}\n")
                    f.write(f"  PSM ATE: {result.get('psm_ate', 'N/A')}\n")

                # 结论
                if result.get('significant', False):
                    f.write(f"  结论: {hypothesis} 成立（传统方法）\n")
                else:
                    f.write(f"  结论: {hypothesis} 不成立（传统方法）\n")

        # ========== 3. 方法对比总结 ==========
        if ('main_analysis' in results and results['main_analysis'] and
                'original_methods' in results and results['original_methods']):

            f.write("\n" + "=" * 80 + "\n")
            f.write("第三部分：方法对比与方法论讨论\n")
            f.write("=" * 80 + "\n\n")

            f.write("表2: DID vs 传统方法结果对比\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'假设':<8} {'DID结果':<15} {'传统方法':<15} {'结论一致性':<15}\n")
            f.write("-" * 70 + "\n")

            unified = results['main_analysis'].get('unified_did', {})
            original = results['original_methods']

            for h_key, orig_result in original.items():
                h_num = h_key  # 'H1', 'H2', 'H3'

                # 找到对应的DID结果
                did_result = None
                for outcome, res in unified.items():
                    if h_key in res.get('hypothesis', ''):
                        did_result = res
                        break

                orig_sig = '显著' if orig_result.get('significant', False) else '不显著'
                did_sig = '显著' if (did_result and did_result.get('did_pvalue', 1) < 0.05) else '不显著'
                consistent = '✓' if orig_sig == did_sig else '✗'

                f.write(f"{h_key:<8} {did_sig:<15} {orig_sig:<15} {consistent:<15}\n")

            f.write("-" * 70 + "\n\n")

            f.write("方法论讨论:\n")
            f.write("1. DID方法的优势:\n")
            f.write("   - 通过对照组消除时间趋势和季节性效应\n")
            f.write("   - 更严格的因果推断框架\n")
            f.write("   - 平行趋势假设可通过事件研究法检验\n\n")

            f.write("2. 传统方法的价值:\n")
            f.write("   - 作为稳健性检验和方法论对比\n")
            f.write("   - 在对照组数据不可用时提供初步证据\n")
            f.write("   - RDD在断点附近提供局部因果效应估计\n\n")

            f.write("3. 建议:\n")
            f.write("   - 以DID结果为主要结论\n")
            f.write("   - 传统方法结果作为辅助证据和稳健性支持\n")
            f.write("   - 报告两种方法的结果以增强透明度\n")

def save_results_json(results: Dict, output_path: Path):
    """将结果保存为JSON格式，便于后续分析"""
    def convert_to_serializable(obj):
        """将numpy类型转换为可JSON序列化的类型"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)