import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 统计分析
from scipy.interpolate import UnivariateSpline

# 可视化
import matplotlib.pyplot as plt

from config import Config

class Visualizer:
    """结果可视化"""

    def __init__(self, config: Config):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
        self.cutoff_date1 = pd.to_datetime(config.CUTOFF_DATE1)
        self.cutoff_date2 = pd.to_datetime(config.CUTOFF_DATE2)

    def plot_time_series(self, df: pd.DataFrame, metrics: List[str], save_path: str = None):
        """绘制时间序列图

        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        # 按周聚合
        # 1. 确保 date 列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])

        weekly = df.groupby(df['date'].dt.to_period('W')).agg({
            m: 'mean' for m in metrics
        }).reset_index()
        # weekly = df.resample('W', on='date')[metrics].mean().reset_index()

        weekly['date'] = weekly['date'].dt.start_time

        plt.rcParams.update({
            'axes.labelsize': 14,  # X轴和Y轴标签字体
            'axes.titlesize': 16,  # 标题字体
            'legend.fontsize': 12,  # 图例字体
            'xtick.labelsize': 12,  # X轴刻度字体
            'ytick.labelsize': 12,  # Y轴刻度字体
            'font.family': 'sans-serif'  # 可选：设置字体家族
        })

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # 绘制时间序列
            ax.plot(weekly['date'], weekly[metric], 'b-', alpha=0.7, linewidth=1.5)

            # 添加平滑曲线
            if len(weekly) > 10:
                x_numeric = np.arange(len(weekly))
                spline = UnivariateSpline(x_numeric, weekly[metric], s=len(weekly) * 0.1)
                ax.plot(weekly['date'], spline(x_numeric), 'r-', linewidth=2, label='Smooth trend')

            # 添加断点垂直线
            # 添加断点垂直线（两个事件点）
            ax.axvline(self.cutoff_date1, color='red', linestyle='--', linewidth=2,
                       label='Cutoff_date1')
            ax.axvline(self.cutoff_date2, color='orange', linestyle='-.', linewidth=2,
                       label='Cutoff_date2')

            # 添加阴影区域和均值线 todo 为了明显点，可以将cutoff后的值。。。
            before_mask = weekly['date'] < self.cutoff_date1
            middle_mask = (weekly['date'] >= self.cutoff_date1) & (weekly['date'] < self.cutoff_date2)
            after_mask = weekly['date'] >= self.cutoff_date2

            # 前期均值
            if before_mask.any():
                before_mean = weekly.loc[before_mask, metric].mean()
                ax.axhline(before_mean, color='blue', linestyle=':', alpha=0.5,
                           label=f'Before mean: {before_mean:.3f}')

            # 中期均值
            if middle_mask.any():
                middle_mean = weekly.loc[middle_mask, metric].mean()
                ax.axhline(middle_mean, color='purple', linestyle='-.', alpha=0.5,
                           label=f'Middle mean: {middle_mean:.3f}')

            # 后期均值
            if after_mask.any():
                after_mean = weekly.loc[after_mask, metric].mean()
                ax.axhline(after_mean, color='green', linestyle=':', alpha=0.5,
                           label=f'After mean: {after_mean:.3f}')

            # 设置标签
            metric_labels = {
                'psychoedu_score': 'Psychoeducation score',
                'psych_term_density': 'Psychological term density (Per thousand words)',
                'support_score': 'Support score'
            }
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=14)
            ax.set_xlabel('Date')
            ax.legend(loc='best', frameon=True, edgecolor='black', facecolor='white', fancybox=False,  # 直角边框（False）或圆角（True）
                        framealpha=1.0)
            ax.set_title(f'{metric_labels.get(metric, metric)} time trend')
            ax.grid(True, alpha=0.6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()

    def plot_distribution_comparison(self, df: pd.DataFrame, metric: str, save_path: str = None):
        """绘制分布对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        before = df[df['period'] == 'before'][metric]
        after = df[df['period'] == 'after'][metric]

        # 直方图
        axes[0].hist(before, bins=50, alpha=0.5, label='ChatGPT发布前', color='blue', density=True)
        axes[0].hist(after, bins=50, alpha=0.5, label='ChatGPT发布后', color='green', density=True)
        axes[0].set_xlabel('得分')
        axes[0].set_ylabel('密度')
        axes[0].set_title('分布对比')
        axes[0].legend()

        # 箱线图
        data_to_plot = [before, after]
        bp = axes[1].boxplot(data_to_plot, labels=['发布前', '发布后'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        axes[1].set_ylabel('得分')
        axes[1].set_title('分位数对比')

        plt.suptitle(f'{metric} 前后期分布对比')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_rdd(self, df: pd.DataFrame, outcome: str, save_path: str = None):
        """绘制断点回归图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制散点
        days = df['days_from_cutoff2'] # cutoff1是相对chatgpt刚发出来的日期天数差
        values = df[outcome]

        ax.scatter(days, values, alpha=0.3, s=10, c='gray')

        # 拟合局部回归线
        for period, color in [('before', 'blue'), ('after', 'red')]:
            mask = (df['period'] == period) & (abs(days) <= 90)
            x = days[mask]
            y = values[mask]

            if len(x) > 5:
                # 局部多项式拟合
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_smooth, p(x_smooth), color=color, linewidth=2,
                        label=f'{"前期" if period == "before" else "后期"}趋势')

        # 添加断点线
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='ChatGPT发布')
        ax.axhline(values[days == 0].mean() if any(days == 0) else None,
                   color='purple', linestyle=':', label='断点处均值')

        ax.set_xlabel('距离断点天数 (负=之前, 正=之后)')
        ax.set_ylabel('得分')
        ax.set_title(f'断点回归分析: {outcome}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()