"""
这里是显示数据处理的过程

"""
import time

import pandas as pd
import json, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from sympy.physics.units import us

from config import Config
warnings.filterwarnings('ignore')


class DataLoader:
    """加载并预处理Reddit数据"""
    def __init__(self, config: Config):
        self.config = config
        self.cutoff_date1 = pd.to_datetime(config.CUTOFF_DATE1)
        self.cutoff_date2 = pd.to_datetime(config.CUTOFF_DATE2)

    def load_subreddit_data(self, subreddit: str, data_type: str = 'both') -> pd.DataFrame:
        """
        加载单个子版块的数据

        Parameters:
        -----------
        subreddit: 子版块名称
        data_type: 'posts', 'comments', 或 'both'，注意both就是都加载
        """
        all_data = []

        # 确定文件路径
        post_file = self.config.DATA_DIR / f'r_{subreddit}_posts.jsonl' # 用Path库的文件路径，与os.path用法有区别
        comment_file = self.config.DATA_DIR / f'r_{subreddit}_comments.jsonl'

        # 加载帖子
        print('加载帖子')
        if data_type in ['posts', 'both'] and post_file.exists():
            posts = self._load_jsonl(post_file)
            posts['type'] = 'post'
            all_data.append(posts)
            print(f"  加载 {subreddit} 帖子: {len(posts)} 条")

        # 加载评论
        if data_type in ['comments', 'both'] and comment_file.exists():
            comments = self._load_jsonl(comment_file)
            comments['type'] = 'comment'
            all_data.append(comments)
            print(f"  加载 {subreddit} 评论: {len(comments)} 条")

        if not all_data:
            return pd.DataFrame()

        # 合并
        df = pd.concat(all_data, ignore_index=True)

        # 数据的基本清洗
        df = self._clean_dataframe(df)

        # 时间过滤
        df = self._filter_by_date(df)

        return df

    def _load_jsonl(self, filepath: Path) -> pd.DataFrame:
        """加载JSONL文件（分块处理大文件）"""
        data = []

        # 先尝试读取前几行判断是否需要分块
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # 如果超过100行，使用分块读取
                        return self._load_jsonl_chunked(filepath)
                    if line.strip():
                        data.append(json.loads(line))
            return pd.DataFrame(data)
        except:
            return self._load_jsonl_chunked(filepath)

    def _load_jsonl_chunked(self, filepath: Path) -> pd.DataFrame:
        """分块加载大文件, 当然也可以 with open(filepath, 'w') as f:
        for line in f:
        if line:
        .......
        """
        chunks = []
        for chunk in pd.read_json(filepath, lines=True, chunksize=self.config.CHUNK_SIZE):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗DataFrame"""

        #  先按 type 分开处理，帖子和评论分开处理，不分开的话会出现问题：body 列在帖子中全部为 NaN，selftext 列在评论中全部为 NaN
        df_post = df[df['type'] == 'post'].copy()
        df_comment = df[df['type'] == 'comment'].copy()

        # 处理帖子
        print('去除空值之前帖子的数量:', len(df_post))
        df_post = df_post.dropna(subset=['selftext', 'ups', 'id'], how='any')
        print('去除空值之后帖子的数量:', len(df_post))

        df_post['text'] = df_post['selftext'].fillna('')
        # if 'title' in df_post.columns:
        df_post['text'] = df_post['title'].fillna('') + ' ' + df_post['text']

        # 处理评论
        print('去除空值之前的评论数量:', len(df_comment))
        df_comment = df_comment.dropna(subset=['body', 'ups', 'id'], how='any')
        df_comment['text'] = df_comment['body'].fillna('')
        print('去除空值之后的评论数量:', len(df_comment))

        # 合并
        df = pd.concat([df_post, df_comment], ignore_index=True)

        # 转换时间，帖子或者评论发布的时间
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')

        # 过滤无效文本，或者小于指定长度的文本
        print('过滤无效文本前的长度:', len(df))
        df = df[df['text'].str.len() >= self.config.MIN_TEXT_LENGTH]
        print('过滤无效文本后的长度:', len(df))

        # Remove deleted/removed content 删除那些内容已被版主或用户移除/删除的帖子或评论
        print('删除deleted/removed内容之前的长度：', len(df))
        df = df[~df['text'].str.contains(r'\[removed\]|\[deleted\]', regex=True, na=False)]
        print('删除deleted/removed内容之后的长度：', len(df))

        # 删除重复的id
        print('删除重复id之前的长度:', len(df))
        df = df.drop_duplicates(subset=['id'], keep='first')
        print('删除重复id之后的长度:', len(df))

        # 添加时期标识，gpt发布前后的标识
        df['period'] = df['date'].apply(
            lambda x: 'after' if x >= self.cutoff_date2 else ('before' if x<= self.cutoff_date1 else 'middle'))

        # 添加相对时间（用于RDD），以天为单位？
        df['days_from_cutoff1'] = (df['date'] - self.cutoff_date1).dt.days # 与chatgpt刚发出来的日期的天数差
        df['days_from_cutoff2'] = (df['date'] - self.cutoff_date2).dt.days

        return df

    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期范围过滤"""
        start = pd.to_datetime(self.config.START_DATE)
        end = pd.to_datetime(self.config.END_DATE)
        return df[(df['date'] >= start) & (df['date'] <= end)]

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有子版块数据"""
        all_data = {}
        for subreddit in self.config.SUBREDDITS:
            print(f"\n加载 r/{subreddit}:")
            df = self.load_subreddit_data(subreddit)
            if not df.empty:
                all_data[subreddit] = df
                print(f"  {subreddit}板块最终保留: {len(df)} 条记录")
        return all_data

if __name__ == '__main__':
    config = Config()

    # 处理组：心理健康子版块
    TREATMENT_SUBREDDITS = ['mentalhealth', 'anxiety', 'depression', 'psychology']

    # 对照组：非心理健康子版块（需手动确认这些数据存在）fixme, 个人理财板块太多了
    CONTROL_SUBREDDITS = [
        'sports',  # 体育
        # 'technology',  # 科技
        'nutrition',  # 营养学
    ]
    config.SUBREDDITS = TREATMENT_SUBREDDITS + CONTROL_SUBREDDITS

    loader = DataLoader(config)
    # all_data = loader.load_all_data()

    dtype_dict = {
        'score': 'int64',
        'subreddit': 'object',
        'ups': 'int64',
        'type': 'object',
        'text': 'object',
        'date': 'object',
        'period': 'object',
        'days_from_cutoff1': 'int64',
        'days_from_cutoff2': 'int64',
        'psychoedu_score': 'float64',
        'psych_term_density': 'float64',
        'support_score': 'float64',
        'sentiment_positive': 'float64',
        'sentiment_negative': 'float64',
        'text_length': 'int64',
        'week': 'int64',
        'year_week': 'object'
    }

    USE_COLS = ['subreddit', 'score', 'ups', 'num_comments', 'text_length']

    initial_time = time.time()
    combined_df_control = pd.read_csv('saved_data/combined_df_control_with_feature.csv',
                                      usecols=USE_COLS, engine='c', dtype=dtype_dict)

    combined_df_treatment = pd.read_csv('saved_data/combined_df_treatment_with_feature.csv', usecols=USE_COLS)

    combined_df = pd.concat([combined_df_control, combined_df_treatment], ignore_index=True)

    print("Descriptive Statistics (numeric variables):")
    total_tongji_df = pd.DataFrame()
    for subreddit in combined_df['subreddit'].unique():
        print(f"================={subreddit}=============")
        res = combined_df[combined_df['subreddit']==subreddit].loc[:, USE_COLS[1:]].describe().round(2)
        res['subreddit'] = subreddit
        print('Value count sum')
        print(combined_df[combined_df['subreddit']==subreddit].loc[:, USE_COLS[1:]].sum())
        print('Value count mean')
        print(combined_df[combined_df['subreddit']==subreddit].loc[:, USE_COLS[1:]].mean())

        total_tongji_df = pd.concat([total_tongji_df, res], axis=0)

    # total_tongji_df.to_csv('saved_data/total_tongji_df.csv', index=False)