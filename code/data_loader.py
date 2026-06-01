import numpy as np
import pandas as pd
import json, os, re, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from config import Config
warnings.filterwarnings('ignore')
from nltk.sentiment import SentimentIntensityAnalyzer # 这是 NLTK 库中的 VADER 情感分析器，专门用于分析文本的情感倾向（积极、消极、中性）。
# 下载NLTK数据（首次运行）
# import nltk
# nltk.download('vader_lexicon')

from tqdm import tqdm


# ==================== 数据加载与预处理 ====================
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
        df_post = df[df['type'] == 'post'].copy()
        df_comment = df[df['type'] == 'comment'].copy()

        # 处理帖子
        df_post = df_post.dropna(subset=['selftext', 'ups', 'id'], how='any')

        df_post['text'] = df_post['selftext'].fillna('')
        # if 'title' in df_post.columns:
        df_post['text'] = df_post['title'].fillna('') + ' ' + df_post['text']

        # 处理评论
        df_comment = df_comment.dropna(subset=['body', 'ups', 'id'], how='any')
        df_comment['text'] = df_comment['body'].fillna('')

        # 合并
        df = pd.concat([df_post, df_comment], ignore_index=True)

        # 转换时间，帖子或者评论发布的时间
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')

        # 过滤空文本，或者小于指定长度的文本
        df = df[df['text'].str.len() >= self.config.MIN_TEXT_LENGTH]

        # 添加时期标识，gpt发布前后的标识
        df['period'] = df['date'].apply(
            lambda x: 'after' if x >= self.cutoff_date2 else ('before' if x<= self.cutoff_date1 else 'middle')
        )

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
                print(f"  最终保留: {len(df)} 条记录")
        return all_data

class FeatureExtractor:
    """提取分析所需的特征"""

    def __init__(self, config: Config):
        self.config = config
        self.psych_terms = self._flatten_psych_terms()
        self.sia = SentimentIntensityAnalyzer()

    def _flatten_psych_terms(self) -> List[str]:
        """扁平化心理学术语词典
        扁平化其实就是flatten操作，将分层级的词语拉成一个一维向量
        """
        all_terms = []
        for category, terms in self.config.PSYCH_TERMS.items():
            all_terms.extend(terms)
        return list(set(all_terms))  # set去重，再转换回去列表

    def extract_psychoeducational_score(self, text: str) -> float:
        """
        提取心理教育内容得分
        基于关键词匹配，返回0-1之间的得分，相当于得到一个
        """
        if not isinstance(text, str):
            return 0.0

        text_lower = text.lower() #将文本全部小写~

        # 统计出现了多少种不同的关键词
        matched_keywords = set()
        for keyword in self.config.PSYCHOEDU_KEYWORDS:
            if keyword.lower() in text_lower:
                matched_keywords.add(keyword)

        matches = len(matched_keywords)
        if matches == 0:
            return 0.0

        # 归一化：匹配种类 / min(预期上限, 文本长度/100)
        # 短文本出现少量关键词即可得高分，长文本需要更多关键词
        max_possible = min(len(self.config.PSYCHOEDU_KEYWORDS), max(1, len(text) / 100))

        return round(min(1.0, matches / max_possible), 4)

    def extract_psych_term_density(self, text: str) -> float:
        """
        提取心理学术语密度（每千字中的术语数量）
        """
        if not isinstance(text, str) or len(text) < 10:
            return 0.0

        text_lower = text.lower()

        # 统计术语出现次数，术语要人工提前定义好，尽量不能缺漏
        term_count = 0
        for term in self.psych_terms:
            term_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower)) # 在文本中查找某个单词（term）完整匹配的所有出现位置。

        # 计算密度（每千字），每一千字出现关键词的密度
        text_length_k = len(text) / 1000
        density = term_count / text_length_k if text_length_k > 0 else 0

        return density

    def extract_sentiment_support_score(self, text: str) -> Dict[str, float]:
        """
        提取情感支持得分
        返回多个维度的得分
        """
        if not isinstance(text, str): # 如果不是字符串，是数字、空或者None就返回0
            return {'support_score': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        # 使用VADER进行情感分析
        sentiment = self.sia.polarity_scores(text) # ：使用VADER分析文本的情感倾向，返回：
        # VADER的compound在-1（最消极）到+1（最积极）之间

        # 计算支持性得分（基于特定词汇）， 统计文本中包含多少个支持性词汇
        supportive_words = self.config.SUPPORT_WORDS
        negations = self.config.NEGATION_WORDS
        text_lower = text.lower()

        supportive_score = 0.0

        for weight, words in supportive_words.items():
            for word in words:
                idx = text_lower.find(word)
                if idx == -1:
                    continue
                # 否定检测：取该词前面30个字符，检查是否含否定词
                prefix = text_lower[max(0, idx - 30):idx]
                if any(neg in prefix.split() for neg in negations):
                    continue
                supportive_score += weight

        # tanh归一化，映射到[0, 1]，比线性截断更平滑
        supportive_score = math.tanh(-supportive_score / 3)

        # 综合得分：VADER复合得分 + 支持性词汇得分，权重0.6和0.1，可以设置为超参数？->可以的，这里只是经验值
        ## 为什么要两个合并，而不是单独的情感分析：
        # 单纯情感分析不够：积极文本可能只是"哈哈"而非真正支持；支持性词汇可能出现在消极上下文中（"我希望你能好转"虽然是积极意图但VADER可能判中性）\
        # 把 VADER 的 compound 从 [-1, 1] 线性映射到 [0, 1]。
        combined = (sentiment['compound'] + 1) / 2 * 0.6 + supportive_score * 0.4

        return {
            'support_score': combined,
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu']
        }

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取所有特征"""
        print("提取特征...")

        ## 文章所说的三个维度的指标
        # 心理教育得分
        tqdm.pandas(desc="计算心理教育得分") # tqdm.pandas()：将tqdm集成到pandas中，让pandas的apply操作显示进度，激活一次就行，不过desc这里不太合规
        df['psychoedu_score'] = df['text'].progress_apply(self.extract_psychoeducational_score)

        # 心理术语密度
        df['psych_term_density'] = df['text'].progress_apply(self.extract_psych_term_density)

        # 情感支持得分
        sentiment_results = df['text'].progress_apply(self.extract_sentiment_support_score)
        df['support_score'] = sentiment_results.apply(lambda x: x['support_score'])
        df['sentiment_positive'] = sentiment_results.apply(lambda x: x['positive'])
        df['sentiment_negative'] = sentiment_results.apply(lambda x: x['negative'])

        # 添加文本长度作为控制变量
        df['text_length'] = df['text'].str.len() # 计算每条文本的字符数，作为后续分析的控制变量。

        # 添加周数（用于趋势分析）
        df['week'] = df['date'].dt.isocalendar().week # 提取日期所在的一年中的第几周（1-53）。
        df['year_week'] = df['date'].dt.strftime('%Y-%W')

        return df


if __name__ == '__main__':
    #
    # 1. 配置和加载数据
    config = Config()
    config.SUBREDDITS = ['mentalhealth', 'anxiety', 'depression', 'psychology'] # ['mentalhealth', 'anxiety', 'depression', 'psychology'], ['sports', 'nutrition']
    loader = DataLoader(config)

    USE_COLS = [
        'text', 'date', 'subreddit', 'type', 'period',
        'days_from_cutoff1', 'days_from_cutoff2',
        'score', 'ups']

    saved_file_name = 'combined_df_treatment' # combined_df_treatment   combined_df_control

    print("\n【步骤1】加载数据...")

    if not os.path.exists('saved_data/%s.csv' % saved_file_name):
        all_data = loader.load_all_data()
        combined_df = pd.concat(all_data.values(), ignore_index=True)

        output_path = Path('saved_data/%s.csv' % saved_file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n【步骤2】合并数据...")

        combined_df.to_csv(output_path, index=False)
    else:
        print("\n【步骤2】合并数据...")
        combined_df = pd.read_csv('saved_data/%s.csv' % saved_file_name, usecols=USE_COLS)
    # 2. 合并所有子版块数据,todo现在数据的字段有100多个，其实真正需要的只有几个，到时候选择需要分析的进行合并、特征提取什么的就行了
    print(f"总数据量: {len(combined_df)} 条记录")
    print(f"时间范围: {combined_df['date'].min()} 到 {combined_df['date'].max()}")

    USE_COLS_feature = [
        'text', 'date', 'subreddit', 'type', 'period',
        'days_from_cutoff1', 'days_from_cutoff2',
        'score', 'ups',
        # 如果已经提取过特征，也加上这些
        'psychoedu_score', 'psych_term_density',
        'support_score', 'sentiment_positive', 'sentiment_negative',
        'text_length', 'week', 'year_week'
    ]

    # 3. 特征提取
    print("\n【步骤3】提取特征...")
    if not os.path.exists('saved_data/%s_with_feature.csv' % saved_file_name):
        extractor = FeatureExtractor(config)
        combined_df = extractor.extract_all_features(combined_df)  # 这个很费时间,可以先保存
        combined_df.to_csv('saved_data/%s_with_feature.csv' % saved_file_name, index=False)
    else:
        combined_df = pd.read_csv('saved_data/%s_with_feature.csv' % saved_file_name, usecols=USE_COLS_feature)
        print(combined_df.head(10), combined_df.columns.values)