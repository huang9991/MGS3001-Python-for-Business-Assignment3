import pandas as pd
import re # 处理正则表达式

from typing import Dict, List, Tuple, Optional
import warnings
from config import Config
warnings.filterwarnings('ignore')

from nltk.sentiment import SentimentIntensityAnalyzer # 这是 NLTK 库中的 VADER 情感分析器，专门用于分析文本的情感倾向（积极、消极、中性）。
## 实际上也是一个基于规则匹配的打分机器，与机器学习和深度学习方法不同
# 进度条
from tqdm import tqdm


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

        # 计算匹配的关键词数量，是基于规则的硬匹配，不关心关键词出现的次数（只要出现就算 1 次），也不关心上下文语义。只看是否匹配到预定的字符
        ## 这个匹配算法虽然简单有效，但是比较局限
        matches = sum(1 for keyword in self.config.PSYCHOEDU_KEYWORDS
                      if keyword.lower() in text_lower)

        # 归一化得分（考虑文本长度的影响），就是说同样出现了多种关键词，但是长文本肯定信息量没有短文本密集~
        max_possible = min(10, len(text) / 100)  # 每100字符最多1分
        raw_score = matches / max(max_possible, 1)

        return min(1.0, raw_score)

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
        supportive_words = ['support', 'help', 'understand', 'care', 'hope',
                            'better', 'recover', 'strong', 'you can', 'it gets better',
                            '支持', '帮助', '理解', '关心', '希望', '更好', '康复']

        text_lower = text.lower()
        supportive_count = sum(1 for word in supportive_words if word in text_lower)
        supportive_score = min(1.0, supportive_count / 20)  # 归一化

        # 综合得分：VADER复合得分 + 支持性词汇得分，权重0.6和0.1，可以设置为超参数？->可以的，这里只是经验值
        ## 为什么要两个合并，而不是单独的情感分析：
        # 单纯情感分析不够：积极文本可能只是"哈哈"而非真正支持；支持性词汇可能出现在消极上下文中（"我希望你能好转"虽然是积极意图但VADER可能判中性）
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