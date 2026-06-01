# config.py - 在原有基础上添加DID相关配置
from pathlib import Path


class Config:
    # 时间配置
    CUTOFF_DATE1 = '2022-11-30'  # ChatGPT发布日
    CUTOFF_DATE2 = '2024-09-30'  # 通过文献调研出来，chatgpt普遍使用的日期 # How People Use ChatGPT，应该使用这之后的数据 2024-9-30
    START_DATE = '2021-09-01'
    END_DATE = '2026-04-26'

    # 数据路径
    DATA_DIR = Path('./data')  # 存放jsonl文件的目录
    # SUBREDDITS = ['mentalhealth', 'anxiety', 'depression', 'psychology'] # 在外部定义，因为涉及到对照组和处理组
    # todo 在稳健性方面，可以引入一个与心理学无关的子版块进行分析？看看cutoff前后此子版块中的心理学内容是否出现显著性的区别

    # 分析参数
    RDD_BANDWIDTH = 240  # 断点回归带宽（天）
    MIN_TEXT_LENGTH = 20  # 最小文本长度
    CHUNK_SIZE = 10000  # 分块处理大小

    # 心理学术语词典（基于DSM-5和心理学教科书），事先与定义好 本质上属于 NLP 中的基于词典的方法（Lexicon-based approach），准确来说是 关键词匹配。
    # fixme 这里匹配的缺点就在于,这里定义的词语会影响后面的实验结果
    PSYCH_TERMS = {
        # 心理障碍
        'disorders': ['depression', 'anxiety', 'ptsd', 'ocd', 'adhd', 'bipolar',
                      'schizophrenia', 'panic disorder', 'social anxiety', 'gad'],
        # 治疗相关
        'treatment': ['therapy', 'counseling', 'cbt', 'cognitive behavioral',
                      'dbt', 'exposure therapy', 'medication', 'antidepressant'],

        # 心理机制
        'mechanisms': ['cognition', 'emotion regulation', 'coping', 'resilience',
                       'rumination', 'avoidance', 'hyperarousal', 'flashback'],

        # 评估工具
        'assessment': ['phq-9', 'gad-7', 'beck inventory', 'hamilton scale',
                       'MBTI'],

        # 新增：APA 心理学词典维度
        'personality_traits': [  # 人格特质
            'neuroticism', 'extraversion', 'openness', 'agreeableness', 'conscientiousness',
            'self-esteem', 'self-efficacy', 'locus of control', 'resilience'
        ],
        'cognitive_processes': [  # 认知过程
            'cognitive bias', 'confirmation bias', 'attribution error',
            'metacognition', 'executive function', 'working memory'
        ],
        'developmental': [  # 发展心理学
            'attachment style', 'secure attachment', 'avoidant attachment',
            'childhood trauma', 'developmental milestone'
        ],
        'social_psychology': [  # 社会心理学
            'social support', 'social isolation', 'group dynamics',
            'conformity', 'obedience', 'bystander effect'
        ],
        'theoretical_terms': [  # 理论术语（可能出现在科普内容中）
            'classical conditioning', 'operant conditioning', 'positive reinforcement',
            'cognitive dissonance', 'self-actualization', 'hierarchy of needs'
        ]
    }

    # 心理教育内容关键词,扩充了一下
    PSYCHOEDU_KEYWORDS = [  # ========== 高置信度：几乎必定是心理教育 ==========
        'psychoeducation', 'psychoeducational', 'psychoed', 'psycho-ed',
        'coping strategy', 'coping skill', 'coping mechanism',
        'grounding technique', 'grounding exercise',
        'cognitive restructuring', 'thought record',
        'mental health literacy',
        'therapist approved', 'clinically reviewed',

        # ========== 中置信度：常见于心理教育，需结合上下文 ==========
        # MEDIUM_CONFIDENCE = [
        # 'signs and symptoms', 'diagnostic criteria',
        # 'how to cope', 'self-help', 'self help',
        # 'breathing exercise', 'breathing technique', 'deep breathing',
        # 'journaling', 'progressive muscle', 'body scan',
        # 'mindfulness exercise', 'meditation',
        # 'positive self-talk', 'affirmation',
        # 'fight or flight', 'neurotransmitter',
        # 'the dsm', 'diagnostic manual',
        # 'nami', 'beyond blue', 'headspace',

        # ========== 低置信度：辅助判断，不能独立使用 ==========
        # 'exercise', 'practice',  # 极易误判
        # 'information about', 'learn about',
        # 'research shows', 'studies show',
        # 'how it works', 'what causes',
        # 'i recommend', 'i suggest', 'helpful tip'
    ]

    # todo 情感支持度的词典？分级支持性词表（强/中/弱，权重递减）
    SUPPORT_WORDS = {
        3: ['here for you', 'by your side', 'not alone', 'got your back',
            'you matter', 'proud of you', 'you\'ve got this', 'i believe in you',
            'i hear you', 'i see you', 'you deserve', 'you are worthy'],
        2: ['support', 'help', 'understand', 'care', 'hope', 'better', 'recover',
            'strong', 'you can', 'it gets better', 'hang in there', 'keep going',
            'don\'t give up', 'brave', 'courage', 'healing', 'reach out',
            'i\'m here', 'with you', 'listening', 'talk to me', 'take your time'],
        1: ['okay', 'alright', 'it\'s fine', 'no judgment', 'no pressure',
            'one day at a time', 'this too shall pass']
        }
    # 否定前缀（匹配到则不计分）
    NEGATION_WORDS = ["don't", "do not", "won't", "will not", "can't", "cannot",
                 "not", "no", "never", "doesn't", "didn't"]

    # ==================== DID分析配置 ====================
    # 处理组：心理健康子版块
    TREATMENT_SUBREDDITS = ['mentalhealth', 'anxiety', 'depression', 'psychology']

    # 对照组：非心理健康子版块（需手动确认这些数据存在）fixme, 个人理财板块太多了
    CONTROL_SUBREDDITS = [
        'sports',  # 体育
        # 'technology',  # 科技
        'nutrition',  # 营养学
    ]

    # DID分析参数
    PRE_WINDOW_DAYS = 365  # 处理前窗口
    POST_WINDOW_DAYS = 365  # 处理后窗口
    MIN_OBS_PER_GROUP = 50  # 每组最少观测数

    # 事件研究参数
    EVENT_STUDY_N_LAGS = 8  # 事件研究滞后阶数
    EVENT_STUDY_N_LEADS = 8  # 事件研究前置阶数

    # 稳健性检验
    PLACEBO_DATE_SHIFT = 180  # 安慰剂检验：假处理提前180天
    ALTERNATIVE_N_SAMPLES = 10  # 替代对照组抽样次数