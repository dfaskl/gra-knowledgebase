"""
Codeforces 测试用例知识库 — 全局配置
"""
import os
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── 路径 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
STATEMENTS_DIR = os.path.join(RAW_DIR, "statements")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# 中间文件
PROBLEMS_META_FILE = os.path.join(RAW_DIR, "problems_meta.json")
PROBLEMS_CLEAN_FILE = os.path.join(PROCESSED_DIR, "problems_clean.json")
CONSTRAINTS_FILE = os.path.join(PROCESSED_DIR, "constraints.json")
PATTERNS_FILE = os.path.join(PROCESSED_DIR, "patterns.json")
INDEX_FILE = os.path.join(PROCESSED_DIR, "knowledge_index.json")

# ── Codeforces API ───────────────────────────────────
CF_API_BASE = "https://codeforces.com/api"
CF_PROBLEMS_URL = f"{CF_API_BASE}/problemset.problems"
CF_PROBLEM_PAGE = "https://codeforces.com/problemset/problem/{contestId}/{index}"

# API 限速 (请求/秒)
CF_RATE_LIMIT = 3.0
# 最大重试次数
CF_MAX_RETRIES = 5
# 重试等待基数 (秒)
CF_RETRY_BACKOFF = 2.0

# ── 爬取控制 ─────────────────────────────────────────
# 每个请求间隔 (秒)
FETCH_DELAY = 0.5
# 每 N 题保存一个 checkpoint
CHECKPOINT_INTERVAL = 500
# 请求超时 (秒)
REQUEST_TIMEOUT = 30

# ── 嵌入模型 (OpenAI API) ─────────────────────────────
# 使用 OpenAI Embeddings API, 无需本地 GPU
# text-embedding-3-small: 512维, $0.02/1M tokens
# text-embedding-3-large: 3072维, $0.13/1M tokens
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIM = 512  # text-embedding-3-small 输出 512 维
EMBEDDING_BATCH_SIZE = 20  # OpenAI API 每次最多 2048 个输入

# ── ChromaDB ─────────────────────────────────────────
CHROMA_COLLECTIONS = {
    "problems": "cf_problems",
    "patterns": "cf_patterns",
    "constraints": "cf_constraints",
    "error_patterns": "cf_error_patterns",
}

# ── LLM 配置 ──────────────────────────────────────────
# 用于约束提取和模式构建
# 支持 OpenAI / DeepSeek 兼容 API
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0  # 结构化提取需要确定性输出
LLM_CONCURRENCY = 20   # 并发请求数 (DeepSeek支持更高并发)
LLM_MAX_RETRIES = 5

# ── 聚类 ──────────────────────────────────────────────
CLUSTER_MIN_SIZE = 5       # HDBSCAN 最小簇大小
CLUSTER_MIN_SAMPLES = 3    # HDBSCAN 最小样本数
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5

# ── 标签中英文映射 ───────────────────────────────────
TAG_ZH_MAP = {
    "dp": "动态规划",
    "greedy": "贪心",
    "graphs": "图论",
    "math": "数学",
    "strings": "字符串",
    "data structures": "数据结构",
    "geometry": "计算几何",
    "combinatorics": "组合数学",
    "number theory": "数论",
    "binary search": "二分查找",
    "sortings": "排序",
    "trees": "树",
    "dfs and similar": "DFS及类似",
    "bfs": "广度优先搜索",
    "shortest paths": "最短路",
    "two pointers": "双指针",
    "bitmasks": "位运算",
    "constructive algorithms": "构造性算法",
    "implementation": "实现",
    "brute force": "暴力枚举",
    "divide and conquer": "分治",
    "hashing": "哈希",
    "probabilities": "概率",
    "games": "博弈",
    "flows": "网络流",
    "matrices": "矩阵",
    "fft": "快速傅里叶变换",
    "string suffix structures": "字符串后缀结构",
    "expression parsing": "表达式解析",
    "graph matchings": "图匹配",
    "meet-in-the-middle": "中途相遇",
    "ternary search": "三分查找",
    "chinese remainder theorem": "中国剩余定理",
    "interactive": "交互题",
    " schedules": "调度",
}

# ── 错误类型映射 ─────────────────────────────────────
ERROR_TYPES = {
    "WA": "错误答案 (Wrong Answer)",
    "TLE": "超时 (Time Limit Exceeded)",
    "RE": "运行时错误 (Runtime Error)",
    "MLE": "内存超限 (Memory Limit Exceeded)",
    "overflow": "整数溢出",
    "precision": "浮点精度错误",
    "off_by_one": "边界差一错误",
    "null_reference": "空值/空指针",
    "special_case_miss": "特判遗漏",
    "input_format": "输入格式错误",
}
