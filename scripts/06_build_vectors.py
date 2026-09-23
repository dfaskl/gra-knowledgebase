"""
步骤6: 向量化与 ChromaDB 入库
输入: data/processed/problems_clean.json + constraints.json + patterns.json
输出: data/chroma_db/ (ChromaDB 持久化)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_core.preprocessor import Preprocessor
from kb_core.constraint_extractor import ConstraintExtractor
from kb_core.pattern_builder import PatternBuilder
from kb_core.vector_store import VectorStore


def main():
    print("=" * 60)
    print("Step 6: 向量化与 ChromaDB 入库")
    print("=" * 60)

    # 加载数据
    problems = Preprocessor.load_clean()
    extractor = ConstraintExtractor()
    constraints = extractor.load()
    builder = PatternBuilder()
    patterns = builder.load()

    print(f"加载: {len(problems)} 道题目, {len(constraints)} 条约束, {len(patterns)} 条模式")

    # 初始化向量存储
    store = VectorStore()

    # 1. 题目向量化
    if problems:
        store.index_problems(problems)
    else:
        print("[跳过] 无题目数据")

    # 2. 模式向量化
    if patterns:
        store.index_patterns(patterns)
    else:
        print("[跳过] 无模式数据")

    # 3. 约束向量化
    if constraints:
        store.index_constraints(constraints)
    else:
        print("[跳过] 无约束数据")

    # 4. 错误模式向量化
    if patterns:
        store.index_error_patterns(patterns)
    else:
        print("[跳过] 无错误模式数据")

    # ── 验证检索 ──
    print(f"\n{'=' * 60}")
    print("[验证] 测试检索功能")
    print("=" * 60)

    test_queries = [
        "Given an array of integers, find the maximum subarray sum",
        "Check if a string is a palindrome",
        "Find the shortest path in a weighted graph",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = store.search_similar_problems(query, n_results=3)
        if results:
            for i, r in enumerate(results):
                meta = r.get("metadata", {})
                print(f"  {i+1}. [{meta.get('problem_id', '?')}] "
                      f"{meta.get('title', '')[:60]} "
                      f"(score: {r['score']:.3f})")
        else:
            print("  (无结果)")

    print(f"\n✓ 完成! ChromaDB 存储在: {store.chroma_dir}")


if __name__ == "__main__":
    main()
