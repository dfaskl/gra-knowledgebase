"""
步骤7: 聚类与层次化知识索引
输入: data/processed/problems_clean.json + patterns.json + constraints.json
输出: data/processed/knowledge_index.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kb_core.preprocessor import Preprocessor
from kb_core.constraint_extractor import ConstraintExtractor
from kb_core.pattern_builder import PatternBuilder
from kb_core.cluster import KnowledgeClusterer


def main():
    parser = argparse.ArgumentParser(description="聚类与层次化索引")
    parser.add_argument("--max-cluster", type=int, default=5000, help="聚类用的最大题目数")
    parser.add_argument("--no-cluster", action="store_true", help="跳过聚类")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 7: 聚类与层次化知识索引")
    print("=" * 60)

    # 加载数据
    problems = Preprocessor.load_clean()
    extractor = ConstraintExtractor()
    constraints = extractor.load()
    builder = PatternBuilder()
    patterns = builder.load()

    print(f"加载: {len(problems)} 道题目, {len(constraints)} 条约束, {len(patterns)} 条模式")

    # 初始化聚类器
    clusterer = KnowledgeClusterer()

    # ── 聚类 ──
    cluster_result = {}
    if not args.no_cluster:
        # 取一部分题目聚类 (全量太耗时)
        cluster_subset = problems[:args.max_cluster]
        print(f"聚类样本: {len(cluster_subset)} 道 (max_cluster={args.max_cluster})")

        embeddings = clusterer.embed_problems(cluster_subset)
        cluster_result = clusterer.cluster(embeddings)
    else:
        cluster_result = {"n_clusters": 0, "n_noise": 0, "clusters": {}, "cluster_centers": {}}

    # ── 构建层次化组织 ──
    hierarchy = clusterer.build_hierarchy(
        problems, patterns, constraints, cluster_result
    )

    # 保存
    clusterer.save_index(hierarchy)

    # ── 统计 ──
    print(f"\n[统计]")
    meta = hierarchy["meta"]
    print(f"  总题目: {meta['total_problems']}")
    print(f"  总模式: {meta['total_patterns']}")
    print(f"  聚类簇数: {meta['n_clusters']}")
    print(f"  噪声点: {meta['n_noise']}")

    # 全局层
    global_layer = hierarchy["global_layer"]
    print(f"  全局簇数: {len(global_layer['clusters'])}")
    print(f"  通用模式种类: {len(global_layer['universal_patterns'])}")

    # 领域层
    domain_layer = hierarchy["domain_layer"]
    domains = domain_layer["domains"]
    print(f"  领域数: {len(domains)}")
    print(f"  Top 5 领域:")
    for tag in sorted(domains, key=lambda t: domains[t]["problem_count"], reverse=True)[:5]:
        d = domains[tag]
        print(f"    {d['tag_zh']} ({d['tag_en']}): {d['problem_count']} 题, "
              f"难度 {d['rating_range']['avg']:.0f}" if d['rating_range']['avg'] else f"    {d['tag_zh']}: {d['problem_count']} 题")

    # 实例层
    instance_layer = hierarchy["instance_layer"]
    print(f"  实例条目: {len(instance_layer['by_id'])}")
    for level in instance_layer["by_rating_range"]:
        count = len(instance_layer["by_rating_range"][level])
        print(f"    {level}: {count} 题")

    # ── 检索验证 ──
    print(f"\n[验证] 检索测试:")
    test_tags = ["dp", "greedy", "graphs"]
    for tag in test_tags:
        ids = clusterer.get_by_tag(hierarchy, tag, limit=5)
        print(f"  {tag}: {len(ids)} 题 → {ids[:3]}...")

    test_levels = ["easy", "medium", "hard"]
    for level in test_levels:
        ids = clusterer.get_by_rating(hierarchy, level, limit=5)
        print(f"  {level}: {len(ids)} 题 → {ids[:3]}...")

    print(f"\n✓ 完成! 知识索引: {clusterer.load_index() and 'OK'}")


if __name__ == "__main__":
    main()
