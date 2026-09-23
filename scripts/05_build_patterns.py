"""
步骤5: 测试用例模式库构建
输入: data/processed/problems_clean.json + data/processed/constraints.json
输出: data/processed/patterns.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kb_core.preprocessor import Preprocessor
from kb_core.constraint_extractor import ConstraintExtractor
from kb_core.pattern_builder import PatternBuilder


def main():
    parser = argparse.ArgumentParser(description="测试用例模式库构建")
    parser.add_argument("--max", type=int, default=None, help="最大处理题目数")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5: 测试用例模式库构建")
    print("=" * 60)

    # 加载数据
    problems = Preprocessor.load_clean()
    if not problems:
        print("[错误] 未找到清洗后数据")
        return

    extractor = ConstraintExtractor()
    constraints = extractor.load()
    print(f"加载: {len(problems)} 道题目, {len(constraints)} 条约束")

    # 选择子集
    end = min(args.start + args.max, len(problems)) if args.max else len(problems)
    subset = problems[args.start:end]
    print(f"处理范围: [{args.start}:{end}] = {len(subset)} 道")

    # 构建模式
    builder = PatternBuilder()
    builder.concurrency = args.concurrency
    patterns = builder.build_batch(subset, constraints)

    # 保存
    builder.save(patterns)

    # 统计
    stats = builder.get_pattern_statistics(patterns)
    print(f"\n[统计]")
    print(f"  有模式的题目: {stats['total_problems']}")
    print(f"  随机测试模式: {stats['total_random']}")
    print(f"  边界测试模式: {stats['total_boundary']}")
    print(f"  对抗测试模式: {stats['total_adversarial']}")
    print(f"  错误类型分布 (Top 5):")
    for et, count in sorted(stats["error_types_distribution"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {et}: {count}")
    print(f"  标签覆盖 (Top 10):")
    for tag, count in sorted(stats["tag_coverage"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {tag}: {count}")

    # 示例
    ok_patterns = [p for p in patterns if "error" not in p]
    if ok_patterns:
        sample = ok_patterns[0]
        print(f"\n[示例] {sample.get('problem_id', '')}:")
        pats = sample.get("patterns", {})
        for pt in ["random", "boundary", "adversarial"]:
            items = pats.get(pt, [])
            if items:
                print(f"  {pt}: {len(items)} 个模式")
                print(f"    示例: {items[0].get('description', '')[:120]}")

    print(f"\n✓ 完成!")


if __name__ == "__main__":
    main()
