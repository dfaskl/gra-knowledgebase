"""
步骤1: 从 Codeforces API 获取全量题目元数据
输出: data/raw/problems_meta.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_core.fetcher import CodeforcesFetcher


def main():
    fetcher = CodeforcesFetcher()

    print("=" * 60)
    print("Step 1: 获取 Codeforces 题目元数据")
    print("=" * 60)

    # 获取全部题目
    problems = fetcher.fetch_problems_meta()

    # 统计
    print(f"\n[统计]")
    print(f"  总题数: {len(problems)}")
    tags_stats = fetcher.get_tag_statistics(problems)
    print(f"  标签种类: {len(tags_stats)}")
    print(f"  Top 10 标签:")
    for tag, count in list(tags_stats.items())[:10]:
        print(f"    {tag}: {count}")

    # 按 rating 统计
    with_rating = [p for p in problems if p.get("rating")]
    print(f"  有难度评分的题目: {len(with_rating)}")
    if with_rating:
        ratings = [p["rating"] for p in with_rating]
        print(f"  难度范围: {min(ratings)} ~ {max(ratings)}")
        print(f"  难度均值: {sum(ratings) / len(ratings):.0f}")

    # 保存
    fetcher.save_problems_meta(problems)
    print(f"\n✓ 完成! 保存了 {len(problems)} 道题目的元数据")


if __name__ == "__main__":
    main()
