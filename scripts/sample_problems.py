"""
分层抽样: 从 11282 道题目中选出代表性样本
策略: 按 rating 分层 + 标签覆盖 + 难度均匀分布
目标: ~2500 题
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from collections import defaultdict
from kb_core.fetcher import CodeforcesFetcher

random.seed(42)

# ── 配置 ──
TARGET = 2500
RATING_RANGES = [
    (0,    1200, "beginner",    200),   # 基础
    (1200, 1600, "easy",        600),   # 简单
    (1600, 2000, "medium",      700),   # 中等
    (2000, 2400, "hard",        500),   # 困难
    (2400, 3000, "expert",      350),   # 专家
    (3000, 9999, "grandmaster", 150),   # 大师
]
# 优先覆盖的核心标签
CORE_TAGS = [
    "dp", "greedy", "graphs", "math", "strings",
    "data structures", "geometry", "combinatorics", "number theory",
    "binary search", "sortings", "trees", "dfs and similar",
    "shortest paths", "two pointers", "bitmasks",
    "divide and conquer", "hashing", "games", "flows",
]

def main():
    fetcher = CodeforcesFetcher()
    problems = fetcher.load_problems_meta()
    print(f"总题数: {len(problems)}")

    # 按 rating 分层
    by_range = defaultdict(list)
    for p in problems:
        r = p.get("rating")
        if r is None:
            continue
        for low, high, label, _ in RATING_RANGES:
            if low <= r < high:
                by_range[label].append(p)
                break

    print("\n各层题目数:")
    for _, _, label, _ in RATING_RANGES:
        print(f"  {label}: {len(by_range[label])}")

    # ── 分层抽样 ──
    sampled = []
    tag_coverage = defaultdict(int)

    for low, high, label, quota in RATING_RANGES:
        pool = by_range[label]
        print(f"\n--- {label} (rating {low}-{high}): {len(pool)} -> quota {quota} ---")

        selected_ids = set()
        selected = []

        # 第1轮: 确保每个核心标签至少选1题
        for tag in CORE_TAGS:
            candidates = [p for p in pool if tag in p.get("tags", [])
                         and p["problem_id"] not in selected_ids]
            if candidates:
                # 选难度最居中的
                mid_rating = (low + high) / 2
                candidates.sort(key=lambda p: abs((p.get("rating") or mid_rating) - mid_rating))
                pick = candidates[0]
                selected.append(pick)
                selected_ids.add(pick["problem_id"])
                for t in pick.get("tags", []):
                    tag_coverage[t] += 1

        print(f"  核心标签覆盖: {len(selected)} 题")

        # 第2轮: 填充剩余配额，保持难度均匀
        remaining_pool = [p for p in pool if p["problem_id"] not in selected_ids]
        needed = quota - len(selected)
        if needed > 0 and remaining_pool:
            # 按 rating 均匀采样
            remaining_pool.sort(key=lambda p: p.get("rating") or 0)
            step = max(1, len(remaining_pool) // needed)
            for i in range(0, len(remaining_pool), step):
                if len(selected) >= quota:
                    break
                pick = remaining_pool[i]
                selected.append(pick)
                for t in pick.get("tags", []):
                    tag_coverage[t] += 1

        sampled.extend(selected)
        print(f"  实际选取: {len(selected)} 题")

    print(f"\n==============")
    print(f"总抽样: {len(sampled)} 题")

    # ── 统计 ──
    print(f"\n难度分布:")
    for low, high, label, _ in RATING_RANGES:
        count = sum(1 for p in sampled if p.get("rating") and low <= p["rating"] < high)
        print(f"  {label}: {count}")

    print(f"\n标签覆盖 (Top 20):")
    for tag, count in sorted(tag_coverage.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag}: {count}")

    # ── 保存 ──
    from config import RAW_DIR
    sampled_ids = [p["problem_id"] for p in sampled]

    outpath = os.path.join(RAW_DIR, "sampled_ids.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(sampled_ids, f, ensure_ascii=False, indent=2)

    meta_outpath = os.path.join(RAW_DIR, "sampled_problems.json")
    with open(meta_outpath, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {outpath} ({len(sampled_ids)} IDs)")
    print(f"已保存: {meta_outpath} ({len(sampled)} 完整元数据)")


if __name__ == "__main__":
    main()
