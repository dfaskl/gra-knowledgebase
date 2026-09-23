"""
步骤2: 批量爬取 Codeforces 题目页面
输入: data/raw/problems_meta.json
输出: data/raw/statements/*.json
支持断点续传
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from kb_core.fetcher import CodeforcesFetcher


def main():
    parser = argparse.ArgumentParser(description="批量爬取 Codeforces 题目页面")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--max", type=int, default=None, help="最大爬取数量")
    parser.add_argument("--no-resume", action="store_true", help="不续传，重新爬取")
    parser.add_argument("--sample", action="store_true", help="只爬取抽样题目 (sampled_ids.json)")
    args = parser.parse_args()

    fetcher = CodeforcesFetcher()

    print("=" * 60)
    print("Step 2: 爬取 Codeforces 题目页面")
    print("=" * 60)

    # 加载元数据
    problems = fetcher.load_problems_meta()
    if not problems:
        print("[错误] 未找到元数据文件，请先运行 01_fetch_metadata.py")
        return

    # 抽样模式: 只爬取选中的题目
    if args.sample:
        import json
        sample_path = os.path.join(os.path.dirname(fetcher.load_problems_meta.__code__.co_filename),
                                   "..", "..", "data", "raw", "sampled_ids.json")
        # Find the actual path
        from config import RAW_DIR
        sample_path = os.path.join(RAW_DIR, "sampled_ids.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                sampled_ids = set(json.load(f))
            problems = [p for p in problems if p["problem_id"] in sampled_ids]
            print(f"抽样模式: {len(problems)} 道题目 (来自 {sample_path})")
        else:
            print(f"[警告] 未找到抽样文件: {sample_path}, 使用全部题目")

    print(f"已加载 {len(problems)} 道题目元数据")

    # 批量爬取
    results = fetcher.fetch_all_statements(
        problems,
        resume=not args.no_resume,
        start_from=args.start,
        max_problems=args.max,
    )

    ok = sum(1 for r in results if r.get("_fetch_status") == "ok")
    skipped = sum(1 for r in results if r.get("_fetch_status") == "skipped")
    errors = sum(1 for r in results if "error" in str(r.get("_fetch_status", "")))

    print(f"\n✓ 完成! 成功: {ok}, 跳过: {skipped}, 失败: {errors}")


if __name__ == "__main__":
    main()
