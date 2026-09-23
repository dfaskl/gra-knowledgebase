"""
步骤3: 数据预处理与清洗
输入: data/raw/statements/*.json
输出: data/processed/problems_clean.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_core.preprocessor import Preprocessor


def main():
    print("=" * 60)
    print("Step 3: 数据预处理与清洗")
    print("=" * 60)

    # 批量清洗
    clean_problems = Preprocessor.batch_clean()

    if not clean_problems:
        print("[错误] 没有找到题目文件，请先运行 02_fetch_statements.py")
        return

    # 统计
    print(f"\n[统计]")
    print(f"  清洗后题目数: {len(clean_problems)}")

    tags_count = {}
    rating_count = 0
    for p in clean_problems:
        for tag in p.get("tags", []):
            tags_count[tag] = tags_count.get(tag, 0) + 1
        if p.get("rating"):
            rating_count += 1
    print(f"  标签种类: {len(tags_count)}")
    print(f"  有难度评分: {rating_count}")

    # 检查清洗质量
    empty_desc = sum(1 for p in clean_problems if not p.get("description"))
    empty_input = sum(1 for p in clean_problems if not p.get("input_spec"))
    empty_output = sum(1 for p in clean_problems if not p.get("output_spec"))
    has_samples = sum(1 for p in clean_problems if p.get("sample_tests"))
    print(f"\n[质量检查]")
    print(f"  空描述: {empty_desc}")
    print(f"  空输入说明: {empty_input}")
    print(f"  空输出说明: {empty_output}")
    print(f"  有示例: {has_samples}")

    # 保存
    Preprocessor.save_clean(clean_problems)

    # 展示一个示例
    print(f"\n[示例] 展示第一道题目的清洗结果:")
    if clean_problems:
        sample = clean_problems[0]
        print(f"  ID: {sample.get('problem_id', '')}")
        print(f"  Title: {sample.get('title', '')[:80]}")
        print(f"  Tags: {', '.join(sample.get('tags', []))}")
        print(f"  Description: {sample.get('description', '')[:200]}...")
        print(f"  Input Spec: {sample.get('input_spec', '')[:200]}...")
        print(f"  Samples: {len(sample.get('sample_tests', []))}")

    print(f"\n✓ 完成!")


if __name__ == "__main__":
    main()
