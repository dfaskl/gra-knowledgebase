"""
步骤4: LLM 约束条件提取
输入: data/processed/problems_clean.json
输出: data/processed/constraints.json

需要设置环境变量:
  OPENAI_API_KEY  - API 密钥
  OPENAI_API_BASE - API 端点 (默认 OpenAI)
  LLM_MODEL       - 模型名 (默认 gpt-4o-mini)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from kb_core.preprocessor import Preprocessor
from kb_core.constraint_extractor import ConstraintExtractor


def main():
    parser = argparse.ArgumentParser(description="LLM 约束条件提取")
    parser.add_argument("--max", type=int, default=None, help="最大处理题目数")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: LLM 约束条件提取")
    print("=" * 60)

    # 检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("[警告] 未设置 OPENAI_API_KEY 环境变量!")
        print("请设置: $env:OPENAI_API_KEY = 'your-key'  (PowerShell)")
        print("或: export OPENAI_API_KEY='your-key'     (Bash)")

        # 询问是否继续
        resp = input("是否仍然继续? (y/n): ")
        if resp.lower() != 'y':
            return

    # 加载清洗后的题目
    problems = Preprocessor.load_clean()
    if not problems:
        print("[错误] 未找到清洗后数据，请先运行 03_preprocess.py")
        return

    print(f"已加载 {len(problems)} 道清洗后题目")

    # 选择处理的子集
    end = min(args.start + args.max, len(problems)) if args.max else len(problems)
    subset = problems[args.start:end]
    print(f"处理范围: [{args.start}:{end}] = {len(subset)} 道")

    # 提取约束
    extractor = ConstraintExtractor()
    extractor.concurrency = args.concurrency
    constraints = extractor.extract_batch(subset)

    # 合并已有约束（如果增量处理）
    if args.start > 0:
        existing = extractor.load()
        print(f"合并已有 {len(existing)} 条约束...")
        # 去重: 新结果覆盖旧结果中相同 problem_id 的条目
        existing_map = {c.get("problem_id", ""): c for c in existing}
        for c in constraints:
            existing_map[c.get("problem_id", "")] = c
        constraints = list(existing_map.values())

    # 保存
    extractor.save(constraints)

    # 统计
    success = sum(1 for c in constraints if "error" not in c)
    print(f"\n[统计] 成功: {success}/{len(constraints)}")

    if success > 0:
        # 展示一个示例
        ok_constraints = [c for c in constraints if "error" not in c]
        sample = ok_constraints[0]
        print(f"\n[示例] {sample.get('problem_id', '')}:")
        inp = sample.get("input_format", {})
        print(f"  输入格式: {inp.get('description', '')[:100]}")
        explicit = sample.get("constraints", {}).get("explicit", [])
        print(f"  显式约束数: {len(explicit)}")
        implicit = sample.get("implicit_constraints", [])
        print(f"  隐含约束数: {len(implicit)}")
        boundaries = sample.get("boundary_conditions", [])
        print(f"  边界条件数: {len(boundaries)}")
        special = sample.get("special_cases", [])
        print(f"  特殊用例数: {len(special)}")

    print(f"\n✓ 完成!")


if __name__ == "__main__":
    main()
