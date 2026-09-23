"""
测试用例模式库构建模块
从历史数据和约束条件中提取三类测试模式:
1. 随机测试模式 (Random)
2. 边界测试模式 (Boundary)
3. 对抗测试模式 (Adversarial)
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_API_BASE,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    PATTERNS_FILE,
    ERROR_TYPES,
)


PATTERN_GENERATION_PROMPT = """You are an expert in competitive programming test case generation. Given a problem's constraints and structure, generate structured test case patterns.

## Problem Information
{problem_info}

## Extracted Constraints
{constraints}

## Task
Generate 3 types of test case patterns for this problem. For each type, provide:
1. The pattern description
2. Concrete example test cases
3. What kinds of bugs this pattern can catch

## Output Format (JSON only, no markdown):
```json
{{
  "problem_id": "string",
  "patterns": {{
    "random": [
      {{
        "pattern_id": "random_1",
        "description": "description of the random test pattern",
        "generation_strategy": "how to generate random cases following this pattern",
        "example_input": "concrete example",
        "example_output": "expected output (if known, else null)",
        "coverage_dimension": "normal_case | stress_test | large_range",
        "error_types_caught": ["list of error types this pattern can catch"]
      }}
    ],
    "boundary": [
      {{
        "pattern_id": "boundary_1",
        "description": "description of the boundary test",
        "boundary_type": "min_value | max_value | zero | empty | single_element | extreme_range | overflow_risk",
        "test_values": "specific boundary values and why",
        "example_input": "concrete example",
        "example_output": "expected output (if known, else null)",
        "error_types_caught": ["list of error types this pattern can catch"]
      }}
    ],
    "adversarial": [
      {{
        "pattern_id": "adversarial_1",
        "description": "description of the adversarial/hack test",
        "attack_vector": "what vulnerability or weakness this exploits",
        "target_error": "WA | TLE | RE | MLE | precision_error | overflow",
        "example_input": "concrete example designed to break naive solutions",
        "example_output": "expected output (if known, else null)",
        "why_effective": "why naive solutions fail on this input",
        "error_types_caught": ["list of error types"]
      }}
    ]
  }},
  "pattern_summary": {{
    "total_random": "number",
    "total_boundary": "number",
    "total_adversarial": "number",
    "key_coverage_gaps": ["gaps that existing patterns don't cover"],
    "recommended_test_count": "minimum number of test cases needed for adequate coverage"
  }}
}}
```

## Important
1. Each pattern should have at least 1 concrete example
2. Adversarial patterns should specifically target common algorithmic mistakes for this problem type
3. Boundary patterns should cover all constraint extremes mentioned in the problem
4. Random patterns should cover both small (verification) and large (stress) input sizes
5. Include at least 2 patterns of each type (random, boundary, adversarial)
6. Return ONLY valid JSON"""


class PatternBuilder:
    """测试用例模式库构建器"""

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = None,
    ):
        self.client = OpenAI(
            api_key=api_key or LLM_API_KEY,
            base_url=api_base or LLM_API_BASE,
        )
        self.model = model or LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = 0.3  # 略高于约束提取，需要一定创造性
        self.concurrency = LLM_CONCURRENCY
        self.max_retries = LLM_MAX_RETRIES

    def build_problem_info(self, problem: dict) -> str:
        """构建题目信息文本"""
        parts = []
        parts.append(f"Problem ID: {problem.get('problem_id', '')}")
        parts.append(f"Title: {problem.get('title', '')}")
        parts.append(f"Tags: {', '.join(problem.get('tags', []))}")
        if problem.get('rating'):
            parts.append(f"Rating: {problem['rating']}")
        parts.append(f"Description: {problem.get('description', '')[:2000]}")
        parts.append(f"Input: {problem.get('input_spec', '')}")
        parts.append(f"Output: {problem.get('output_spec', '')}")
        return "\n".join(parts)

    def build_single(self, problem: dict, constraint: Optional[dict] = None) -> Optional[dict]:
        """为单道题目构建测试模式"""
        problem_info = self.build_problem_info(problem)
        constraints_text = json.dumps(constraint, ensure_ascii=False, indent=2) if constraint else "Not available"
        pid = problem.get("problem_id", "unknown")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a test case generation expert. Always output valid JSON."},
                        {"role": "user", "content": PATTERN_GENERATION_PROMPT.format(
                            problem_info=problem_info[:6000],
                            constraints=constraints_text[:4000],
                        )},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                result = json.loads(content)
                result["problem_id"] = pid

                # 添加元数据
                result["tags"] = problem.get("tags", [])
                result["rating"] = problem.get("rating")

                return result

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[JSON错误] {pid}: {e}")
                return {"problem_id": pid, "error": "JSON parse error"}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[LLM错误] {pid}: {e}")
                return {"problem_id": pid, "error": str(e)}

    def build_batch(
        self,
        problems: list[dict],
        constraints: Optional[list[dict]] = None,
    ) -> list[dict]:
        """批量构建测试模式"""
        # 构建 constraint 查找表
        constraint_map = {}
        if constraints:
            for c in constraints:
                if "error" not in c:
                    constraint_map[c.get("problem_id", "")] = c

        results = []
        total = len(problems)
        print(f"[模式构建] 共 {total} 道题目, 并发数={self.concurrency}")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {}
            for p in problems:
                pid = p.get("problem_id", "")
                c = constraint_map.get(pid)
                futures[executor.submit(self.build_single, p, c)] = p

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    problem = futures[future]
                    print(f"[错误] {problem.get('problem_id', '?')}: {e}")

                if (i + 1) % 100 == 0:
                    success = sum(1 for r in results if "error" not in r)
                    print(f"  进度: {i + 1}/{total} (成功: {success})")

        success = sum(1 for r in results if "error" not in r)
        print(f"[完成] 成功: {success}/{total}")
        return results

    def save(self, patterns: list[dict], filepath: str = PATTERNS_FILE):
        """保存测试模式"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
        print(f"[保存] 测试模式 → {filepath}")

    def load(self, filepath: str = PATTERNS_FILE) -> list[dict]:
        """加载测试模式"""
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── 模式统计 ──

    @staticmethod
    def get_pattern_statistics(patterns: list[dict]) -> dict:
        """统计模式库信息"""
        stats = {
            "total_problems": len(patterns),
            "total_random": 0,
            "total_boundary": 0,
            "total_adversarial": 0,
            "error_types_distribution": {},
            "boundary_types_distribution": {},
            "tag_coverage": {},
        }

        for p in patterns:
            if "error" in p:
                continue
            pats = p.get("patterns", {})

            stats["total_random"] += len(pats.get("random", []))
            stats["total_boundary"] += len(pats.get("boundary", []))
            stats["total_adversarial"] += len(pats.get("adversarial", []))

            for tag in p.get("tags", []):
                stats["tag_coverage"][tag] = stats["tag_coverage"].get(tag, 0) + 1

            # 统计边界类型
            for b in pats.get("boundary", []):
                bt = b.get("boundary_type", "unknown")
                stats["boundary_types_distribution"][bt] = \
                    stats["boundary_types_distribution"].get(bt, 0) + 1

            # 统计错误类型
            for pat_type in ["random", "boundary", "adversarial"]:
                for pat in pats.get(pat_type, []):
                    for et in pat.get("error_types_caught", []):
                        stats["error_types_distribution"][et] = \
                            stats["error_types_distribution"].get(et, 0) + 1

        return stats

    @staticmethod
    def get_patterns_by_tag(patterns: list[dict], tag: str) -> list[dict]:
        """按标签筛选模式"""
        return [
            p for p in patterns
            if tag in p.get("tags", [])
        ]

    @staticmethod
    def get_patterns_by_error_type(patterns: list[dict], error_type: str) -> list[dict]:
        """按错误类型筛选模式"""
        result = []
        for p in patterns:
            if "error" in p:
                continue
            pats = p.get("patterns", {})
            for pat_type in ["random", "boundary", "adversarial"]:
                for pat in pats.get(pat_type, []):
                    if error_type in pat.get("error_types_caught", []):
                        result.append({
                            "problem_id": p.get("problem_id"),
                            "pattern_type": pat_type,
                            "pattern": pat,
                        })
        return result

    @staticmethod
    def flatten_patterns(patterns: list[dict]) -> list[dict]:
        """将所有模式展开为扁平列表，便于向量化"""
        flat = []
        for p in patterns:
            if "error" in p:
                continue
            pid = p.get("problem_id", "")
            tags = p.get("tags", [])
            rating = p.get("rating")
            pats = p.get("patterns", {})

            for pat_type in ["random", "boundary", "adversarial"]:
                for pat in pats.get(pat_type, []):
                    pat["problem_id"] = pid
                    pat["tags"] = tags
                    pat["rating"] = rating
                    pat["pattern_type"] = pat_type
                    pat["uid"] = f"{pid}_{pat.get('pattern_id', '')}"
                    flat.append(pat)

        return flat

    @staticmethod
    def build_pattern_text(pattern: dict) -> str:
        """构建用于嵌入的模式文本"""
        parts = [
            f"Pattern: {pattern.get('pattern_id', '')}",
            f"Type: {pattern.get('pattern_type', '')}",
            f"Description: {pattern.get('description', '')}",
        ]

        gen_strat = pattern.get("generation_strategy", "")
        if gen_strat:
            parts.append(f"Strategy: {gen_strat}")

        example = pattern.get("example_input", "")
        if example:
            parts.append(f"Example Input: {example}")

        errors = pattern.get("error_types_caught", [])
        if errors:
            parts.append(f"Catches: {', '.join(errors)}")

        if pattern.get("pattern_type") == "boundary":
            parts.append(f"Boundary Type: {pattern.get('boundary_type', '')}")
        elif pattern.get("pattern_type") == "adversarial":
            parts.append(f"Attack Vector: {pattern.get('attack_vector', '')}")
            parts.append(f"Why Effective: {pattern.get('why_effective', '')}")

        return "\n".join(parts)
