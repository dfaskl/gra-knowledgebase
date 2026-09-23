"""
约束条件提取模块
使用 LLM 从题目描述中结构化提取:
- 输入/输出格式
- 显式约束条件
- 隐含约束条件
- 边界条件
- 特殊用例
"""
import json
import os
import time
import asyncio
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
    CONSTRAINTS_FILE,
)


EXTRACTION_PROMPT = """You are an expert in competitive programming problem analysis. Given a problem statement from Codeforces, extract the following structured information.

## Problem Statement
{problem_text}

## Output Format
Return a JSON object with the following structure. All fields must be in English:

```json
{{
  "problem_id": "string (e.g. '4A')",
  "input_format": {{
    "description": "brief description of input format in English",
    "format_type": "single_line | multi_line | matrix | tree | graph | undetermined",
    "fields": [
      {{
        "name": "variable name",
        "type": "integer | float | string | list_int | list_str | matrix_int | tree_structure | graph_structure",
        "dimension": "description of dimension if list/matrix (e.g. 'length n')",
        "position": "1-based position in input"
      }}
    ]
  }},
  "output_format": {{
    "description": "brief description of output format in English",
    "format_type": "single_line | multi_line | yes_no | number | list | undetermined"
  }},
  "constraints": {{
    "explicit": [
      {{
        "variable": "variable name",
        "lower_bound": "minimum value (number or null)",
        "upper_bound": "maximum value (number or null)",
        "type": "integer | float | string",
        "description": "constraint description"
      }}
    ],
    "time_limit_seconds": "number or null",
    "memory_limit_mb": "number or null"
  }},
  "implicit_constraints": [
    {{
      "description": "description of implicit constraint that is not explicitly stated but logically follows from the problem",
      "type": "logical_constraint | edge_case | domain_knowledge | efficiency_hint",
      "triggered_by": "what part of the problem implies this constraint"
    }}
  ],
  "boundary_conditions": [
    {{
      "variable": "variable name",
      "values": ["list of specific boundary values to test"],
      "condition": "what boundary this tests (e.g. min, max, zero, empty)"
    }}
  ],
  "special_cases": [
    {{
      "description": "description of special case",
      "test_values": "specific values that trigger this case",
      "expected_behavior": "how the solution should handle this case",
      "common_mistakes": "what common mistakes programmers make with this case"
    }}
  ],
  "problem_type_analysis": {{
    "category": "primary algorithm category",
    "sub_category": "more specific sub-category",
    "key_operations": ["list of key operations needed"],
    "common_pitfalls": ["list of common pitfalls for this problem type"]
  }}
}}
```

## Important Rules
1. Extract ALL numbers precisely as they appear in the constraints section
2. For implicit constraints, think about: what input values could break a naive solution? What edge cases exist?
3. Boundary values should include: min constraint, max constraint, min-1 (if possible), max+1 (if possible), 0, 1, empty/null cases
4. Special cases should focus on: cases where standard algorithm fails, cases requiring special handling
5. If a field cannot be determined, use null or empty array [] rather than guessing

Return ONLY valid JSON, no markdown code blocks, no additional text."""


class ConstraintExtractor:
    """基于 LLM 的约束条件提取器"""

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
        self.temperature = LLM_TEMPERATURE
        self.concurrency = LLM_CONCURRENCY
        self.max_retries = LLM_MAX_RETRIES

    def build_problem_text(self, problem: dict) -> str:
        """构建发送给 LLM 的题目文本"""
        parts = []

        title = problem.get("title", "")
        if title:
            parts.append(f"# Problem: {title}")

        tags = problem.get("tags", [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        rating = problem.get("rating")
        if rating:
            parts.append(f"Difficulty Rating: {rating}")

        desc = problem.get("description", "")
        if desc:
            parts.append(f"## Description\n{desc}")

        input_spec = problem.get("input_spec", "")
        if input_spec:
            parts.append(f"## Input\n{input_spec}")

        output_spec = problem.get("output_spec", "")
        if output_spec:
            parts.append(f"## Output\n{output_spec}")

        note = problem.get("note", "")
        if note:
            parts.append(f"## Note\n{note}")

        samples = problem.get("sample_tests", [])
        if samples:
            sample_lines = ["## Examples"]
            for s in samples[:3]:
                sample_lines.append(f"Input:\n{s.get('input', '')}")
                sample_lines.append(f"Output:\n{s.get('output', '')}")
            parts.append("\n".join(sample_lines))

        return "\n\n".join(parts)

    def extract_single(self, problem: dict) -> Optional[dict]:
        """对单道题目提取约束条件"""
        problem_text = self.build_problem_text(problem)
        pid = problem.get("problem_id", "unknown")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a competitive programming analysis expert. Always output valid JSON."},
                        {"role": "user", "content": EXTRACTION_PROMPT.format(problem_text=problem_text[:8000])},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                # 确保 problem_id 正确
                result["problem_id"] = pid
                return result

            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[JSON错误] {pid}: {e}")
                return {"problem_id": pid, "error": f"JSON parse error: {e}", "raw": content}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[LLM错误] {pid}: {e}")
                return {"problem_id": pid, "error": str(e)}

    def extract_batch(self, problems: list[dict]) -> list[dict]:
        """并发批量提取约束条件"""
        results = []
        total = len(problems)

        print(f"[约束提取] 共 {total} 道题目, 并发数={self.concurrency}")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(self.extract_single, p): p
                for p in problems
            }

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    problem = futures[future]
                    pid = problem.get("problem_id", "?")
                    print(f"[错误] {pid}: {e}")
                    results.append({"problem_id": pid, "error": str(e)})

                if (i + 1) % 100 == 0:
                    print(f"  进度: {i + 1}/{total} "
                          f"(成功: {sum(1 for r in results if 'error' not in r)})")

        success = sum(1 for r in results if "error" not in r)
        print(f"[完成] 成功: {success}/{total}")
        return results

    def save(self, constraints: list[dict], filepath: str = CONSTRAINTS_FILE):
        """保存约束提取结果"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(constraints, f, ensure_ascii=False, indent=2)
        print(f"[保存] 约束条件 → {filepath}")

    def load(self, filepath: str = CONSTRAINTS_FILE) -> list[dict]:
        """加载约束提取结果"""
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_constraint_by_id(constraints: list[dict], problem_id: str) -> Optional[dict]:
        """根据 problem_id 获取约束"""
        for c in constraints:
            if c.get("problem_id") == problem_id:
                return c
        return None
