"""
文本预处理模块
- HTML → 纯文本转换
- LaTeX 公式处理
- 文本归一化
- 中英文混合处理
"""
import json
import os
import re
import unicodedata
from html import unescape
from typing import Optional

from config import PROCESSED_DIR, PROBLEMS_CLEAN_FILE, STATEMENTS_DIR


class Preprocessor:
    """题目文本预处理器"""

    # ── LaTeX 模式 ──
    # 行内公式: $...$ 或 \(...\)
    LATEX_INLINE = re.compile(r'\$(.+?)\$|\\\((.+?)\\\)')
    # 独立公式: $$...$$ 或 \[...\]
    LATEX_DISPLAY = re.compile(r'\$\$(.+?)\$\$|\\\[(.+?)\\\]', re.DOTALL)
    # Codeforces 的 LaTeX 环境标记
    LATEX_ENV = re.compile(r'\\begin\{([^}]+)\}(.+?)\\end\{\1\}', re.DOTALL)

    # ── 常见 LaTeX 命令 → Unicode ──
    LATEX_UNICODE_MAP = {
        r"\le": "≤", r"\ge": "≥", r"\ne": "≠",
        r"\leq": "≤", r"\geq": "≥",
        r"\times": "×", r"\cdot": "·",
        r"\dots": "…", r"\ldots": "…",
        r"\cdots": "⋯",
        r"\to": "→", r"\rightarrow": "→",
        r"\leftarrow": "←",
        r"\Rightarrow": "⇒", r"\Leftrightarrow": "⇔",
        r"\infty": "∞",
        r"\sum": "Σ", r"\prod": "Π",
        r"\int": "∫",
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
        r"\delta": "δ", r"\epsilon": "ε",
        r"\pi": "π", r"\theta": "θ",
        r"\sigma": "σ", r"\lambda": "λ",
        r"\mu": "μ", r"\omega": "ω",
        r"\pm": "±", r"\mp": "∓",
        r"\sqrt": "√",
        r"\equiv": "≡", r"\approx": "≈",
        r"\subset": "⊂", r"\supset": "⊃",
        r"\subseteq": "⊆", r"\supseteq": "⊇",
        r"\in": "∈", r"\notin": "∉",
        r"\cup": "∪", r"\cap": "∩",
        r"\land": "∧", r"\lor": "∨",
        r"\forall": "∀", r"\exists": "∃",
        r"\emptyset": "∅",
        r"\text": "", r"\texttt": "", r"\textbf": "", r"\textit": "",
    }

    @classmethod
    def clean_html(cls, text: str) -> str:
        """清理 HTML 标签和实体"""
        # HTML 实体解码
        text = unescape(text)
        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除多余的 class/style 属性残留
        text = re.sub(r'\{[^}]*\}', '', text)
        return text.strip()

    @classmethod
    def normalize_latex(cls, text: str) -> str:
        """
        将 LaTeX 公式转换为可读形式
        保留关键数学符号，替换为 Unicode 等价物
        """
        # 替换 LaTeX 环境为标记
        text = cls.LATEX_ENV.sub(r'[数学公式: \1]', text)

        # 替换独立公式为标记
        text = cls.LATEX_DISPLAY.sub(r'[数学公式]', text)

        # 处理行内公式
        def replace_inline(match):
            formula = match.group(1) or match.group(2) or ""
            # 替换常见命令
            for cmd, uni in cls.LATEX_UNICODE_MAP.items():
                formula = formula.replace(cmd, uni)
            # 去除多余花括号和反斜杠
            formula = re.sub(r'[{}]', '', formula)
            return formula.strip()

        text = cls.LATEX_INLINE.sub(replace_inline, text)

        # 清理残留的 LaTeX 命令
        text = re.sub(r'\\[a-zA-Z]+', '', text)

        return text

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """通用文本归一化"""
        # Unicode 归一化 (全角 → 半角)
        text = unicodedata.normalize('NFKC', text)
        # 多余空白
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除零宽字符
        text = re.sub(r'[​‌‍‎‏﻿]', '', text)
        return text.strip()

    @classmethod
    def clean_problem_statement(cls, raw_content: dict) -> dict:
        """
        清洗单道题目的完整内容
        输入: 爬取的原始内容 dict
        输出: 清洗后的结构化 dict
        """
        clean = {
            "problem_id": raw_content.get("problem_id", ""),
            "contestId": raw_content.get("contestId"),
            "index": raw_content.get("index"),
            "title": cls.normalize_text(raw_content.get("title", "")),
            "time_limit": raw_content.get("time_limit", ""),
            "memory_limit": raw_content.get("memory_limit", ""),
            "tags": raw_content.get("tags", []),
            "rating": raw_content.get("rating"),
        }

        # 清洗描述
        for field in ["description", "input_spec", "output_spec", "note"]:
            raw = raw_content.get(field, "")
            if raw:
                text = cls.clean_html(raw)
                text = cls.normalize_latex(text)
                text = cls.normalize_text(text)
                clean[field] = text
            else:
                clean[field] = ""

        # 清洗示例测试用例
        clean["sample_tests"] = []
        for st in raw_content.get("sample_tests", []):
            clean["sample_tests"].append({
                "index": st.get("index"),
                "input": cls.normalize_text(st.get("input", "")),
                "output": cls.normalize_text(st.get("output", "")),
            })

        return clean

    @classmethod
    def batch_clean(cls, statements_dir: str = STATEMENTS_DIR) -> list[dict]:
        """批量清洗所有已爬取的题目"""
        results = []
        if not os.path.exists(statements_dir):
            print(f"[预处理] 目录不存在: {statements_dir}")
            return results

        files = sorted([f for f in os.listdir(statements_dir) if f.endswith(".json")])
        print(f"[预处理] 共 {len(files)} 个题目文件待清洗")

        for i, fname in enumerate(files):
            filepath = os.path.join(statements_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                clean = cls.clean_problem_statement(raw)
                results.append(clean)

                if (i + 1) % 500 == 0:
                    print(f"  进度: {i + 1}/{len(files)}")
            except Exception as e:
                print(f"[错误] {fname}: {e}")

        print(f"[完成] 清洗 {len(results)} 道题目")
        return results

    @classmethod
    def save_clean(cls, problems: list[dict], filepath: str = PROBLEMS_CLEAN_FILE):
        """保存清洗后的数据"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"[保存] 清洗后数据 → {filepath}")

    @classmethod
    def load_clean(cls, filepath: str = PROBLEMS_CLEAN_FILE) -> list[dict]:
        """加载清洗后的数据"""
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def build_combined_text(cls, problem: dict) -> str:
        """
        构建用于嵌入的合并文本
        将题目的所有文本字段合并为一个字符串
        """
        parts = []

        title = problem.get("title", "")
        if title:
            parts.append(f"题目: {title}")

        tags = problem.get("tags", [])
        if tags:
            parts.append(f"标签: {', '.join(tags)}")

        rating = problem.get("rating")
        if rating:
            parts.append(f"难度: {rating}")

        desc = problem.get("description", "")
        if desc:
            parts.append(f"描述: {desc}")

        input_spec = problem.get("input_spec", "")
        if input_spec:
            parts.append(f"输入: {input_spec}")

        output_spec = problem.get("output_spec", "")
        if output_spec:
            parts.append(f"输出: {output_spec}")

        note = problem.get("note", "")
        if note:
            parts.append(f"备注: {note}")

        # 示例
        samples = problem.get("sample_tests", [])
        if samples:
            sample_texts = []
            for s in samples[:3]:  # 最多取前3个示例
                sample_texts.append(f"样例输入: {s.get('input', '')}")
                sample_texts.append(f"样例输出: {s.get('output', '')}")
            parts.append("\n".join(sample_texts))

        return "\n\n".join(parts)
