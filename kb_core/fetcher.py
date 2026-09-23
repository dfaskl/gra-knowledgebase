"""
Codeforces 数据获取模块
- API: problemset.problems 获取题目元数据
- Web: 爬取题目完整描述页面
"""
import json
import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional

from config import (
    CF_PROBLEMS_URL,
    CF_PROBLEM_PAGE,
    CF_RATE_LIMIT,
    CF_MAX_RETRIES,
    CF_RETRY_BACKOFF,
    FETCH_DELAY,
    CHECKPOINT_INTERVAL,
    REQUEST_TIMEOUT,
    PROBLEMS_META_FILE,
    STATEMENTS_DIR,
    TAG_ZH_MAP,
)


class CodeforcesFetcher:
    """Codeforces 数据获取器，支持断点续传"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        })
        self._last_request_time = 0
        self._rate_limit = CF_RATE_LIMIT

    # ═══════════════════════════════════════════════════════
    # Step 1: 获取题目元数据
    # ═══════════════════════════════════════════════════════

    @retry(
        stop=stop_after_attempt(CF_MAX_RETRIES),
        wait=wait_exponential(multiplier=CF_RETRY_BACKOFF, min=2, max=60),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
    )
    def fetch_problems_meta(self) -> list[dict]:
        """从 Codeforces API 获取全部题目元数据"""
        self._respect_rate_limit()
        resp = self.session.get(CF_PROBLEMS_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data["status"] != "OK":
            raise ValueError(f"API returned status: {data['status']}")

        problems = data["result"]["problems"]
        problem_stats = data["result"].get("problemStatistics", [])

        # 构建 solvedCount 查找表 (problemStatistics 只提供 solvedCount)
        solved_map = {}
        for ps in problem_stats:
            key = f"{ps['contestId']}{ps['index']}"
            solved_map[key] = ps.get("solvedCount", 0)

        results = []
        for p in problems:
            key = f"{p['contestId']}{p['index']}"
            entry = {
                "contestId": p.get("contestId"),
                "index": p.get("index"),
                "name": p.get("name", ""),
                "tags": p.get("tags", []),
                "rating": p.get("rating"),  # rating 是 Problem 对象的直接属性
                "solvedCount": solved_map.get(key, 0),
                "problem_id": key,
            }
            results.append(entry)

        print(f"[API] 获取到 {len(results)} 道题目元数据")
        return results

    def save_problems_meta(self, problems: list[dict], filepath: str = PROBLEMS_META_FILE):
        """保存题目元数据到 JSON 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)
        print(f"[保存] 题目元数据 → {filepath}")

    def load_problems_meta(self, filepath: str = PROBLEMS_META_FILE) -> list[dict]:
        """加载已保存的题目元数据"""
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ═══════════════════════════════════════════════════════
    # Step 2: 爬取题目页面
    # ═══════════════════════════════════════════════════════

    def problem_url(self, contest_id: int, index: str) -> str:
        """构建题目页面 URL"""
        return CF_PROBLEM_PAGE.format(contestId=contest_id, index=index)

    @retry(
        stop=stop_after_attempt(CF_MAX_RETRIES),
        wait=wait_exponential(multiplier=CF_RETRY_BACKOFF, min=2, max=60),
        retry=retry_if_exception_type((requests.RequestException,)),
    )
    def fetch_problem_statement(self, contest_id: int, index: str) -> Optional[str]:
        """爬取单个题目的 HTML 页面"""
        url = self.problem_url(contest_id, index)
        self._respect_rate_limit()
        resp = self.session.get(url, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 404:
            print(f"[跳过] 题目 {contest_id}{index} 不存在 (404)")
            return None

        resp.raise_for_status()
        return resp.text

    def extract_problem_content(self, html: str) -> dict:
        """从 HTML 中提取题目核心内容 (修复版: 正确分离 header/description/input/output)"""
        soup = BeautifulSoup(html, "lxml")

        result = {
            "title": "",
            "time_limit": "",
            "memory_limit": "",
            "description": "",
            "input_spec": "",
            "output_spec": "",
            "sample_tests": [],
            "note": "",
            "raw_html": html,
        }

        problem_statement = soup.find("div", class_="problem-statement")
        if not problem_statement:
            return result

        # ── Header: 标题 + 时间/内存限制 ──
        header_div = problem_statement.find("div", class_="header")
        if header_div:
            title_tag = header_div.find("div", class_="title")
            if title_tag:
                result["title"] = title_tag.get_text(strip=True)

            time_tag = header_div.find("div", class_="time-limit")
            if time_tag:
                text = time_tag.get_text(" ", strip=True)
                result["time_limit"] = text.replace("time limit per test", "").strip()

            mem_tag = header_div.find("div", class_="memory-limit")
            if mem_tag:
                text = mem_tag.get_text(" ", strip=True)
                result["memory_limit"] = text.replace("memory limit per test", "").strip()

        # ── Description: header 之后、input-specification 之前的内容 ──
        # Codeforces 题目结构中, description 是最外层 problem-statement 下
        # 夹在 header 和 input-specification 之间的 div (无 class 或有通用 class)
        for child in problem_statement.find_all("div", recursive=False):
            cls = child.get("class", [])
            # 跳过 header, input-spec, output-spec, sample-tests, note
            if "header" in cls:
                continue
            if "input-specification" in cls:
                # 提取输入说明 (去除 section-title)
                section_title = child.find("div", class_="section-title")
                if section_title:
                    section_title.decompose()
                result["input_spec"] = child.get_text("\n", strip=True)
                continue
            if "output-specification" in cls:
                section_title = child.find("div", class_="section-title")
                if section_title:
                    section_title.decompose()
                result["output_spec"] = child.get_text("\n", strip=True)
                continue
            if "sample-tests" in cls:
                continue  # 单独处理
            if "note" in cls:
                section_title = child.find("div", class_="section-title")
                if section_title:
                    section_title.decompose()
                result["note"] = child.get_text("\n", strip=True)
                continue
            # 其余 div (无特定 class) → description
            if not result["description"] and not cls:
                text = child.get_text("\n", strip=True)
                if text:
                    result["description"] = text

        # 如果还是没找到 description, 尝试从 <p> 标签提取
        if not result["description"]:
            # 排除 header 下的 <p> 和 input/output/note 下的 <p>
            exclude_parents = ["header", "input-specification", "output-specification",
                              "sample-tests", "sample-test", "note"]
            for p in problem_statement.find_all("p"):
                parent = p.parent
                skip = False
                while parent and parent != problem_statement:
                    if parent.get("class") and any(c in exclude_parents for c in parent.get("class", [])):
                        skip = True
                        break
                    parent = parent.parent
                if skip:
                    continue
                result["description"] += p.get_text("\n", strip=True) + "\n"
            result["description"] = result["description"].strip()

        # ── 示例测试用例 ──
        sample_div = problem_statement.find("div", class_="sample-test")
        if sample_div:
            inputs = sample_div.find_all("div", class_="input")
            outputs = sample_div.find_all("div", class_="output")
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                pre_in = inp.find("pre")
                pre_out = out.find("pre")
                result["sample_tests"].append({
                    "index": i,
                    "input": pre_in.get_text("\n") if pre_in else inp.get_text("\n", strip=True),
                    "output": pre_out.get_text("\n") if pre_out else out.get_text("\n", strip=True),
                })

        # ── Note ──
        if not result["note"]:
            note_div = problem_statement.find("div", class_="note")
            if note_div:
                section_title = note_div.find("div", class_="section-title")
                if section_title:
                    section_title.decompose()
                result["note"] = note_div.get_text("\n", strip=True)

        return result

    def save_statement(self, problem_id: str, content: dict, statements_dir: str = STATEMENTS_DIR):
        """保存单个题目的结构化内容"""
        os.makedirs(statements_dir, exist_ok=True)
        filepath = os.path.join(statements_dir, f"{problem_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    def load_statement(self, problem_id: str, statements_dir: str = STATEMENTS_DIR) -> Optional[dict]:
        """加载已保存的题目内容"""
        filepath = os.path.join(statements_dir, f"{problem_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ═══════════════════════════════════════════════════════
    # 批量爬取
    # ═══════════════════════════════════════════════════════

    def fetch_all_statements(
        self,
        problems: list[dict],
        resume: bool = True,
        start_from: int = 0,
        max_problems: Optional[int] = None,
    ) -> list[dict]:
        """
        批量爬取题目页面

        Args:
            problems: 题目元数据列表
            resume: 是否断点续传（跳过已有文件）
            start_from: 起始索引
            max_problems: 最大爬取数量，None 表示全部
        """
        results = []
        total = len(problems)
        end = min(start_from + max_problems, total) if max_problems else total

        print(f"[批量爬取] 共 {end - start_from} 道题目 (从索引 {start_from} 开始)")

        for i in range(start_from, end):
            p = problems[i]
            pid = p["problem_id"]

            # 断点续传
            if resume and self.load_statement(pid):
                continue

            contest_id = p["contestId"]
            index = p["index"]

            try:
                html = self.fetch_problem_statement(contest_id, index)
                if html is None:
                    p["_fetch_status"] = "skipped"
                    results.append(p)
                    continue

                content = self.extract_problem_content(html)
                content["problem_id"] = pid
                content["contestId"] = contest_id
                content["index"] = index
                content["tags"] = p.get("tags", [])
                content["rating"] = p.get("rating")
                content["_fetch_status"] = "ok"

                self.save_statement(pid, content)
                results.append(content)
                p["_fetch_status"] = "ok"

                if (i - start_from + 1) % 50 == 0:
                    print(f"  进度: {i - start_from + 1}/{end - start_from} "
                          f"({100*(i-start_from+1)/(end-start_from):.1f}%)")

            except Exception as e:
                print(f"[错误] {pid}: {type(e).__name__}: {e}")
                p["_fetch_status"] = f"error: {e}"
                results.append(p)

            # Checkpoint
            if (i - start_from + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(results, i)

            time.sleep(FETCH_DELAY)

        print(f"[完成] 成功: {sum(1 for r in results if r.get('_fetch_status') == 'ok')}, "
              f"跳过: {sum(1 for r in results if r.get('_fetch_status') == 'skipped')}, "
              f"失败: {sum(1 for r in results if 'error' in str(r.get('_fetch_status', '')))}")
        return results

    def _save_checkpoint(self, results: list[dict], idx: int):
        """保存断点 checkpoint"""
        checkpoint_path = os.path.join(
            os.path.dirname(PROBLEMS_META_FILE), f"fetch_checkpoint_{idx}.json"
        )
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"last_index": idx, "count": len(results)}, f)

    def _respect_rate_limit(self):
        """遵守 API 限速"""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self._rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def translate_tags(tags: list[str]) -> list[dict]:
        """将英文标签翻译为中文"""
        return [
            {"en": tag, "zh": TAG_ZH_MAP.get(tag, tag)}
            for tag in tags
        ]

    @staticmethod
    def get_problems_by_tag(problems: list[dict], tag: str) -> list[dict]:
        """按标签筛选题目"""
        return [p for p in problems if tag in p.get("tags", [])]

    @staticmethod
    def get_tag_statistics(problems: list[dict]) -> dict:
        """统计标签分布"""
        stats = {}
        for p in problems:
            for tag in p.get("tags", []):
                stats[tag] = stats.get(tag, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))
