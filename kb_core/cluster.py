"""
知识聚类与层次化组织模块
- 使用 TF-IDF + SVD 生成文本向量（无需外部API）
- HDBSCAN + UMAP 聚类
- 三层知识架构：全局层 / 领域层 / 实例层
"""
import json
import os
import pickle
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config import (
    EMBEDDING_DIM,
    CLUSTER_MIN_SIZE,
    CLUSTER_MIN_SAMPLES,
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    INDEX_FILE,
    TAG_ZH_MAP,
)


class KnowledgeClusterer:
    """知识聚类与层次化组织器"""

    def __init__(self):
        self.embedding_dim = EMBEDDING_DIM
        self.vectorizer = None
        self.svd = None
        self._fitted = False

    # ═══════════════════════════════════════════════════════
    # TF-IDF 嵌入
    # ═══════════════════════════════════════════════════════

    def fit_embeddings(self, texts: list[str]):
        """训练 TF-IDF + SVD"""
        print(f"[聚类] 训练 TF-IDF (语料: {len(texts)} 条)...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            stop_words='english', sublinear_tf=True,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.embedding_dim, tfidf.shape[1] - 1)
        print(f"[聚类] SVD: {tfidf.shape[1]} -> {n_comp}")
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True

    def embed_problems(self, problems: list[dict]) -> np.ndarray:
        """为题目列表生成嵌入向量"""
        from kb_core.preprocessor import Preprocessor
        texts = [Preprocessor.build_combined_text(p) for p in problems]
        print(f"[聚类] 生成 {len(texts)} 个题目嵌入...")
        self.fit_embeddings(texts)
        tfidf = self.vectorizer.transform(texts)
        reduced = self.svd.transform(tfidf)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return reduced / norms

    # ═══════════════════════════════════════════════════════
    # 聚类
    # ═══════════════════════════════════════════════════════

    def cluster(self, embeddings: np.ndarray) -> dict:
        import hdbscan
        import umap

        print(f"[聚类] UMAP: {embeddings.shape[1]} -> {UMAP_N_COMPONENTS}")
        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
            metric="cosine", random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)

        print(f"[聚类] HDBSCAN: min_size={CLUSTER_MIN_SIZE}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=CLUSTER_MIN_SIZE, min_samples=CLUSTER_MIN_SAMPLES,
            metric="euclidean", cluster_selection_epsilon=0.1,
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for l in labels if l == -1)
        print(f"[聚类] {n_clusters} 簇, {n_noise} 噪声")

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        centers = {}
        for label, indices in clusters.items():
            if label != -1:
                centers[label] = np.mean(reduced[indices], axis=0)

        return {
            "n_clusters": n_clusters, "n_noise": n_noise,
            "labels": labels.tolist(),
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_centers": {str(k): v.tolist() for k, v in centers.items()},
            "reduced_embeddings": reduced.tolist(),
        }

    # ═══════════════════════════════════════════════════════
    # 层次化组织
    # ═══════════════════════════════════════════════════════

    def build_hierarchy(self, problems, patterns, constraints, cluster_result) -> dict:
        return {
            "meta": {
                "total_problems": len(problems),
                "total_patterns": len(patterns),
                "total_constraints": len(constraints),
                "n_clusters": cluster_result.get("n_clusters", 0),
                "n_noise": cluster_result.get("n_noise", 0),
            },
            "global_layer": self._build_global_layer(problems, patterns, cluster_result),
            "domain_layer": self._build_domain_layer(problems),
            "instance_layer": self._build_instance_map(problems, patterns, constraints),
        }

    def _build_global_layer(self, problems, patterns, cluster_result) -> dict:
        from kb_core.pattern_builder import PatternBuilder
        global_knowledge = {
            "description": "跨题目的通用测试模式和知识",
            "clusters": {},
            "universal_patterns": self._extract_universal_patterns(patterns),
        }
        clusters = cluster_result.get("clusters", {})
        for label_str, indices in clusters.items():
            if int(label_str) == -1:
                continue
            reps = []
            for idx in indices[:5]:
                if idx < len(problems):
                    p = problems[idx]
                    reps.append({"problem_id": p.get("problem_id", ""), "title": p.get("title", ""),
                                "tags": p.get("tags", []), "rating": p.get("rating")})
            all_tags = []
            for idx in indices:
                if idx < len(problems):
                    all_tags.extend(problems[idx].get("tags", []))
            tag_counts = defaultdict(int)
            for t in all_tags:
                tag_counts[t] += 1
            common = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
            global_knowledge["clusters"][label_str] = {
                "size": len(indices), "representative_problems": reps,
                "common_tags": [{"tag": t, "count": c} for t, c in common],
            }
        return global_knowledge

    def _build_domain_layer(self, problems) -> dict:
        domain_knowledge = {"description": "按算法类型和标签分类", "domains": {}}
        tag_groups = defaultdict(list)
        for i, p in enumerate(problems):
            for tag in p.get("tags", []):
                tag_groups[tag].append(i)

        for tag, indices in sorted(tag_groups.items(), key=lambda x: -len(x[1])):
            ratings = []
            for idx in indices[:200]:
                if idx < len(problems):
                    r = problems[idx].get("rating")
                    if r:
                        ratings.append(r)
            sorted_ps = sorted(
                [problems[idx] for idx in indices if idx < len(problems)],
                key=lambda p: p.get("rating") or 0,
            )
            domain = {
                "tag_en": tag, "tag_zh": TAG_ZH_MAP.get(tag, tag),
                "problem_count": len(indices),
                "rating_range": {"min": min(ratings) if ratings else None,
                                "max": max(ratings) if ratings else None,
                                "avg": sum(ratings)/len(ratings) if ratings else None},
                "sample_problems": [{"problem_id": p.get("problem_id", ""),
                                    "title": p.get("title", ""),
                                    "rating": p.get("rating")}
                                   for p in sorted_ps[::max(1, len(sorted_ps)//5)][:5]],
            }
            domain_knowledge["domains"][tag] = domain
        return domain_knowledge

    def _build_instance_map(self, problems, patterns, constraints) -> dict:
        pattern_map = {p.get("problem_id", ""): p for p in patterns if "error" not in p}
        constraint_map = {c.get("problem_id", ""): c for c in constraints if "error" not in c}

        instance_map = {"description": "具体题目的实例级知识", "by_id": {}, "by_tag": {}, "by_rating_range": {}}
        for p in problems:
            pid = p.get("problem_id", "")
            instance_map["by_id"][pid] = {
                "problem_id": pid, "title": p.get("title", ""),
                "tags": p.get("tags", []), "rating": p.get("rating"),
                "description": p.get("description", "")[:500],
                "has_constraints": pid in constraint_map,
                "has_patterns": pid in pattern_map,
            }

        for tag in TAG_ZH_MAP:
            instance_map["by_tag"][tag] = []
        for p in problems:
            pid = p.get("problem_id", "")
            for tag in p.get("tags", []):
                if tag in instance_map["by_tag"]:
                    instance_map["by_tag"][tag].append(pid)

        ranges = [(0, 1200, "beginner"), (1200, 1600, "easy"), (1600, 2000, "medium"),
                  (2000, 2400, "hard"), (2400, 3000, "expert"), (3000, 9999, "grandmaster")]
        for _, _, label in ranges:
            instance_map["by_rating_range"][label] = []
        for p in problems:
            r = p.get("rating") or 0
            for low, high, label in ranges:
                if low <= r < high:
                    instance_map["by_rating_range"][label].append(p.get("problem_id", ""))
                    break
        return instance_map

    def _extract_universal_patterns(self, patterns) -> list[dict]:
        bt = defaultdict(int); et = defaultdict(int); av = defaultdict(int)
        for p in patterns:
            if "error" in p:
                continue
            for b in p.get("patterns", {}).get("boundary", []):
                x = b.get("boundary_type", ""); x and bt.__setitem__(x, bt[x]+1)
            for a in p.get("patterns", {}).get("adversarial", []):
                x = a.get("target_error", ""); x and et.__setitem__(x, et[x]+1)
                x = a.get("attack_vector", ""); x and av.__setitem__(x[:100], av[x[:100]]+1)
        return [
            {"type": "common_boundary", "patterns": [{"type": k, "freq": v} for k, v in sorted(bt.items(), key=lambda x: -x[1])[:20]]},
            {"type": "common_errors", "patterns": [{"type": k, "freq": v} for k, v in sorted(et.items(), key=lambda x: -x[1])[:20]]},
            {"type": "common_attacks", "patterns": [{"vector": k, "freq": v} for k, v in sorted(av.items(), key=lambda x: -x[1])[:20]]},
        ]

    # ═══════════════════════════════════════════════════════
    # 保存 & 检索
    # ═══════════════════════════════════════════════════════

    def save_index(self, hierarchy: dict, filepath: str = INDEX_FILE):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, ensure_ascii=False, indent=2)
        print(f"[保存] 知识索引 -> {filepath}")

    def load_index(self, filepath: str = INDEX_FILE) -> Optional[dict]:
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_by_tag(self, index, tag: str, limit: int = 20) -> list[str]:
        return index.get("instance_layer", {}).get("by_tag", {}).get(tag, [])[:limit]

    def get_by_rating(self, index, level: str, limit: int = 20) -> list[str]:
        return index.get("instance_layer", {}).get("by_rating_range", {}).get(level, [])[:limit]

    def get_problem_info(self, index, problem_id: str) -> Optional[dict]:
        return index.get("instance_layer", {}).get("by_id", {}).get(problem_id)

    def get_domain_info(self, index, tag: str) -> Optional[dict]:
        return index.get("domain_layer", {}).get("domains", {}).get(tag)
