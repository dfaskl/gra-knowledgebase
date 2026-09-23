"""
向量存储与检索模块
- 使用 TF-IDF + SVD 降维生成文本向量（无需外部API或模型下载）
- 使用 ChromaDB 存储和检索
"""
import json
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import (
    EMBEDDING_DIM,
    CHROMA_DIR,
    CHROMA_COLLECTIONS,
)


class VectorStore:
    """向量存储与检索引擎 (基于 TF-IDF + SVD)"""

    def __init__(self, chroma_dir: str = None):
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.embedding_dim = EMBEDDING_DIM  # 512
        self.vectorizer = None   # TfidfVectorizer (fit后)
        self.svd = None          # TruncatedSVD (fit后)
        self._fitted = False
        self._active_model = None

        os.makedirs(self.chroma_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    # ═══════════════════════════════════════════════════════
    # TF-IDF 嵌入
    # ═══════════════════════════════════════════════════════

    def fit(self, texts: list[str]):
        """在全体语料上训练 TF-IDF + SVD"""
        print(f"[向量存储] 训练 TF-IDF 向量化器 (语料: {len(texts)} 条)...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # SVD 降维到目标维度
        n_components = min(self.embedding_dim, tfidf_matrix.shape[1] - 1)
        print(f"[向量存储] SVD 降维: {tfidf_matrix.shape[1]} -> {n_components}")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)

        self._fitted = True
        print(f"[向量存储] 训练完成")

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量生成 TF-IDF + SVD 向量"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 训练向量化器")

        tfidf = self.vectorizer.transform(texts)
        reduced = self.svd.transform(tfidf)

        # L2 归一化
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return reduced / norms

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _model_path(self, collection_name: str) -> str:
        return os.path.join(self.chroma_dir, f"tfidf_{collection_name}.pkl")

    def save_model(self, collection_name: str):
        """Save the TF-IDF/SVD model paired with one collection."""
        path = self._model_path(collection_name)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "fitted": self._fitted,
            }, f)
        self._active_model = collection_name
        print(f"[saved] TF-IDF model for {collection_name}: {path}")

    def load_model(self, collection_name: str) -> bool:
        """Load the TF-IDF/SVD model paired with one collection."""
        if self._active_model == collection_name and self._fitted:
            return True
        path = self._model_path(collection_name)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]
        self._fitted = data.get("fitted", True)
        self._active_model = collection_name
        return True

    # ═══════════════════════════════════════════════════════
    # Collection 管理
    # ═══════════════════════════════════════════════════════

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        return self.chroma_client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self, name: str):
        try:
            self.chroma_client.delete_collection(name)
        except Exception:
            pass
        return self.get_or_create_collection(name)

    # ═══════════════════════════════════════════════════════
    # 批量索引
    # ═══════════════════════════════════════════════════════

    def index_problems(self, problems: list[dict], batch_size: int = 500):
        from kb_core.preprocessor import Preprocessor

        collection = self.reset_collection(CHROMA_COLLECTIONS["problems"])
        total = len(problems)
        print(f"[向量化] 题目: {total} 道")

        # 先收集全部文本用于 fit
        all_texts = [Preprocessor.build_combined_text(p) for p in problems]
        self.fit(all_texts)

        # 分批入库
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = problems[start:end]
            texts = all_texts[start:end]
            embeddings = self.encode(texts)

            ids = [p.get("problem_id", f"p_{i}") for i, p in enumerate(batch)]
            metadatas = [
                {"title": p.get("title", "")[:200],
                 "tags": ", ".join(p.get("tags", [])),
                 "rating": p.get("rating") or 0,
                 "problem_id": p.get("problem_id", "")}
                for p in batch
            ]

            collection.add(ids=ids, embeddings=embeddings.tolist(),
                          documents=texts, metadatas=metadatas)

        self.save_model(CHROMA_COLLECTIONS["problems"])
        print(f"[完成] 题目向量化: {total} 条")

    def index_patterns(self, patterns: list[dict], batch_size: int = 500):
        from kb_core.pattern_builder import PatternBuilder

        flat = PatternBuilder.flatten_patterns(patterns)
        collection = self.reset_collection(CHROMA_COLLECTIONS["patterns"])
        total = len(flat)
        print(f"[向量化] 测试模式: {total} 条")

        all_texts = [PatternBuilder.build_pattern_text(p) for p in flat]
        self.fit(all_texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = flat[start:end]
            texts = all_texts[start:end]
            embeddings = self.encode(texts)

            ids = [p.get("uid", f"pat_{i}") for i, p in enumerate(batch)]
            metadatas = [
                {"problem_id": p.get("problem_id", ""),
                 "pattern_type": p.get("pattern_type", ""),
                 "pattern_id": p.get("pattern_id", ""),
                 "error_types": ", ".join(p.get("error_types_caught", []))}
                for p in batch
            ]

            collection.add(ids=ids, embeddings=embeddings.tolist(),
                          documents=texts, metadatas=metadatas)

        self.save_model(CHROMA_COLLECTIONS["patterns"])
        print(f"[完成] 模式向量化: {total} 条")

    def index_constraints(self, constraints: list[dict], batch_size: int = 500):
        collection = self.reset_collection(CHROMA_COLLECTIONS["constraints"])
        valid = [c for c in constraints if "error" not in c]
        total = len(valid)
        print(f"[向量化] 约束条件: {total} 条")

        all_texts = []
        for c in valid:
            parts = [f"Problem: {c.get('problem_id', '')}"]
            inp = c.get("input_format", {})
            if inp:
                parts.append(f"Input: {json.dumps(inp, ensure_ascii=False)}")
            for ec in (c.get("constraints") or {}).get("explicit", []):
                parts.append(f"Constraint: {json.dumps(ec, ensure_ascii=False)}")
            for ic in c.get("implicit_constraints", []):
                parts.append(f"Implicit: {ic.get('description', '')}")
            for bc in c.get("boundary_conditions", []):
                parts.append(f"Boundary: {json.dumps(bc, ensure_ascii=False)}")
            all_texts.append("\n".join(parts))

        self.fit(all_texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = valid[start:end]
            texts = all_texts[start:end]
            embeddings = self.encode(texts)
            ids = [c.get("problem_id", f"c_{i}") for i, c in enumerate(batch)]
            metadatas = [
                {"problem_id": c.get("problem_id", ""),
                 "num_constraints": len((c.get("constraints") or {}).get("explicit", [])),
                 "num_implicit": len(c.get("implicit_constraints") or [])}
                for c in batch
            ]
            collection.add(ids=ids, embeddings=embeddings.tolist(),
                          documents=texts, metadatas=metadatas)

        self.save_model(CHROMA_COLLECTIONS["constraints"])
        print(f"[完成] 约束向量化: {total} 条")

    def index_error_patterns(self, patterns: list[dict], batch_size: int = 500):
        from kb_core.pattern_builder import PatternBuilder

        collection = self.reset_collection(CHROMA_COLLECTIONS["error_patterns"])
        error_entries = []
        for p in patterns:
            if "error" in p:
                continue
            for pat in p.get("patterns", {}).get("adversarial", []):
                entry = dict(pat)
                entry["problem_id"] = p.get("problem_id", "")
                entry["pattern_type"] = "adversarial"
                entry["uid"] = f"{p.get('problem_id', '')}_{pat.get('pattern_id', '')}"
                error_entries.append(entry)

        total = len(error_entries)
        print(f"[向量化] 错误模式: {total} 条")

        all_texts = [PatternBuilder.build_pattern_text(e) for e in error_entries]
        self.fit(all_texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = error_entries[start:end]
            texts = all_texts[start:end]
            embeddings = self.encode(texts)
            ids = [e.get("uid", f"err_{i}") for i, e in enumerate(batch)]
            metadatas = [
                {"problem_id": e.get("problem_id", ""),
                 "target_error": e.get("target_error", ""),
                 "attack_vector": e.get("attack_vector", "")[:500]}
                for e in batch
            ]
            collection.add(ids=ids, embeddings=embeddings.tolist(),
                          documents=texts, metadatas=metadatas)

        self.save_model(CHROMA_COLLECTIONS["error_patterns"])
        print(f"[完成] 错误模式向量化: {total} 条")

    # ═══════════════════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════════════════

    def _search(self, collection_name: str, query_text: str,
                n_results: int = 10, where_filter: dict = None) -> list[dict]:
        if not self.load_model(collection_name):
            raise FileNotFoundError(
                f"Missing vectorizer for '{collection_name}'. "
                "Run scripts/06_build_vectors.py to rebuild the vector store."
            )
        collection = self.get_or_create_collection(collection_name)
        query_embedding = self.encode_single(query_text).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "documents", "distances"],
        )
        return self._format_results(results)

    def search_similar_problems(self, query_text: str, n_results: int = 10,
                                filter_tags: list[str] = None) -> list[dict]:
        where_filter = None
        if filter_tags:
            conditions = [{"tags": {"$contains": tag}} for tag in filter_tags]
            where_filter = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        return self._search(CHROMA_COLLECTIONS["problems"], query_text, n_results, where_filter)

    def search_similar_patterns(self, query_text: str, pattern_type: str = None,
                                n_results: int = 10) -> list[dict]:
        where_filter = {"pattern_type": pattern_type} if pattern_type else None
        return self._search(CHROMA_COLLECTIONS["patterns"], query_text, n_results, where_filter)

    def search_similar_constraints(self, query_text: str, n_results: int = 10) -> list[dict]:
        return self._search(CHROMA_COLLECTIONS["constraints"], query_text, n_results)

    def search_error_patterns(self, query_text: str, target_error: str = None,
                              n_results: int = 10) -> list[dict]:
        where_filter = {"target_error": target_error} if target_error else None
        return self._search(CHROMA_COLLECTIONS["error_patterns"], query_text, n_results, where_filter)

    def _format_results(self, results: dict) -> list[dict]:
        if not results.get("ids") or not results["ids"][0]:
            return []

        formatted = []
        ids = results["ids"][0]
        distances = results.get("distances", [[1.0] * len(ids)])[0]
        metadatas = results.get("metadatas", [[{}] * len(ids)])[0]
        documents = results.get("documents", [[""] * len(ids)])[0]

        for i, pid in enumerate(ids):
            formatted.append({
                "id": pid,
                "score": 1.0 - float(distances[i]),
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "document": documents[i] if i < len(documents) else "",
            })
        return formatted

    # ═══════════════════════════════════════════════════════
    # 综合检索 (第四章接口)
    # ═══════════════════════════════════════════════════════

    def retrieve_knowledge(self, problem_description: str, top_k: int = 5) -> dict:
        knowledge = {
            "similar_problems": self.search_similar_problems(problem_description, n_results=top_k),
            "similar_patterns": self.search_similar_patterns(problem_description, n_results=top_k),
            "similar_constraints": self.search_similar_constraints(problem_description, n_results=top_k),
            "relevant_error_patterns": self.search_error_patterns(problem_description, n_results=top_k),
        }
        entries = []
        for item in knowledge["similar_problems"]:
            entries.append({"type": "problem", "id": item["id"], "score": item["score"],
                           "content": item["document"][:500] if item["document"] else ""})
        for item in knowledge["similar_patterns"]:
            entries.append({"type": "pattern", "id": item["id"], "score": item["score"],
                           "content": item["document"][:500] if item["document"] else "",
                           "pattern_type": item.get("metadata", {}).get("pattern_type", "")})
        entries.sort(key=lambda x: x["score"], reverse=True)
        knowledge["top_knowledge_entries"] = entries[:top_k]
        return knowledge
