# OJ Test-Pattern Knowledge Base

This repository contains the dataset, implementation, and persistent vector store accompanying the paper **“A Structured Knowledge Base for LLM-Based Test Case Generation in Online Judges.”** It organizes constraint specifications and reusable random, boundary, and adversarial testing patterns extracted from a stratified sample of Codeforces problems.

## Released artifacts

| Artifact | Count |
| --- | ---: |
| Codeforces problems | 2,500 |
| Algorithm tags | 38 |
| Difficulty tiers | 6 |
| Successfully parsed constraint specifications | 2,480 (99.2%) |
| Random test patterns | 4,947 |
| Boundary test patterns | 7,302 |
| Adversarial test patterns | 6,070 |
| Total test patterns | 18,319 |
| HDBSCAN clusters | 135 |

The repository includes the original collected statement records, normalized problem records, generated constraint and pattern annotations, the hierarchical knowledge index, all construction scripts, and a ready-to-query ChromaDB store.

## Repository layout

```text
.
├── kb_core/                  # Collection, preprocessing, extraction, indexing, and retrieval
├── scripts/                  # Reproducible seven-stage construction pipeline
├── data/
│   ├── raw/                  # Codeforces metadata, sample definition, and statement records
│   ├── processed/            # Clean problems, constraints, patterns, and hierarchy
│   └── chroma_db/            # Persistent four-collection vector store and TF-IDF/SVD models
├── query.py                  # Command-line retrieval example
├── requirements.txt
└── .env.example
```

The four ChromaDB collections are `cf_problems`, `cf_patterns`, `cf_constraints`, and `cf_error_patterns`. Each collection has its own fitted TF-IDF/SVD model so that released queries use the same feature space as the stored vectors.

## Quick start

Python 3.10 or later is recommended. Git LFS is required when cloning because large research artifacts are stored through LFS.

```bash
git lfs install
git clone https://github.com/dfaskl/gra-knowledgebase.git
cd gra-knowledgebase
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/validate_release.py
python query.py "Find boundary and overflow risks in a graph shortest-path problem" --top-k 5
```

The released vector store is sufficient for querying. No API key is needed for preprocessing, indexing, clustering, validation, or retrieval.

## Rebuild the pipeline

Run the stages from the repository root:

```bash
python scripts/01_fetch_metadata.py
python scripts/sample_problems.py
python scripts/02_fetch_statements.py --sample
python scripts/03_preprocess.py
python scripts/04_extract_constraints.py
python scripts/05_build_patterns.py
python scripts/06_build_vectors.py
python scripts/07_build_index.py
```

Stages 4 and 5 require an OpenAI-compatible chat-completions endpoint. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `LLM_MODEL`. The paper's released annotations were produced with DeepSeek-V4-Flash at temperature 0.0. Outputs can still vary across provider versions, even with deterministic sampling settings.

Stages 1–3 access Codeforces. Please respect the platform's terms and rate limits. The implementation includes request throttling, retries, and resumable checkpoints.

## Data schemas

- `problems_clean.json`: problem ID, contest metadata, title, limits, tags, rating, normalized statement sections, and sample tests.
- `constraints.json`: input/output formats, explicit and implicit constraints, boundary conditions, special cases, and problem-type analysis. Twenty records contain an `error` field for failed extractions and are retained for auditability.
- `patterns.json`: random, boundary, and adversarial pattern families with rationales and targeted failure types.
- `knowledge_index.json`: global cluster summaries, domain aggregates for all 38 tags, and problem-level instance entries.

Generated annotations are research artifacts rather than manually adjudicated ground truth. Parse success and field presence do not establish semantic correctness, and the paper does not claim an end-to-end test-generation improvement.

## Reproducibility notes

- Stratified sampling uses seed `42`.
- Vectorization uses TF-IDF with English stop words and 1–2 grams, followed by 512-dimensional truncated SVD and L2 normalization.
- Offline organization uses UMAP followed by HDBSCAN.
- Online retrieval uses cosine similarity over the four persistent ChromaDB collections.
- `scripts/validate_release.py` checks the headline artifact counts against the paper.

## Citation

If you use this repository, please cite the accompanying paper. Final venue metadata will be added after publication; until then, use the metadata in [`CITATION.cff`](CITATION.cff).

## License and provenance

The original source code is released under the MIT License. Codeforces problem statements and metadata remain subject to their original authors' and Codeforces' rights and terms. Generated annotations are provided for research and reproducibility; users are responsible for complying with applicable source-platform terms.
