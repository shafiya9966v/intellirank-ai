"""
core/jd_parser.py
─────────────────
Parses the raw Job Description text into a structured requirements dict.
No LLM calls. No network. Pure rule-based extraction using curated keyword
lists derived from the actual JD in the dataset.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

JD_TEXT = """
Senior AI Engineer – Founding Team | Redrob AI | Pune / Noida | 5-9 Years | 35-38 LPA

We are building India's AI Operating System. As a founding-team Senior AI Engineer,
you will own our intelligent candidate discovery engine end-to-end — from embedding
pipelines and vector retrieval to ranking models and evaluation frameworks.

Hard Requirements:
- 5-9 years experience, 3+ years in applied ML/AI at a product company
- Production experience with embedding-based retrieval: sentence-transformers, BGE, E5, OpenAI embeddings
- Production experience with vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch
- Strong Python — code quality, testing, code review
- Evaluation frameworks for ranking/retrieval: NDCG, MRR, MAP, A/B testing

Nice to Have:
- LLM fine-tuning: LoRA, QLoRA, PEFT
- Learning-to-rank: XGBoost LTR, neural ranking models
- Open-source contributions in AI/ML
- HR-tech, recruiting-tech, or marketplace experience

NOT a Fit:
- Entire career at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mindtree
- AI experience only LangChain + OpenAI wrappers with no pre-LLM production ML history
- Pure academic / research background with no production deployment
- No production code in last 18 months (architecture/management only)
- Job-hopper: 3+ companies in 4 years
- Primary domain: Computer Vision, Speech Recognition, or Robotics
- 5+ years exclusively on closed-source proprietary systems
"""

SKILL_CATEGORIES = {
    "embedding": [
        "embedding", "embeddings", "sentence-transformer", "sentence_transformer",
        "bge", "e5 model", "openai embedding", "ada-002", "text-embedding",
        "vector representation", "dense vector", "neural embedding",
        "bi-encoder", "cross-encoder", "contrastive learning",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "vector database", "vector store", "vector db",
        "ann search", "approximate nearest neighbor", "knn search",
        "hnsw", "ivf", "product quantization",
    ],
    "retrieval_ranking": [
        "retrieval", "information retrieval", "rag", "retrieval augmented",
        "semantic search", "hybrid search", "dense retrieval", "sparse retrieval",
        "bm25", "ranking", "re-ranking", "reranker", "learning to rank",
        "ltr", "recommendation system", "recommender", "candidate ranking",
        "relevance ranking", "search engine",
    ],
    "llm": [
        "llm", "large language model", "gpt", "bert", "transformer",
        "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
        "instruction tuning", "hugging face", "huggingface",
        "language model", "generative ai", "llama", "mistral", "gemma",
    ],
    "python": [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn",
        "sklearn", "numpy", "pandas", "fastapi", "flask",
    ],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "mean average precision", "a/b test",
        "a/b testing", "evaluation framework", "offline evaluation",
        "online evaluation", "benchmark", "precision@k", "recall@k",
    ],
    "mlops": [
        "mlflow", "kubeflow", "airflow", "spark", "kafka", "docker",
        "kubernetes", "model serving", "inference", "triton", "ray",
        "feature store",
    ],
}

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra",
    "mindtree", "mphasis", "hexaware", "mastech", "niit technologies",
    "l&t infotech", "larsen & toubro infotech", "zensar", "kpit",
    "birlasoft", "coforge",
}

PREFERRED_LOCATIONS = [
    "noida", "pune", "delhi", "new delhi", "gurugram", "gurgaon",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "ncr",
]

NON_TECH_TITLE_KEYWORDS = [
    "accountant", "hr manager", "human resources", "sales manager",
    "content writer", "graphic designer", "civil engineer", "teacher",
    "lawyer", "doctor", "nurse", "chef", "marketing manager",
    "operations manager", "finance manager", "supply chain",
    "logistics", "procurement", "auditor",
]

WRONG_DOMAIN_KEYWORDS = [
    "computer vision", "object detection", "image classification",
    "image segmentation", "opencv", "yolo", "speech recognition",
    "asr", "automatic speech recognition", "text to speech", "tts",
    "robotics", "ros ", "robot operating system", "slam", "lidar",
]


@dataclass
class JDRequirements:
    raw_text: str = JD_TEXT.strip()
    skill_categories: dict = field(default_factory=lambda: SKILL_CATEGORIES)
    services_companies: set = field(default_factory=lambda: SERVICES_COMPANIES)
    preferred_locations: list = field(default_factory=lambda: PREFERRED_LOCATIONS)
    non_tech_titles: list = field(default_factory=lambda: NON_TECH_TITLE_KEYWORDS)
    wrong_domains: list = field(default_factory=lambda: WRONG_DOMAIN_KEYWORDS)
    yoe_min: int = 5
    yoe_max: int = 9
    salary_min_lpa: float = 35.0
    salary_max_lpa: float = 38.0
    preferred_work_mode: str = "hybrid"
    category_weights: dict = field(default_factory=lambda: {
        "embedding": 1.0,
        "vector_db": 1.0,
        "retrieval_ranking": 1.0,
        "llm": 0.8,
        "python": 0.7,
        "eval_frameworks": 0.9,
        "mlops": 0.5,
    })


def parse_jd(jd_text: str | None = None) -> JDRequirements:
    if jd_text is None or jd_text.strip() == "":
        return JDRequirements()
    reqs = JDRequirements()
    reqs.raw_text = jd_text.strip()
    match = re.search(r'(\d+)[–\-]\s*(\d+)\s+years?', jd_text, re.IGNORECASE)
    if match:
        reqs.yoe_min = int(match.group(1))
        reqs.yoe_max = int(match.group(2))
    return reqs


# Singleton
JD_REQUIREMENTS = parse_jd()

if __name__ == "__main__":
    jd = parse_jd()
    print(f"YoE range: {jd.yoe_min}–{jd.yoe_max}")
    print(f"Skill categories: {list(jd.skill_categories.keys())}")
    print(f"Services companies: {len(jd.services_companies)}")
