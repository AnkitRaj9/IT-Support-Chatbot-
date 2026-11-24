import json
from collections import defaultdict

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.rag_project.retriver import (
    build_retriever,
    build_hybrid_retriever,
    FINAL_K,
)

load_dotenv()

GOLDEN_PATH = "golden_data/retriver_golden.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

# Choose ONE: "similarity", "hybrid", or "hybrid_rerank"
MODE = "hybrid_rerank"


# 1. LOAD the golden set
with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)

# 2. PICK the retriever
if MODE == "similarity":
    retriever = build_retriever()
elif MODE == "hybrid":
    retriever = build_hybrid_retriever(use_rerank=False)
elif MODE == "hybrid_rerank":
    retriever = build_hybrid_retriever(use_rerank=True)
else:
    raise ValueError(f"Unknown MODE: {MODE}")

# 3. RETRIEVE for every golden question
test_cases = []
for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

# 4. METRICS
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

# 5. EVALUATE
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": MODE,
        "embedding_model": "text-embedding-3-large",
        "chunk_size": 600,
        "chunk_overlap": 150,
        "top_k": FINAL_K,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)

# 6. AGGREGATE TABLE
scores = defaultdict(list)
passed = defaultdict(int)

for tr in result.test_results:
    for m in tr.metrics_data:
        scores[m.name].append(m.score)
        if m.success:
            passed[m.name] += 1

table = Table(title=f"Aggregate Metrics ({MODE})")
table.add_column("Metric")
table.add_column("Average Score")
table.add_column("Pass Rate")
table.add_column("Total")

for name, vals in scores.items():
    total = len(vals)
    avg = sum(vals) / total
    table.add_row(
        name,
        f"{avg:.2f}",
        f"{passed[name] / total:.2%} | passed={passed[name]} | failed={total - passed[name]}",
        str(total),
    )

Console().print(table)