import json
import os
import re
import time

import ir_datasets
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, RR, P, nDCG

DATASET_NAME = "beir/fiqa"
INDEX_FOLDER = "indexes"
RESULTS_FOLDER = "results"
BEST_PARAMETERS_PATH = f"{INDEX_FOLDER}/best_parameters.json"

INDEX_CONFIGURATIONS = {
    "full": {"stopwords": None, "stemmer": None},
    "stopwords": {"stopwords": "terrier", "stemmer": None},
    "stemming": {"stopwords": None, "stemmer": "porter"},
    "stopwords_stemming": {"stopwords": "terrier", "stemmer": "porter"},
}

RANKING_METRICS = [nDCG @ 5, nDCG @ 10, nDCG @ 20, RR @ 10, P @ 10, R @ 100]


def clean_query_text(query_text: str) -> str:
    without_punctuation = re.sub(r"[^\w\s]", " ", query_text)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def load_documents() -> pd.DataFrame:
    corpus = ir_datasets.load(f"{DATASET_NAME}/test")
    documents = pd.DataFrame(list(corpus.docs_iter()))
    documents = documents.rename(columns={"doc_id": "docno"})
    return documents[["docno", "text"]]


def load_queries_and_qrels(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = ir_datasets.load(f"{DATASET_NAME}/{split_name}")

    queries = pd.DataFrame(list(split.queries_iter()))
    queries = queries.rename(columns={"query_id": "qid", "text": "query"})
    queries["query"] = queries["query"].map(clean_query_text)

    qrels = pd.DataFrame(list(split.qrels_iter()))
    qrels = qrels.rename(columns={"query_id": "qid", "doc_id": "docno", "relevance": "label"})

    queries_with_judgements = queries[queries["qid"].isin(qrels["qid"])]
    return queries_with_judgements[["qid", "query"]], qrels[["qid", "docno", "label"]]


def index_path(index_name: str) -> str:
    return os.path.abspath(f"{INDEX_FOLDER}/{index_name}")


def open_index(index_name: str):
    return pt.IndexFactory.of(index_path(index_name))


def build_retriever(index, weighting_model: str, controls: dict = None, metadata: tuple = ("docno",)):
    return pt.terrier.Retriever(index, wmodel=weighting_model, controls=controls,
                                properties={"termpipelines": ""}, metadata=list(metadata))


def controls_for_model(model_name: str, settings: dict) -> dict:
    if model_name == "BM25":
        return {"bm25.b": str(settings["b"]), "bm25.k_1": str(settings["k1"])}
    return {"dirichletlm.mu": str(settings["mu"])}


def milliseconds_per_query(retrieval_system, queries: pd.DataFrame) -> float:
    start_time = time.time()
    retrieval_system.transform(queries)
    elapsed_seconds = time.time() - start_time
    return 1000 * elapsed_seconds / len(queries)


def save_best_parameters(best_parameters: dict):
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    with open(BEST_PARAMETERS_PATH, "w") as parameters_file:
        json.dump(best_parameters, parameters_file, indent=2)


def load_best_parameters() -> dict:
    with open(BEST_PARAMETERS_PATH) as parameters_file:
        return json.load(parameters_file)


def update_best_parameters(new_parameters: dict):
    best_parameters = load_best_parameters()
    best_parameters.update(new_parameters)
    save_best_parameters(best_parameters)


def save_results(results: pd.DataFrame, file_name: str):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    results.to_csv(f"{RESULTS_FOLDER}/{file_name}.csv", index=False)
