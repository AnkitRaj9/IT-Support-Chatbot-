import os
import glob
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()  # loads OPENAI_API_KEY and PINECONE_API_KEY from .env

DATA_DIR = "data"
INDEX_NAME = "rag-chatbot"
DIMENSION = 3072  # must match text-embedding-3-large

FETCH_K = 20   # candidates taken from EACH search (vector and keyword)
FINAL_K = 5    # chunks returned after reranking


# 1. LOAD ---- read every .md file in the data folder
def load_docs():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.md"):
        docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


# 2. CHUNK ---- split into smaller overlapping pieces
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


# helper: load + chunk in one call (used by both Pinecone ingestion and BM25)
def get_chunks():
    docs = load_docs()
    if not docs:
        raise ValueError(f"No .md files found in '{DATA_DIR}/'")
    return chunk_docs(docs)


# 3. BUILD ---- create the Pinecone index if missing, embed once, store
def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)

    # if the index is empty, ingest; otherwise just connect
    if index.describe_index_stats()["total_vector_count"] == 0:
        chunks = get_chunks()
        print(f"Loaded {len(chunks)} chunks. Embedding...")
        return PineconeVectorStore.from_documents(
            chunks, embeddings, index_name=INDEX_NAME
        )

    return PineconeVectorStore(index=index, embedding=embeddings)


# 4a. BASELINE ---- plain similarity search (kept for comparison)
def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": FINAL_K})


# 4b. HYBRID ---- merge vector + keyword results using Reciprocal Rank Fusion
def reciprocal_rank_fusion(result_lists, c=60):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (c + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered]


# 4c. HYBRID + RERANK ---- the full pipeline
class HybridRerankRetriever:
    def __init__(self, use_rerank=True):
        self.use_rerank = use_rerank

        # vector side (Pinecone)
        self.vector = load_store().as_retriever(search_kwargs={"k": FETCH_K})

        # keyword side (BM25, built in memory from the same chunks)
        self.bm25 = BM25Retriever.from_documents(get_chunks())
        self.bm25.k = FETCH_K

        # reranker client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def invoke(self, query):
        # 1. two searches
        vector_docs = self.vector.invoke(query)
        keyword_docs = self.bm25.invoke(query)

        # 2. merge into one candidate list
        candidates = reciprocal_rank_fusion([vector_docs, keyword_docs])

        if not self.use_rerank:
            return candidates[:FINAL_K]

        # 3. rerank the candidates, keep the best FINAL_K
        result = self.pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=[d.page_content for d in candidates],
            top_n=FINAL_K,
            return_documents=False,
        )
        return [candidates[r.index] for r in result.data]


def build_hybrid_retriever(use_rerank=True):
    return HybridRerankRetriever(use_rerank=use_rerank)


# 5. TRY IT ---- run: python -m src.rag_project.retriver
if __name__ == "__main__":
    retriever = build_hybrid_retriever(use_rerank=True)
    results = retriever.invoke(
        "How do I connect to the VPN when I am working from outside the office?"
    )
    for r in results:
        source = os.path.basename(r.metadata.get("source", "unknown"))
        print(f"[{source}]")
        print(f"{r.page_content[:200]}...\n")