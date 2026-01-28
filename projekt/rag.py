import os
import re
import json
from typing import Any, Dict, List

try:
    import faiss
except Exception as exc:
    raise SystemExit("Install faiss-cpu: pip install faiss-cpu") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise SystemExit("Install sentence-transformers: pip install sentence-transformers") from exc

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
OVERLAP = 120


def simple_chunk(text: str, chunk_chars: int = CHUNK_SIZE, overlap: int = OVERLAP):
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        yield (i, j, text[i:j])
        if j >= len(text):
            break
        i = max(0, j - overlap)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_texts(texts: List[str], batch_size: int = 64):
    return embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

embedder = SentenceTransformer(MODEL_NAME)


def pack_context(hits, max_chars=2000):
    parts = []
    citations = []
    
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] {hit['chunk']}")
        citations.append({
            "n": i,
            "source": hit.get("source"),
            "page": hit.get("page"),
        })
    
    context = "\n\n".join(parts)[:max_chars]
    return {"context": context, "citations": citations}


class MiniRAG:
    
    def __init__(
        self,
        usda_json_path: str = "FoodData_Central_foundation_food_json_2025-12-18.json",
        chunk_chars: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
    ) -> None:
        self.usda_json_path = usda_json_path
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.docs = self._load_usda_corpus(usda_json_path)
        self.chunks = self._make_chunks(self.docs, chunk_chars, overlap)
        self.embs = embed_texts(self.chunks_texts(), batch_size=64)
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs)
        self.bm25 = None
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi([tokenize(c["chunk"]) for c in self.chunks])
    
    def _load_usda_corpus(self, usda_json_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        
        if not os.path.isfile(usda_json_path):
            return [
                {
                    "source": "demo",
                    "page": 1,
                    "text": "RAG łączy retrieval z generacją, doklejając kontekst do promptu.",
                },
                {
                    "source": "demo",
                    "page": 2,
                    "text": "Embeddingi zamieniają tekst na wektory, a FAISS umożliwia szybkie wyszukiwanie.",
                },
            ]
        
        with open(usda_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        foods = data.get("FoundationFoods", []) or []
        for food in foods[:100]:  # max 100 items for demo
            desc = food.get("description", "Unknown")
            nutrients = food.get("foodNutrients", []) or []
            
            # Extract macros
            macros = {}
            for n in nutrients:
                nu = n.get("nutrient", {})
                nid = nu.get("id")
                amount = n.get("amount", 0.0)
                
                if nid == 1008:  # Energy
                    macros["calories"] = f"{amount:.0f} kcal"
                elif nid == 1003:  # Protein
                    macros["protein"] = f"{amount:.1f}g"
                elif nid == 1004:  # Fat
                    macros["fat"] = f"{amount:.1f}g"
                elif nid == 1005:  # Carbs
                    macros["carbs"] = f"{amount:.1f}g"
            
            macro_str = ", ".join(f"{k}={v}" for k, v in macros.items())
            txt = f"{desc}. Wartości odżywcze na 100g: {macro_str}"
            
            rows.append({"source": "USDA", "page": 1, "text": txt})
        
        return rows
    
    def _make_chunks(
        self,
        docs: List[Dict[str, Any]],
        chunk_chars: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for d in docs:
            for k, (a, b, txt) in enumerate(simple_chunk(d.get("text", ""), chunk_chars, overlap)):
                if txt.strip():
                    out.append({
                        "source": d.get("source"),
                        "page": d.get("page", 1),
                        "chunk_id": k,
                        "start": a,
                        "end": b,
                        "chunk": txt.strip(),
                    })
        return out
    
    def chunks_texts(self) -> List[str]:
        return [c["chunk"] for c in self.chunks]
    
    def retrieve_dense(self, query: str, k: int = 5) -> List[Dict]:
        qv = embed_texts([query], batch_size=1)
        scores, idxs = self.index.search(qv, k)
        
        hits = []
        for i in range(min(k, len(idxs[0]))):
            hit = self.chunks[int(idxs[0][i])].copy()
            hit["score"] = float(scores[0][i])
            hits.append(hit)
        return hits
    
    def retrieve_bm25(self, query: str, k: int = 5) -> List[Dict]:
        if self.bm25 is None:
            return []
        
        toks = tokenize(query)
        scores = self.bm25.get_scores(toks)
        order = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:k]
        
        hits = []
        for idx in order:
            hit = self.chunks[idx].copy()
            hit["score"] = float(scores[idx])
            hits.append(hit)
        return hits
    
    def rrf_fuse(self, dense: List[Dict], sparse: List[Dict], k: int = 60) -> List[Dict]:
        scores = {}
        seen = {}  # Śledzenie po "chunk" zawartości
        
        for rank, hit in enumerate(dense):
            chunk_text = hit.get("chunk", "")
            if chunk_text not in seen:
                seen[chunk_text] = hit
            scores[chunk_text] = scores.get(chunk_text, 0.0) + 1.0 / (k + rank)
        
        for rank, hit in enumerate(sparse):
            chunk_text = hit.get("chunk", "")
            if chunk_text not in seen:
                seen[chunk_text] = hit
            scores[chunk_text] = scores.get(chunk_text, 0.0) + 1.0 / (k + rank)
        
        # Sort by fused score
        ranked = sorted(
            [(seen[chunk_text], scores[chunk_text]) for chunk_text in scores],
            key=lambda x: x[1],
            reverse=True
        )
        return [hit for hit, _ in ranked]
    
    def ask(self, question: str, k_dense: int = 4, k_sparse: int = 4, k_final: int = 4) -> Dict[str, Any]:
        dense_hits = self.retrieve_dense(question, k=k_dense)
        sparse_hits = self.retrieve_bm25(question, k=k_sparse)
        
        if sparse_hits:
            fused_hits = self.rrf_fuse(dense_hits, sparse_hits, k=60)[:k_final]
        else:
            fused_hits = dense_hits[:k_final]
        
        packed = pack_context(fused_hits, max_chars=2000)
        
        return {
            "question": question,
            "hits": fused_hits,
            "context": packed["context"],
            "citations": packed["citations"],
        }
