import os
import re
import json
from typing import Any, Dict, List, Tuple

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
OVERLAP = 80


def simple_chunk(text: str, chunk_chars: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Tuple[int, int, str]]:
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        out.append((i, j, text[i:j]))
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> Any:
    return embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def pack_context(
    hits: List[Tuple[float, Dict[str, Any]]],
    max_per_source: int = 2,
    max_chars: int = 2000,
) -> Tuple[str, List[Dict[str, Any]]]:
    per = {}
    ordered: List[Dict[str, Any]] = []
    
    for _, rec in hits:
        key = (rec.get("source"), rec.get("page"))
        per.setdefault(key, 0)
        if per[key] < max_per_source:
            ordered.append(rec)
            per[key] += 1
    
    cites = []
    parts = []
    for i, rec in enumerate(ordered, start=1):
        cites.append({
            "n": i,
            "source": rec.get("source"),
            "page": rec.get("page"),
            "chunk_id": rec.get("chunk_id"),
        })
        parts.append(f"[{i}] {rec.get('chunk', '')}")
    
    ctx = "\n\n".join(parts)[:max_chars]
    return ctx, cites


class MiniRAG:
    
    def __init__(
        self,
        usda_json_path: str = "FoodData_Central_foundation_food_json_2025-12-18.json",
        chunk_chars: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.usda_json_path = usda_json_path
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.docs = self._load_usda_corpus(usda_json_path)
        self.chunks = self._make_chunks(self.docs, chunk_chars, overlap)
        self.embs = embed_texts(self.embedder, [c["chunk"] for c in self.chunks])
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
        for food in foods[:100]:  # Limit to 100 for demo
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
    
    def retrieve_dense(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        qv = embed_texts(self.embedder, [query])
        scores, idxs = self.index.search(qv, k)
        
        res: List[Tuple[float, Dict[str, Any]]] = []
        for i in range(min(k, len(idxs[0]))):
            res.append((float(scores[0][i]), self.chunks[int(idxs[0][i])]))
        return res
    
    def retrieve_bm25(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if self.bm25 is None:
            return []
        
        toks = tokenize(query)
        scores = self.bm25.get_scores(toks)
        order = list(reversed(sorted(range(len(scores)), key=lambda i: scores[i])))[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]
    
    def rrf_fuse(
        self,
        dense: List[Tuple[float, Dict[str, Any]]],
        sparse: List[Tuple[float, Dict[str, Any]]],
        k: int = 60,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scores: Dict[int, float] = {}
        for rank, (_, doc) in enumerate(dense):
            try:
                idx = self.chunks.index(doc)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            except ValueError:
                pass
        for rank, (_, doc) in enumerate(sparse):
            try:
                idx = self.chunks.index(doc)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            except ValueError:
                pass
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(scores[idx], self.chunks[idx]) for idx, _ in ranked]
    
    def ask(
        self,
        question: str,
        k_dense: int = 4,
        k_sparse: int = 4,
        k_final: int = 4,
    ) -> Dict[str, Any]:
        dense_hits = self.retrieve_dense(question, k=k_dense)
        sparse_hits = self.retrieve_bm25(question, k=k_sparse)
        
        if sparse_hits:
            fused_hits = self.rrf_fuse(dense_hits, sparse_hits, k=60)[:k_final]
        else:
            fused_hits = dense_hits[:k_final]
        
        context, citations = pack_context(fused_hits, max_per_source=2, max_chars=2000)
        
        return {
            "question": question,
            "hits": fused_hits,
            "context": context,
            "citations": citations,
        }
