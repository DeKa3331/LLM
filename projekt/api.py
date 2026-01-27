import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import ask_rag, chat_once, looks_like_injection, RagQueryArgs

app = FastAPI(
    title="Mini-RAG API",
    description="Lab 7: REST API for RAG with guardrails and function-calling",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=200, description="Query for RAG")
    k: int = Field(4, ge=1, le=10, description="Number of retrieval results")
    use_functions: bool = Field(True, description="Use function-calling mode")
    mode: str = Field("local", description="Model backend: local, gemini, groq")


class AskResponse(BaseModel):
    status: str
    mode: Optional[str] = None
    latency_s: Optional[float] = None
    final: Optional[str] = None  # For chat_once responses
    text: Optional[str] = None  # For Gemini/Groq text
    flags: Optional[dict] = None
    hits: Optional[list] = None
    context: Optional[str] = None
    citations: Optional[list] = None
    tool: Optional[str] = None
    tool_out: Optional[dict] = None
    meta: Optional[dict] = None
    error: Optional[str] = None
    reason: Optional[str] = None


@app.get("/")
async def root():
    return {
        "service": "mini-rag-api",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/ask": "POST - Query RAG with guardrails",
            "/ask-raw": "POST - Raw query without strict validation",
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mini-rag-api"}


@app.post("/ask", response_model=AskResponse, status_code=200)
async def ask_endpoint(req: AskRequest):
    """
    Status codes:
      - 200: OK
      - 400: Validation error or injection detected
      - 503: Model/tool error
    """
    start = time.perf_counter()
    
    if looks_like_injection(req.question):
        raise HTTPException(
            status_code=400,
            detail="Possible prompt injection detected"
        )
    
    try:
        args = RagQueryArgs(question=req.question, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    
    try:
        if req.use_functions:
            result = ask_rag(args.question, k=args.k)
        else:
            os.environ["MODEL_MODE"] = req.mode.lower()
            result = chat_once(
                prompt=args.question,
                k=args.k,
            )
        
        latency = round(time.perf_counter() - start, 3)
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=503,
                detail=result.get("error", "Unknown error")
            )
        
        return AskResponse(
            status=result.get("status", "ok"),
            mode=result.get("mode", req.mode),
            latency_s=result.get("latency_s", latency),
            final=result.get("final"),
            text=result.get("text"),
            flags=result.get("flags"),
            hits=result.get("hits"),
            context=result.get("context"),
            citations=result.get("citations"),
            tool=result.get("tool", "rag.search" if req.use_functions else None),
            tool_out={
                "hits": len(result.get("hits", [])),
                "context_chars": len(result.get("context", "")),
            } if req.use_functions else None,
            meta={
                "retrieved_ids": [i for i, _ in enumerate(result.get("hits", []))],
                "mode": req.mode,
                "use_functions": req.use_functions,
            },
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Internal error: {str(e)}")


@app.post("/ask-raw")
async def ask_raw_endpoint(req: AskRequest):
    if looks_like_injection(req.question):
        raise HTTPException(status_code=400, detail="Injection detected")
    
    os.environ["MODEL_MODE"] = req.mode.lower()
    
    if req.use_functions:
        result = ask_rag(req.question, k=req.k)
    else:
        result = chat_once(prompt=req.question, k=req.k)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=503, detail=result.get("error"))
    
    return result


# Run the API with: uvicorn api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    print("Starting Mini-RAG API...")
    print("Run with: uvicorn api:app --reload --port 8000")
    print("Docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
