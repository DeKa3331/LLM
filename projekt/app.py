import json
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from rag import MiniRAG

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception as exc:
    raise SystemExit("Install pydantic: pip install pydantic") from exc

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
except Exception:
    genai = None

try:
    from groq import Groq
except Exception:
    Groq = None

DEFAULT_SYSTEM = "You are a factual assistant. Answer only using provided context."
MODEL_MODE = os.getenv("MODEL_MODE", "local").lower()

#security checks for number/id/credicard-like patterns/bank accounts/profanity/injection/links
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RE_PHONE = re.compile(r"(?:\+?48)?\s?(?:\d[ -]?){9,}")
RE_PESEL = re.compile(r"\b\d{11}\b")
RE_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
RE_IBAN = re.compile(r"\bPL\d{26}\b", re.IGNORECASE)
PROFANITY = {"cholera", "kurde", "pierd", "jebać", "jebać", "chuj", "pizda", "skurwysyn", "zajebisty"}

INJECTION_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"reveal.*?(system|developer) prompt",
    r"override .* rules",
    r"act as (system|developer)",
    r"you are now",
    r"jailbreak",
    r"follow the (next|below) instructions",
    r"disregard (your|the) (prior|previous) (directions|instructions)",
    r"(?i)\[system\].*?\[\/system\]",
    r"(?i)\[developer\].*?\[\/developer\]",
    r"(?i)\[\/?prompt\]",
    r"(?i)system:.*?user:",
    r"(?i)developer:.*?user:",
    r"(?i)prompt:.*?user:",
    r"shutdown",
    r"format your response as",
    r"delete (all )?your (memory|knowledge)",
    r"what is the (system|developer) prompt\?",
    r"print out (the )?(system|developer) prompt",
    r"show me (the )?(system|developer) prompt",
]

ALLOWED_DOMAINS = {"example.com"}


def contains_pii(text: str) -> dict:
    return {
        "email": bool(RE_EMAIL.search(text)),
        "phone": bool(RE_PHONE.search(text)),
        "pesel": bool(RE_PESEL.search(text)),
        "card": bool(RE_CARD.search(text)),
        "iban": bool(RE_IBAN.search(text)),
    }


def contains_profanity(text: str) -> bool:
    return any(w in text.lower() for w in PROFANITY)


def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)


def links_not_allowed(text: str) -> bool:
    urls = re.findall(r"https?://([^/\s]+)", text)
    return any(u.lower() not in ALLOWED_DOMAINS for u in urls)


def scrub_user_input(user: str) -> str:
    out = re.sub(
        r"(?i)(ignore (all|previous|above) instructions|reveal (system|developer) prompt|jailbreak|act as developer)",
        "",
        user,
    )
    out = re.sub(r"(?i)\[system\].*?\[\/system\]", "", out)
    return out.strip()


class RagQueryArgs(BaseModel):
    question: str = Field(..., min_length=2, max_length=200, description="Food item to search")
    k: int = Field(4, ge=1, le=10, description="Number of results")


ToolName = Literal["rag.search"]


class ToolCall(BaseModel):
    tool: ToolName
    args: Dict[str, Any]


ALLOWED_TOOLS = {"rag.search"}

TOOL_REGISTRY = {
    "rag.search": lambda question, k=4: _ask_rag_impl(question, k),
}

RAG_INSTANCE: Optional[MiniRAG] = None


def get_rag() -> MiniRAG:
    global RAG_INSTANCE
    if RAG_INSTANCE is None:
        RAG_INSTANCE = MiniRAG()
    return RAG_INSTANCE


class RagResponse(BaseModel):
    status: str
    mode: Optional[str] = None
    latency_s: Optional[float] = None
    flags: Optional[dict] = None
    hits: Optional[List[Any]] = None
    context: Optional[str] = None
    citations: Optional[List[Any]] = None
    question: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None


def gemini_generate(prompt: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0, max_output_tokens: int = 512) -> Dict[str, Any]:
    if genai is None:
        return {"status": "error", "mode": "gemini", "error": "google-genai not installed"}
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        
        if not api_key:
            return {"status": "error", "mode": "gemini", "error": "GOOGLE_API_KEY not set"}
        
        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=int(max_output_tokens),
        )
        
        start = time.perf_counter()
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        latency = round(time.perf_counter() - start, 3)
        
        usage = getattr(resp, "usage_metadata", None)
        usage_dict = {}
        if usage:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }
        
        return {
            "status": "ok",
            "mode": "gemini",
            "text": getattr(resp, "text", str(resp)),
            "latency_s": latency,
            "usage": usage_dict,
        }
    except Exception as e:
        return {"status": "error", "mode": "gemini", "error": str(e)}


def groq_generate(prompt: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0, max_output_tokens: int = 512) -> Dict[str, Any]:
    if Groq is None:
        return {"status": "error", "mode": "groq", "error": "groq-sdk not installed"}
    
    try:
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            return {"status": "error", "mode": "groq", "error": "GROQ_API_KEY not set"}
        
        client = Groq(api_key=api_key)
        
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", #pierwszy lepszy model ktory dziala, ten z zajec juz nie jest aktualny
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=int(max_output_tokens),
        )
        latency = round(time.perf_counter() - start, 3)
        
        return {
            "status": "ok",
            "mode": "groq",
            "text": resp.choices[0].message.content,
            "latency_s": latency,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }
    except Exception as e:
        return {"status": "error", "mode": "groq", "error": str(e)}


def local_generate(prompt: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0, max_output_tokens: int = 512, k: int = 4) -> Dict[str, Any]:
    tc = ToolCall(tool="rag.search", args={"question": prompt, "k": k})
    result = run_tool(tc, timeout_s=2.0)
    if not result["ok"]:
        return {"status": "error", "error": result["error"], "mode": "local"}
    try:
        RagResponse(**result["payload"])
        return result["payload"]
    except ValidationError as e:
        return {"status": "error", "error": f"Invalid response format: {str(e)}", "mode": "local"}


def chat_once(prompt: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0, max_output_tokens: int = 512, k: int = 4) -> Dict[str, Any]:
    mode = os.getenv("MODEL_MODE", MODEL_MODE).lower()
    if mode == "gemini":
        return gemini_generate(prompt, system, temperature=temperature, max_output_tokens=max_output_tokens)
    if mode == "groq":
        return groq_generate(prompt, system, temperature=temperature, max_output_tokens=max_output_tokens)
    return local_generate(prompt, system, temperature=temperature, max_output_tokens=max_output_tokens, k=k)


def _ask_rag_impl(question: str, k: int = 4) -> Dict[str, Any]:
    start = time.perf_counter()
    pii_check = contains_pii(question)
    flags = {
        "pii": any(pii_check.values()),
        "profanity": contains_profanity(question),
        "injection": looks_like_injection(question),
        "links_bad": links_not_allowed(question),
    }
    
    if flags["injection"] or flags["links_bad"]:
        return {
            "status": "blocked",
            "reason": "injection_or_disallowed_links",
            "flags": flags,
        }
    
    cleaned = scrub_user_input(question)
    rag = get_rag()
    res = rag.ask(cleaned, k_dense=k, k_sparse=k, k_final=k)
    latency = round(time.perf_counter() - start, 3)
    
    return {
        "status": "ok",
        "mode": "rag",
        "latency_s": latency,
        "flags": flags,
        **res,
    }


def _run_tool_sync(tc: ToolCall) -> Dict[str, Any]:
    if tc.tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool not allowed: {tc.tool}")
    
    args = RagQueryArgs(**tc.args)
    
    impl = TOOL_REGISTRY.get(tc.tool)
    if impl is None:
        raise ValueError("Unknown tool")
    return impl(args.question, args.k)


def run_tool(tc: ToolCall, timeout_s: float = 5.0) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run_tool_sync, tc)
        try:
            out = fut.result(timeout=timeout_s)
            return {"ok": True, "payload": out, "error": None}
        except FuturesTimeout:
            return {"ok": False, "payload": {}, "error": "timeout"}
        except ValidationError as err:
            return {"ok": False, "payload": {}, "error": f"validation_error: {err}"}
        except Exception as err:
            return {"ok": False, "payload": {}, "error": str(err)}


def ask_rag(question: str, k: int = 4) -> Dict[str, Any]:
    tc = ToolCall(tool="rag.search", args={"question": question, "k": k})
    
    result = run_tool(tc, timeout_s=5.0)
    
    if not result["ok"]:
        return {
            "status": "error",
            "mode": "local",
            "latency_s": 0.0,
            "error": result["error"],
            "flags": {},
            "hits": [],
            "context": "",
            "citations": [],
            "tool": "rag.search",
        }
    
    try:
        RagResponse(**result["payload"])
        return result["payload"]
    except ValidationError as e:
        return {
            "status": "error",
            "mode": "local",
            "latency_s": result["payload"].get("latency_s", 0.0),
            "error": f"Invalid response format: {str(e)}",
            "flags": result["payload"].get("flags", {}),
            "hits": [],
            "context": "",
            "citations": [],
            "tool": "rag.search",
        }


def main() -> None:
    print("Mini-RAG: Zapytaj o produkty żywieniowe z bazy USDA")
    print("Przykład: 'hummus', 'beef hot dog', 'tomatoes'")
    print()
    question = input("Twoje pytanie: ").strip()
    if not question:
        print("Brak pytania, koniec.")
        return
    
    result = ask_rag(question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

