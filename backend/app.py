"""
EcoPolicy / Prakriti-Niti AI  -  FastAPI backend + static frontend host.

Works the same way in both places:
  * Local :  python app.py                 -> http://localhost:8000
  * Render:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from openai import OpenAI, AuthenticationError, RateLimitError

# --------------------------------------------------------------------------
# Environment & Logging
# --------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    env_path = THIS_DIR / ".env"
    if env_path.exists():
        # override=False (default): real environment variables (e.g. on Render) always win.
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecopolicy")

# --------------------------------------------------------------------------
# Provider Resolution & Client Setup
# --------------------------------------------------------------------------
PROVIDER_CONFIGS = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "fallback_models": [
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ],
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "fallback_models": ["gpt-4o-mini", "gpt-4o"],
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "fallback_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "fallback_models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"],
    },
}

# Model families that Google has shut down (Gemini 1.x and 2.0). 2.5 is NOT listed: still served.
DEPRECATED_MODEL_MARKERS = ("gemini-1.0", "gemini-1.5", "gemini-2.0")


def resolve_provider() -> dict:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced and forced not in PROVIDER_CONFIGS:
        logger.warning("LLM_PROVIDER='%s' is not recognised; auto-detecting instead.", forced)
    order = [forced] if forced in PROVIDER_CONFIGS else ["gemini", "openai", "groq", "openrouter"]

    for name in order:
        cfg = PROVIDER_CONFIGS[name]
        key = (os.getenv(cfg["env_key"]) or "").strip()  # strip stray spaces/newlines from copy-paste
        if key:
            # `or` (not a default arg) so an EMPTY env var falls back to the provider default.
            base_url = (os.getenv("LLM_BASE_URL") or "").strip() or cfg["base_url"]
            fallback_env = os.getenv("LLM_FALLBACK_MODELS") or ""
            models = [m.strip() for m in fallback_env.split(",") if m.strip()] or list(cfg["fallback_models"])
            return {"provider": name, "api_key": key, "base_url": base_url, "fallback_models": models}

    return {"provider": None, "api_key": None, "base_url": None, "fallback_models": []}


PROVIDER = resolve_provider()

if PROVIDER["api_key"]:
    logger.info(
        "Provider resolved: %s (base_url=%s)",
        PROVIDER["provider"],
        PROVIDER["base_url"] or "default OpenAI endpoint",
    )
else:
    logger.warning("No API key found. The server will start, but /api/chat will return 500 errors.")

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not PROVIDER["api_key"]:
        raise HTTPException(
            status_code=500,
            detail="No API key configured on the server. Set GEMINI_API_KEY (or another provider key) in the environment.",
        )
    # max_retries=0: our own model-fallback chain handles retries, and it keeps total time bounded.
    _client = OpenAI(api_key=PROVIDER["api_key"], base_url=PROVIDER["base_url"], max_retries=0, timeout=60.0)
    return _client


def normalize_model(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    if any(marker in model for marker in DEPRECATED_MODEL_MARKERS) and PROVIDER["fallback_models"]:
        logger.info("'%s' looks deprecated - substituting '%s'", model, PROVIDER["fallback_models"][0])
        return PROVIDER["fallback_models"][0]
    return model


# --------------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------------
app = FastAPI(title="EcoPolicy Research Assistant API")

# The frontend is served by this same app (same origin), so CORS is only needed if you
# host the frontend elsewhere. Add extra origins via ALLOWED_ORIGINS (comma-separated).
_port_for_cors = os.getenv("PORT", "8000")
_origins = [
    f"http://localhost:{_port_for_cors}",
    f"http://127.0.0.1:{_port_for_cors}",
]
_origins += [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
_render_url = (os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")  # set automatically by Render
if _render_url:
    _origins.append(_render_url)
ALLOWED_ORIGINS = list(dict.fromkeys(_origins))  # de-duplicate, keep order

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # no cookies/auth are used
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are "EcoPolicy-AI Scholar," an authoritative research assistant specialized exclusively in the research paper: "AI-Powered Conversational Assistants for Environmental Management Policy in India" (a scoping review based on the Arksey & O'Malley framework and PRISMA-ScR guidelines).

### CORE RULES & OPERATING PRINCIPLES:
1. Grounding & Faithfulness: Answer questions using ONLY the findings, methodology, and proposed architecture from this paper. Do not extrapolate or hallucinate external laws.
2. Evidence Tiering Discipline:
   - Tier 1 (India Direct Environment): Only ONE verified system exists: SukhaRakshak AI (Gemini-based RAG drought advisory by IWMI & ICAR-CRIDA, role-tiered for farmers, extension workers, and district managers; unreviewed pilot). Note that the SIH 2025 INGRES chatbot was excluded due to unverified bibliographic data.
   - Tier 2 (India Adjacent): Non-environmental e-governance & public AI infrastructure. Includes National Consumer Helpline AI integration, NITI Aayog municipal grievance chatbots, NITI Aayog 7 Principles for Responsible AI (2021), MeitY/IndiaAI AI Governance Guidelines (2025), Bhashini/AI4Bharat (22 scheduled languages), SnehAI (Hinglish health bot), and PARIVESH (MoEFCC single-window clearance portal).
   - Tier 3 (International Transferable Climate/Policy RAG): ClimateGPT, ChatClimate, Juhasz et al. (2024), and Thulke et al. (2025, demonstrating baseline RAG faithfulness gaps).
3. Core Argument: Chatbots are feasible as assistive administrative interfaces, but NOT as autonomous decision-makers (statutory power stays with MoEFCC, CPCB, SPCBs/PCCs). Semantic vector similarity alone fails in regulatory domains due to versioning and temporal validity.
4. Abstention Protocol: If asked about empirical performance figures for Indian environmental bots in regional languages or legal liability frameworks, state that the paper identifies these as critical, unresolved research gaps.
"""

# Limits (protect your API quota and keep requests fast)
MAX_HISTORY_MESSAGES = 20      # only the most recent N turns are sent to the model
MAX_CHARS_PER_MESSAGE = 8000
PER_CALL_TIMEOUT = 50.0        # seconds for a single model attempt
TOTAL_BUDGET = 85.0            # seconds for the whole fallback chain (Render cuts requests at ~100s)
ALLOW_CLIENT_MODEL = os.getenv("ALLOW_CLIENT_MODEL", "").strip().lower() in ("1", "true", "yes")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    model: Optional[str] = None            # ignored unless ALLOW_CLIENT_MODEL=true
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


def clean_messages(messages: List[ChatMessage]) -> List[dict]:
    """Keep only user/assistant turns, drop blanks, merge same-role neighbours, cap size."""
    cleaned: List[dict] = []
    for m in messages:
        role = (m.role or "").strip().lower()
        if role not in ("user", "assistant"):  # never let a client inject its own 'system' prompt
            continue
        text = (m.content or "").strip()
        if not text:
            continue
        text = text[:MAX_CHARS_PER_MESSAGE]
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n\n" + text
        else:
            cleaned.append({"role": role, "content": text})
    cleaned = cleaned[-MAX_HISTORY_MESSAGES:]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def build_attempt_chain(requested_model: Optional[str]) -> List[str]:
    chain: List[str] = []
    normalized = normalize_model(requested_model) if ALLOW_CLIENT_MODEL else None
    if normalized:
        chain.append(normalized)
    for m in PROVIDER["fallback_models"]:
        if m not in chain:
            chain.append(m)
    return chain


def looks_like_auth_error(err: Exception) -> bool:
    if isinstance(err, AuthenticationError):
        return True
    text = str(err).lower()
    # Gemini's OpenAI-compatible endpoint reports a bad key as HTTP 400 "API key not valid".
    return "api key not valid" in text or "api_key_invalid" in text


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the frontend UI."""
    frontend_path = THIS_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html", headers={"Cache-Control": "no-cache"})
    return HTMLResponse("<h1>index.html not found</h1><p>Keep index.html next to app.py.</p>", status_code=404)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "provider": PROVIDER["provider"],
        "api_key_configured": bool(PROVIDER["api_key"]),
        "fallback_models": PROVIDER["fallback_models"],
    }


# NOTE: plain `def` (not `async def`) on purpose. The OpenAI client call below is blocking; FastAPI
# runs sync endpoints in a worker thread so one slow request cannot freeze every other visitor.
@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    client = get_client()
    attempt_chain = build_attempt_chain(request.model)
    if not attempt_chain:
        raise HTTPException(status_code=500, detail="No models configured.")

    history = clean_messages(request.messages)
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Send at least one user message (the last message must be from the user).")

    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    started = time.monotonic()
    last_error: Optional[Exception] = None

    for model_name in attempt_chain:
        remaining = TOTAL_BUDGET - (time.monotonic() - started)
        if remaining < 5:
            logger.warning("Time budget exhausted before trying '%s'", model_name)
            break
        try:
            logger.info("Trying model: %s", model_name)
            kwargs = {
                "model": model_name,
                "messages": formatted_messages,
                "timeout": min(PER_CALL_TIMEOUT, remaining),
            }
            # Google has deprecated temperature/top_p/top_k for Gemini 3.x models, so don't send it there.
            if PROVIDER["provider"] != "gemini":
                kwargs["temperature"] = request.temperature

            response = client.chat.completions.create(**kwargs)
            reply = ""
            if response.choices:
                reply = (response.choices[0].message.content or "").strip()
            if not reply:
                logger.warning("Model '%s' returned an empty reply; trying next model", model_name)
                last_error = RuntimeError("The model returned an empty reply.")
                continue

            logger.info("Success with model: %s", model_name)
            return {"response": reply, "content": reply, "model_used": model_name}

        except Exception as e:  # noqa: BLE001 - we deliberately try the next model on any failure
            if looks_like_auth_error(e):
                logger.error("Authentication failed with provider: %s", e)
                raise HTTPException(
                    status_code=502,
                    detail="The AI provider rejected the server's API key. Check the API key configured on the server.",
                )
            logger.warning("Model '%s' failed: %s", model_name, e)
            last_error = e
            continue

    logger.error("ALL MODELS FAILED (last error: %s)", last_error)
    if isinstance(last_error, RateLimitError):
        raise HTTPException(status_code=429, detail="The AI provider is rate-limiting requests. Please wait a moment and try again.")
    hint = str(last_error)[:300] if last_error else "unknown error"
    raise HTTPException(status_code=502, detail=f"All models failed. Last error: {hint}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    # Render (and other hosts) set PORT and need 0.0.0.0. Locally we default to loopback only.
    host = os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    logger.info("=" * 70)
    logger.info("Starting EcoPolicy Research Assistant on http://%s:%s  (open http://localhost:%s)", host, port, port)
    logger.info("=" * 70)
    uvicorn.run(app, host=host, port=port)