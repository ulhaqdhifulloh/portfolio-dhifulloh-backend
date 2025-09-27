import os, time, math, re
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ==== CONFIG via ENV ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

MODEL = "gpt-4o-mini"
DAY_LIMIT = int(os.getenv("DAY_LIMIT", "20"))
MIN_LIMIT = int(os.getenv("MIN_LIMIT", "6"))

# Comma-separated origins, e.g. "https://ulhaqdhifulloh.github.io,https://your-custom.com"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://ulhaqdhifulloh.github.io/portfolio-dhifulloh").split(",") if o.strip()]
# =========================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # untuk debug awal bisa ["*"], tapi kembali ketat setelah OK
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)
# in-memory rate limiter (cukup untuk demo; untuk produksi pakai Redis/KV)
buckets: Dict[str, Dict[str, Any]] = {}

def rate_limited(ip: str):
    now = time.time()
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    b = buckets.get(ip, {"day": 0, "min": 0, "tmin": now, "day_key": day_key})
    if now - b["tmin"] > 60:  # reset per minute
        b["min"] = 0
        b["tmin"] = now
    if b["day_key"] != day_key:  # reset per day
        b["day"] = 0
        b["day_key"] = day_key
    if b["day"] >= DAY_LIMIT:
        return True, "Daily limit reached"
    if b["min"] >= MIN_LIMIT:
        return True, "Too many requests"
    b["day"] += 1
    b["min"] += 1
    buckets[ip] = b
    return False, DAY_LIMIT - b["day"]

def tokenize(s: str) -> List[str]:
    return [t for t in re.sub(r"[^a-z0-9_ \-]", " ", (s or "").lower()).split() if t]

def tf(tokens: List[str]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for t in tokens: m[t] = m.get(t, 0) + 1
    return m

def cos(a: Dict[str, int], b: Dict[str, int]) -> float:
    dot = sum(a[k]*b.get(k,0) for k in a)
    a2 = sum(v*v for v in a.values())
    b2 = sum(v*v for v in b.values())
    if not a2 or not b2: return 0.0
    return dot / (math.sqrt(a2) * math.sqrt(b2))

def retrieve_top_k(query: str, sections: List[Dict[str, Any]], k: int = 3):
    qtf = tf(tokenize(query))
    scored = []
    for s in sections or []:
        stf = tf(tokenize(f"{s.get('title','')} {s.get('text','')}"))
        score = cos(qtf, stf)
        scored.append({
            "id": s.get("id"), "title": s.get("title"),
            "text": str(s.get("text",""))[:900], "score": score
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def build_system_prompt(p: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> str:
    skills = ", ".join(p.get("skillsTop", []))
    exp = "\n".join([f"- {e['role']} @ {e['org']} ({e['period']}) — {e['summary']}"
                     for e in p.get("experience", [])])
    pro = "\n".join([f"- {pr['name']} ({pr['year']}): {pr['blurb']}"
                     for pr in p.get("projects", [])])
    ctx = "\n\n".join([f"[{s['id']}] {s['title']}\n{s['text']}" for s in (retrieved or [])]) or "(no extra context)"
    return f"""
You are an enthusiastic, concise assistant representing Dhifulloh Dhiya Ulhaq.
Use ONLY the provided profile and context. If unknown, say you don't know.

Basic:
- Name: {p.get('basics',{}).get('name')}
- Title: {p.get('basics',{}).get('title')}
- Location: {p.get('basics',{}).get('location')}
- Email: {p.get('basics',{}).get('email')}
- Links: GitHub {p.get('basics',{}).get('links',{}).get('github')}, LinkedIn {p.get('basics',{}).get('links',{}).get('linkedin')}, YouTube {p.get('basics',{}).get('links',{}).get('youtube')}

Top skills: {skills}

Experiences:
{exp}

Key projects:
{pro}

Retrieved context (RAG-lite):
{ctx}

Guidelines:
- Be truthful and specific. Use bullet points when suitable.
- Keep answers compact unless asked for details.
- If asked for links, point to the links above.
- For collaboration/requests, suggest emailing {p.get('basics',{}).get('email')}.
- Tone: friendly, professional.
""".strip()

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: Request):
    ip = req.client.host or "0.0.0.0"
    limited, remaining = rate_limited(ip)
    if limited:
        raise HTTPException(status_code=429, detail=str(remaining))

    body = await req.json()
    question = str(body.get("question",""))[:600]
    profile = body.get("profile", {}) or {}

    retrieved = retrieve_top_k(question, profile.get("longForm", []), 3)
    system_prompt = build_system_prompt(profile, retrieved)

    r = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":question}
        ],
    )
    answer = (r.choices[0].message.content or "No answer").strip()
    return {"answer": answer, "remaining": remaining}