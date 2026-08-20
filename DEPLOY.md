# Deploying to Render's free tier

A public URL you can put in an application or open during an interview.

The deployed service is **query-only and strictly read-only**. Transcription and
index building happen on your machine; the service ships a committed NumPy index
and answers questions against it. Nothing in the runtime writes to disk, because
Render's free tier has no persistent storage.

---

## The shape of it

```
YOUR MACHINE (once per batch of episodes)          RENDER (free web service)
-----------------------------------------          -------------------------
 episodes/*.mp3                                      data/index/  (committed)
   | faster-whisper                                       | mmap, read-only
 data/transcripts/*.json  (cached)                   POST /ask
   | chunker                                              | embed question
 ONNX MiniLM (committed, 23MB)  <---- same model ---> ONNX MiniLM (local, 3ms)
   |                                                      | dot product
 data/index/{embeddings.npy, chunks.json}            Groq -> answer + citations
   | git push
```

Two consequences worth internalising:

- **The runtime has no ML framework.** `requirements.txt` contains no torch, no
  sentence-transformers, no chromadb. Questions are embedded locally by ONNX
  Runtime against a committed 23MB MiniLM export — the same model
  sentence-transformers would run, without torch's ~830MB. Measured: **192MB
  peak RSS**, **0.2s boot**, **3ms retrieval**.
- **Whichever embedder builds the index must also serve queries.** Question and
  document vectors have to come from the same model or retrieval degrades
  *silently*. `data/index/manifest.json` records the embedder and the service
  refuses to start on a mismatch.

---

## Read this first

**A public link spends your Groq quota.** Embedding is local and free, but
generation is not, and there is no authentication on `/ask`. Groq's free tier measures at **8,000 tokens/minute and 1,000 requests/day**
(~2,000 tokens per query, so about 4 questions/minute). The client honours
`Retry-After`, so bursts come back slow rather than broken. Fine for a link sent
to a handful of interviewers; see [Protecting the quota](#protecting-the-quota)
if it spreads.

**Your transcripts become public.** `/ask` returns chunk text, so anything said in
an episode is readable by anyone with the link.

**Free-tier services idle out.** Render spins a free web service down after ~15
minutes without traffic, and the next request waits for a cold start. See
[Keeping it warm](#keeping-it-warm).

---

## 1. Push the repo

There is no preparation step. The index (`data/index/`, 264KB) and the embedding
model (`models/minilm-onnx/`, 23MB) are both committed, and the service uses the
same embedder that built the index — so a fresh clone deploys as-is.

```bash
git add -f data/index models/minilm-onnx
git commit -m "Ask the Commit"
git push
```

> Only rebuild the index (`python ingest.py --force`) if you add episodes or
> change the embedder. It takes seconds — transcripts are cached.

## 2. Get a Groq key

<https://console.groq.com/keys> — free, no card. This is the **only** credential
the service needs; embedding runs locally.

## 3. Create the Render service

Render dashboard → **New** → **Blueprint** → select your repo. It reads
`render.yaml`, creates one free web service, and prompts for the single secret:

| Secret | Where from |
| --- | --- |
| `GROQ_API_KEY` | <https://console.groq.com/keys> |

Prefer clicking through instead of a blueprint? **New → Web Service**, connect the
repo, then set:

- Language / Runtime: **Python 3** (not Docker — see the troubleshooting table)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Plan: **Free**
- Health check path: `/health`

## 4. Verify

```bash
curl -s https://ask-the-commit.onrender.com/health | jq
# {"status":"ok","indexed_chunks":96,"indexed_episodes":3,"boot_seconds":0.2,
#  "cached_answers":0,"cache_hit_rate":0.0,...}
```

| Symptom | Cause |
| --- | --- |
| `status: degraded` | `data/index/` did not make it into the repo — check `git ls-files data/index` |
| Boot fails, "misconfigured" | A secret is unset; the log names exactly which |
| Boot fails, "same model" | `EMBEDDING_PROVIDER` does not match `manifest.json` — rebuild with `ingest.py --force` |
| 502 on `/ask` | Groq unreachable or out of quota |

---

## Keeping it warm

Render's free tier idles a service out after ~15 minutes. `/health` is built for
an external pinger: it answers from a dict cached at startup, touching neither
the index nor any upstream API, and returns in **~1.9ms**. Pinging it costs
nothing and never spends quota.

Point any free uptime monitor (UptimeRobot, cron-job.org, Better Stack) at
`https://<your-service>.onrender.com/health` every 10 minutes.

Render's free plan also caps total monthly instance hours, so a permanent pinger
trades cold starts for hours of that budget. Pinging only around the days you
expect traffic is the better deal.

---

## Verified locally

The exact runtime environment was built and exercised before this was written:

```
slim venv          196MB, 35 packages
                   torch / sentence-transformers / chromadb / faster-whisper: absent
boot               0.20s  (including ONNX model warm-up)
peak RSS           192MB  (Render free tier: 512MB)
GET  /health       1.9ms, no index or API access
GET  /             UI, 200
POST /ask          "Journalists use a Qubes OS-based system..." [5]
                   1539ms - retrieval 3ms (local ONNX), generation 1536ms
POST /ask          refusal short-circuits in 4.5ms, no generation call spent
read-only store    upsert/delete/set_manifest all raise; index mtimes unchanged under load
missing secrets    startup aborts, listing exactly what is missing
```

To repeat it:

```bash
python3.13 -m venv /tmp/slim && /tmp/slim/bin/pip install -r requirements.txt
GROQ_API_KEY=... /tmp/slim/bin/uvicorn main:app --port 8000
```

---

## Protecting the quota

In increasing order of effort:

1. **Reduce `TOP_K` to 3.** Cuts prompt tokens by ~40%, roughly doubling
   questions per minute. Retrieval recall is 100% at k=5, so there is headroom.
2. **Answer caching — already enabled** (`ANSWER_CACHE_SIZE=256`). Repeat
   questions cost no generation call at all: measured 1937ms → 0ms. Since demo
   traffic is mostly the same handful of example questions, this removes most of
   the quota pressure on its own.
3. **Per-IP rate limiting.** `slowapi` is a two-line FastAPI integration.
4. **A shared password**, handed out with the link.
5. **Search-only mode.** `LLM_PROVIDER=echo` never calls a chat model, so there is
   no quota to spend at all — embedding is already local. Visitors get ranked
   transcript passages with timestamps instead of prose.

Option 5 is the only one that makes quota abuse structurally impossible, and with
local embeddings it makes the service entirely free to run.

---

## Other hosts

`Dockerfile.serve` still builds a container of the same service for anywhere that
runs Docker — Cloud Run, Fly, a VPS. Render does not need it; the Python runtime
path above is simpler and faster to deploy.

Note the Dockerfiles are deliberately **not** named `Dockerfile`: Render (and
several other PaaS hosts) auto-detect a root `Dockerfile` and silently prefer it
over a declared runtime. `Dockerfile.ingest` builds the transcription image;
`Dockerfile.serve` builds the query-only one. Build them explicitly:

```bash
docker build -f Dockerfile.serve -t ask-the-commit-serve .
```

For a demo with no hosting at all, a Cloudflare tunnel to your laptop works and
dies when you close it:

```bash
uvicorn main:app --port 8000
cloudflared tunnel --url http://localhost:8000
```
