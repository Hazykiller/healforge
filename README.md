# HEALFORGE

**Autonomous PR diagnosis, repair, and sandbox verification.**

HEALFORGE turns a failing GitHub pull request into an evidence-backed,
test-verified candidate repair.

```text
GitHub PR
   ↓
Repository evidence
   ↓
AI diagnosis
   ↓
Minimal unified diff
   ↓
Isolated Docker verification
   ↓
PASS → verified patch
   │
   └── FAIL → verification feedback → repair attempt 2
```

## Core capabilities

- GitHub pull-request inspection
- Paginated PR-file retrieval
- Git tree inspection
- CI check and annotation evidence
- Repository-aware evidence ranking
- Local-import discovery
- Test-file discovery
- OpenRouter model failover
- Robust JSON normalization
- Unified-diff validation
- Multi-file repair support
- Verification-feedback repair retry
- Python, Node.js/TypeScript, Java, Go, Rust and C/C++ project detection
- Docker sandbox verification
- Sensitive-file and traversal protection
- Optional verified-PR publishing

## AI routing

HEALFORGE features generic, dynamic model routing configurable via environment variables:

1. Primary model (default: `openrouter/free` - OpenRouter's dynamic free-model router)
2. Fallback models (e.g. `nex-agi/nex-n2.5-mini:free`, `cohere/north-mini-code:free`)

You can configure models via `.env` without modifying Python code:
```bash
# Set primary model
OPENROUTER_MODEL=openrouter/free

# Set fallback models (comma-separated)
OPENROUTER_FALLBACK_MODELS=nex-agi/nex-n2.5-mini:free,cohere/north-mini-code:free

# Or specify a complete prioritized chain:
# OPENROUTER_MODELS=openrouter/free,nex-agi/nex-n2.5-mini:free,cohere/north-mini-code:free
```

Model failure classification:
- **Account Daily Quota Exhausted (`free-models-per-day` / 429)**: Halts immediately to prevent quota waste, caches status, and returns a clean user-facing error with reset time.
- **Model Rate Limit (429) / Timeout / 404 / Connection Error**: Proceeds directly to the next configured fallback model.
- **Syntax / Formatting / Test Edit Error**: Bounded normalization attempt.

## Local setup

1. Copy `.env.example` to `.env`.
2. Put your GitHub and OpenRouter credentials in `.env`.
3. Create/activate a Python 3.11+ virtual environment.
4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Run verification:

```powershell
python -m pytest -q
python -m compileall -q app
node --check static/app.js
```

6. Start:

```powershell
python -m uvicorn app.main:app --log-level debug
```

Do not use `--reload` when `workspace/` is being used for sandbox artifacts.

## Docker

Docker Desktop is required for actual sandbox verification.

The build context is sanitized before Docker sees it. Sensitive files and
unsafe symlinks are excluded/rejected. The build may need network access to
prepare third-party dependencies; the actual test execution container runs
with networking disabled. Runtime repository files are copied into an
ephemeral writable `/work` tmpfs while the container root remains read-only.
The sandbox also drops Linux capabilities and enables `no-new-privileges`.

## Write actions

Publishing is disabled by default. Enable it deliberately with:

```text
ALLOW_WRITE_ACTIONS=true
```

Only verified patches can be published.

## Testing

The repository includes unit and API-contract tests for:

- AI failover
- empty model responses
- diagnosis normalization
- patch parsing/security
- evidence construction
- repository detection
- sandbox-context secret exclusion
- FastAPI inspection/diagnosis flow
- verification-feedback retry plumbing

The development validation suite in this package passes all included tests in
the offline test environment. Docker execution itself requires Docker Desktop.
