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

Default route:

1. NVIDIA Nemotron 3 Ultra (free)
2. Qwen3 Coder (free)
3. OpenRouter free-model router

These are configurable through `.env`.

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

The image build has network access only during dependency preparation. The
actual test execution container runs with networking disabled and a restricted
filesystem/capability set.

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
