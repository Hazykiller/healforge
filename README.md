# HEALFORGE

Autonomous PR reviewer and repair engine for Orchestrate PS #07 — Self-Heal Git.

HEALFORGE is intentionally an engineering pipeline, not a chatbot wrapper:

1. Pull request metadata, changed files and CI checks are collected from GitHub.
2. The changed surface is bounded and relevant repository context is assembled.
3. The AI diagnoses the failure and proposes a minimal unified diff.
4. The patch is applied to an isolated local checkout.
5. A deterministic test command is selected from the repository and executed in Docker when available.
6. Failed repairs are rejected; a second repair attempt can be generated from the new evidence.
7. A verified patch can be exported and, when write credentials are supplied, a fix branch and PR can be created.

No fake metrics, placeholder results, hard-coded repository data, or fake AI responses are used.

## Requirements

- Python 3.11+
- Git
- Docker Desktop (strongly recommended; required for safe arbitrary repository test execution)
- GitHub token with repository contents/pull-request read access; write access is required only to create a fix branch/PR
- OpenAI-compatible API key

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Environment

`GITHUB_TOKEN` — fine-grained GitHub token.
`OPENAI_API_KEY` — model API key.
`OPENAI_MODEL` — model available to your API account. The default is configurable; set it explicitly for your account.
`MAX_CONTEXT_CHARS` — maximum repository context sent to the model.
`MAX_PATCH_CHARS` — maximum generated patch size.
`DOCKER_IMAGE_PYTHON` — sandbox image for Python repositories.
`DOCKER_IMAGE_NODE` — sandbox image for Node repositories.
`ALLOW_WRITE_ACTIONS=false` — safety default. Set true only when you want HEALFORGE to create a fix branch/PR.

## What is implemented

- GitHub PR inspection
- Changed-file retrieval
- CI check retrieval
- Repository checkout at the PR head SHA
- Language/test-command detection for common Python and Node projects
- Docker sandbox execution with network disabled and resource limits
- AI root-cause diagnosis
- Minimal unified-diff repair generation
- Patch application and verification
- One evidence-driven repair retry
- Dynamic web dashboard
- Patch download
- Optional GitHub fix branch + PR creation

## Safety boundary

Never run untrusted repositories directly on the host. HEALFORGE uses Docker with `--network none`, memory/CPU/PID limits, a read-only root filesystem, a temporary writable work directory, and a timeout. Docker is intentionally required for execution of arbitrary repository code.

## Suggested demo

Use a small public repository you control. Create a PR that introduces a deterministic test failure. Paste its PR URL into the dashboard. Show the failure evidence, diagnosis, candidate patch, sandbox verification, and generated PR.

## Competition alignment

Orchestrate requires a real end-to-end product, a central AI component, a polished interface, empirical verification, a public GitHub repository, setup instructions and development history. HEALFORGE is structured around those requirements.
