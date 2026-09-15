# HEALFORGE Security Model

## Secrets

- `.env` is intentionally excluded from Git and Docker build contexts.
- API keys are read only from environment variables.
- HEALFORGE never includes `.env`, private keys, or certificate files in AI evidence.
- Common GitHub and OpenRouter token formats are redacted from model context.

## Patch safety

Generated patches are checked before `git apply`.

Rejected paths include:

- absolute paths
- `../` traversal
- `.env` files
- private-key filenames

## Sandbox

The actual verification container uses:

- `--network none`
- `--read-only`
- `--cap-drop ALL`
- `--security-opt no-new-privileges:true`
- CPU limit
- memory limit
- PID limit
- temporary writable `/tmp`

Dependencies are installed while building the image. The patched code is then
executed with networking disabled.

## Write actions

GitHub branch/PR creation is disabled by default.

Set:

```text
ALLOW_WRITE_ACTIONS=true
```

only when you intentionally want HEALFORGE to publish a verified repair.

## Trust boundary

AI output is treated as untrusted input. A model suggestion is not considered
a repair until the patch applies cleanly and the sandbox verifies it.
