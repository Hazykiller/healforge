$ErrorActionPreference = "Stop"

python -m pytest -q
python -m compileall -q app
node --check static/app.js

git diff --check

Write-Host "HEALFORGE verification checks passed." -ForegroundColor Green
