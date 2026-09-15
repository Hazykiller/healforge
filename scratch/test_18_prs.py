import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

PRS = [
    ("Requests #7388", "https://github.com/psf/requests/pull/7388"),
    ("Requests #7217", "https://github.com/psf/requests/pull/7217"),
    ("Requests #7213", "https://github.com/psf/requests/pull/7213"),
    ("HTTPX #3766", "https://github.com/encode/httpx/pull/3766"),
    ("HTTPX #3771", "https://github.com/encode/httpx/pull/3771"),
    ("HTTPX #3769", "https://github.com/encode/httpx/pull/3769"),
    ("FastAPI #16265", "https://github.com/fastapi/fastapi/pull/16265"),
    ("FastAPI #16245", "https://github.com/fastapi/fastapi/pull/16245"),
    ("FastAPI #15641", "https://github.com/fastapi/fastapi/pull/15641"),
    ("TypeScript #63440", "https://github.com/microsoft/TypeScript/pull/63440"),
    ("TypeScript #63730", "https://github.com/microsoft/TypeScript/pull/63730"),
    ("TypeScript #63415", "https://github.com/microsoft/TypeScript/pull/63415"),
    ("Go #79300", "https://github.com/golang/go/pull/79300"),
    ("Go #79297", "https://github.com/golang/go/pull/79297"),
    ("Go #79312", "https://github.com/golang/go/pull/79312"),
    ("Rust/Serde #3055", "https://github.com/serde-rs/serde/pull/3055"),
    ("Rust/Serde #3062", "https://github.com/serde-rs/serde/pull/3062"),
    ("Rust/Serde #3024", "https://github.com/serde-rs/serde/pull/3024"),
]

def post(path, body, timeout=60):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8")), e.code
        except Exception:
            return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 0

def run():
    print("=" * 60)
    print("HEALFORGE 18-PR BENCHMARK (PRIMARY MODEL: GEMINI)")
    print("=" * 60)

    results = []
    for idx, (label, pr_url) in enumerate(PRS, 1):
        print(f"\n[{idx}/18] {label}: {pr_url}")
        t0 = time.time()

        # Step 1: Inspect
        print("  -> Inspecting...")
        res_insp, code_insp = post("/api/inspect", {"pr_url": pr_url}, timeout=60)
        if code_insp != 200 or "session_id" not in res_insp:
            err = res_insp.get("detail", res_insp.get("error", "inspect_failed"))
            print(f"  ❌ Inspect failed ({code_insp}): {err}")
            results.append({"label": label, "status": "INSPECT_FAIL", "error": str(err)})
            continue

        sid = res_insp["session_id"]
        files_cnt = res_insp.get("evidence_files", 0)
        chars_cnt = res_insp.get("context_chars", 0)
        print(f"  ✅ Inspect OK: {files_cnt} files, {chars_cnt} chars (sid={sid[:8]})")

        # Step 2: Analyze
        print("  -> Diagnosing (Gemini)...")
        res_diag, code_diag = post("/api/analyze", {"session_id": sid}, timeout=60)
        if code_diag != 200:
            err = res_diag.get("detail", res_diag.get("error", "analyze_failed"))
            print(f"  ❌ Diagnose failed ({code_diag}): {err}")
            results.append({"label": label, "status": "DIAGNOSE_FAIL", "error": str(err)})
            continue

        summary = res_diag.get("summary", "")
        print(f"  ✅ Diagnosis: {summary[:80]}...")

        # Step 3: Repair
        print("  -> Generating Patch (Gemini)...")
        res_repair, code_repair = post("/api/repair", {"session_id": sid, "attempt": 1}, timeout=60)
        if code_repair != 200:
            err = res_repair.get("detail", res_repair.get("error", "repair_failed"))
            print(f"  ❌ Repair failed ({code_repair}): {err}")
            results.append({"label": label, "status": "REPAIR_FAIL", "error": str(err)})
            continue

        patch = res_repair.get("patch", "")
        patch_len = len(patch)
        patch_lines = len(patch.splitlines()) if patch else 0
        elapsed = round(time.time() - t0, 1)
        print(f"  ✅ Patch generated: {patch_lines} lines, {patch_len} chars ({elapsed}s)")
        results.append({
            "label": label,
            "status": "SUCCESS",
            "files": files_cnt,
            "patch_lines": patch_lines,
            "elapsed": elapsed,
            "summary": summary[:100]
        })

    print("\n" + "=" * 60)
    print("FINAL 18-PR BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
        lines = r.get("patch_lines", 0)
        elapsed = r.get("elapsed", 0)
        print(f"{status_icon} {r['label']:<20} {r['status']:<15} {lines} lines ({elapsed}s)")

if __name__ == "__main__":
    run()
