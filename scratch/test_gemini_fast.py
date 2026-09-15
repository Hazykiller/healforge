import os
import time
import httpx
import dotenv

dotenv.load_dotenv()
key = os.getenv("GEMINI_API_KEY")

for m in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash"]:
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": "Return JSON: {\"status\": \"success\"}"}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
            },
            timeout=10.0,
        )
        dt = time.perf_counter() - t0
        print(f"{m}: status={r.status_code} in {dt:.2f}s -> {r.text[:60]}")
    except Exception as e:
        print(f"{m}: error={e}")
