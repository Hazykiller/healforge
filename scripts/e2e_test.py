import urllib.request
import urllib.error
import json
import sys
import time

API_URL = "http://127.0.0.1:8000/api"

def post(path, payload):
    req = urllib.request.Request(f"{API_URL}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTPError {e.code}:", e.read().decode())
        sys.exit(1)

print("1. Inspecting PR...")
data = post("/inspect", {"pr_url": "https://github.com/Hazykiller/healforge-challenge-python/pull/1"})
sid = data["session_id"]
print(f"Session ID: {sid}")

print("2. Diagnosing...")
diagnosis = post("/analyze", {"session_id": sid})
print("Diagnosis:", diagnosis.get("summary"))

for attempt in range(1, 4):
    print(f"3. Generating Repair (Attempt {attempt})...")
    repair_data = post("/repair", {"session_id": sid, "attempt": attempt})
    print("Patch generated:")
    print(repair_data.get("patch", ""))

    print(f"4. Verifying (Attempt {attempt})...")
    v_data = post("/verify", {"session_id": sid, "attempt": attempt})
    print("Verification result:", v_data.get("passed"))
    
    if v_data.get("passed"):
        print("SUCCESS! The repair was verified.")
        sys.exit(0)
    else:
        print("Verification failed. Output:")
        print(v_data.get("output", ""))
        print("Retrying...")

print("FAILED: Maximum attempts reached.")
sys.exit(1)
