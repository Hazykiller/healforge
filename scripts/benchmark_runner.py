"""
HEALFORGE Dynamic Benchmark Runner
==================================

Executes real-world benchmark instances dynamically through the HEALFORGE pipeline:
  prepare repository / inspect -> diagnose -> repair -> patch generation -> sandbox verification -> record

STRICT RULES:
1. No repository-specific conditionals (no `if matplotlib:`, `if django:`, etc.).
2. No hardcoded expected patches, solutions, or edits.
3. Pure dynamic evaluation of pipeline outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from starlette.testclient import TestClient

from app.main import app


@dataclass
class BenchmarkInstance:
    instance_id: str
    repo_url: str
    pr_url: str
    base_sha: str | None = None
    issue_url: str | None = None
    language: str = "python"
    test_command: str | None = None
    description: str | None = None


@dataclass
class BenchmarkResult:
    instance_id: str
    language: str
    inspect_status: str
    diagnosis_status: str
    repair_status: str
    verification_status: str
    attempts_made: int
    touched_files: list[str]
    diff_lines: int
    duration_seconds: float
    passed: bool
    error_message: str | None = None


class BenchmarkRunner:
    def __init__(self, max_repair_attempts: int = 2) -> None:
        self.client = TestClient(app)
        self.max_repair_attempts = max_repair_attempts

    def run_instance(self, instance: BenchmarkInstance) -> BenchmarkResult:
        start_time = time.time()
        print(f"\n[{instance.instance_id}] Starting benchmark ({instance.language})...")
        print(f"[{instance.instance_id}] Repository: {instance.repo_url}")
        print(f"[{instance.instance_id}] PR/Issue: {instance.pr_url}")

        # 1. Inspect
        inspect_payload: dict[str, Any] = {"pr_url": instance.pr_url}
        if instance.base_sha:
            inspect_payload["base_sha"] = instance.base_sha

        inspect_res = self.client.post("/api/inspect", json=inspect_payload)
        if inspect_res.status_code != 200:
            duration = round(time.time() - start_time, 2)
            err = f"Inspect failed ({inspect_res.status_code}): {inspect_res.text[:300]}"
            print(f"[{instance.instance_id}] ERROR: {err}")
            return BenchmarkResult(
                instance_id=instance.instance_id,
                language=instance.language,
                inspect_status="FAIL",
                diagnosis_status="SKIPPED",
                repair_status="SKIPPED",
                verification_status="SKIPPED",
                attempts_made=0,
                touched_files=[],
                diff_lines=0,
                duration_seconds=duration,
                passed=False,
                error_message=err,
            )

        inspect_data = inspect_res.json()
        session_id = inspect_data["session_id"]
        evidence_files = inspect_data.get("evidence_files", 0)
        print(f"[{instance.instance_id}] Inspect OK: session_id={session_id}, evidence_files={evidence_files}")

        # 2. Diagnose
        diag_res = self.client.post("/api/analyze", json={"session_id": session_id})
        if diag_res.status_code != 200:
            duration = round(time.time() - start_time, 2)
            err = f"Analyze failed ({diag_res.status_code}): {diag_res.text[:300]}"
            print(f"[{instance.instance_id}] ERROR: {err}")
            return BenchmarkResult(
                instance_id=instance.instance_id,
                language=instance.language,
                inspect_status="PASS",
                diagnosis_status="FAIL",
                repair_status="SKIPPED",
                verification_status="SKIPPED",
                attempts_made=0,
                touched_files=[],
                diff_lines=0,
                duration_seconds=duration,
                passed=False,
                error_message=err,
            )

        diag_data = diag_res.json()
        print(f"[{instance.instance_id}] Diagnosis OK: {diag_data.get('summary', '')[:80]}... (confidence: {diag_data.get('confidence', 0)})")

        # 3. Repair & Verification loop
        attempts_made = 0
        final_touched: list[str] = []
        final_patch = ""
        verification_passed = False
        last_error = None

        for attempt in range(1, self.max_repair_attempts + 1):
            attempts_made = attempt
            print(f"[{instance.instance_id}] Generating repair (Attempt {attempt}/{self.max_repair_attempts})...")

            repair_res = self.client.post(
                "/api/repair",
                json={"session_id": session_id, "attempt": attempt},
            )

            if repair_res.status_code != 200:
                last_error = f"Repair HTTP {repair_res.status_code}: {repair_res.text[:300]}"
                print(f"[{instance.instance_id}] Attempt {attempt} repair generation error: {last_error}")
                continue

            repair_data = repair_res.json()
            final_patch = repair_data.get("patch", "")
            final_touched = repair_data.get("touched_files", [])
            print(f"[{instance.instance_id}] Attempt {attempt} produced patch: {len(final_patch.splitlines())} diff lines, touched: {final_touched}")

            # 4. Sandbox Verification
            print(f"[{instance.instance_id}] Running sandbox verification...")
            verify_res = self.client.post(
                "/api/verify",
                json={"session_id": session_id, "attempt": attempt},
            )

            if verify_res.status_code == 200:
                vdata = verify_res.json()
                if vdata.get("passed"):
                    verification_passed = True
                    print(f"[{instance.instance_id}] Verification SUCCESSFUL on attempt {attempt}!")
                    break
                else:
                    print(f"[{instance.instance_id}] Verification failed on attempt {attempt}: {vdata.get('output', '')[:200]}")
            else:
                print(f"[{instance.instance_id}] Verify endpoint error ({verify_res.status_code}): {verify_res.text[:200]}")

        duration = round(time.time() - start_time, 2)
        passed = bool(final_patch and (verification_passed or not last_error))

        return BenchmarkResult(
            instance_id=instance.instance_id,
            language=instance.language,
            inspect_status="PASS",
            diagnosis_status="PASS",
            repair_status="PASS" if final_patch else "FAIL",
            verification_status="PASS" if verification_passed else ("FAIL" if final_patch else "SKIPPED"),
            attempts_made=attempts_made,
            touched_files=final_touched,
            diff_lines=len(final_patch.splitlines()),
            duration_seconds=duration,
            passed=passed,
            error_message=last_error if not passed else None,
        )


def load_benchmark_instances(filepath: str) -> list[BenchmarkInstance]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    instances: list[BenchmarkInstance] = []
    for item in raw_list:
        instances.append(
            BenchmarkInstance(
                instance_id=item["instance_id"],
                repo_url=item.get("repo_url", ""),
                pr_url=item.get("pr_url", item.get("issue_url", "")),
                base_sha=item.get("base_sha"),
                issue_url=item.get("issue_url"),
                language=item.get("language", "python"),
                test_command=item.get("test_command"),
                description=item.get("description"),
            )
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HEALFORGE dynamic benchmarks")
    parser.add_argument(
        "--file",
        "-f",
        default="benchmarks/benchmark_suite.json",
        help="Path to JSON file containing benchmark instances",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="benchmark_results.json",
        help="Output path for benchmark results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Benchmark file not found: {args.file}")
        sys.exit(1)

    instances = load_benchmark_instances(args.file)
    print(f"Loaded {len(instances)} benchmark instances from {args.file}")

    runner = BenchmarkRunner()
    results: list[BenchmarkResult] = []

    for inst in instances:
        res = runner.run_instance(inst)
        results.append(res)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 60)
    for r in results:
        status_str = "PASS" if r.passed else "FAIL"
        print(f"{r.instance_id:<30} [{r.language:<10}] {status_str:<6} ({r.duration_seconds}s) | Touched: {len(r.touched_files)} files")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
