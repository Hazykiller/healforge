import json
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # keeps non-AI health/inspection imports usable in minimal environments
    OpenAI = None  # type: ignore[assignment]

from .config import settings
from .security import is_safe_repo_path, is_sensitive_path


class AIEngine:
    """Reliable OpenRouter-backed diagnosis and repair engine."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if OpenAI is None:
            raise RuntimeError("The openai package is not installed. Run pip install -r requirements.txt")

        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.ai_timeout_seconds,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://tcet-openai-it.vercel.app",
                "X-Title": "HEALFORGE",
            },
        )

    def _call_once(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            extra_body={
                # OpenRouter performs ordered model fallback server-side.
                # Passing the route once avoids duplicate sequential calls.
                "models": settings.ai_models,
            },
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(f"Model {model} returned no choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None

        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        if not content or not str(content).strip():
            raise RuntimeError(f"Model {model} returned an empty response")

        return str(content).strip()

    def _call_model(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> str:
        # OpenRouter handles ordered model fallback. We retry the same routed
        # request once only for a transport/empty-response failure so the
        # provider gets one chance to recover without us calling every model.
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return self._call_once(
                    settings.ai_models[0],
                    system,
                    user,
                    temperature,
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"AI request failed after configured model fallback: {type(last_error).__name__}: {last_error}"
        ) from last_error

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | None:
        text = content.strip()

        candidates = [text]

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            candidates.append(fenced.group(1).strip())

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        return None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return [value]
        return [str(value)]

    @staticmethod
    def _confidence(value: Any, default: float = 0.8) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
        return max(0.0, min(1.0, result))

    @classmethod
    def _normalize_diagnosis(cls, result: dict[str, Any]) -> dict[str, Any]:
        required = [
            "summary",
            "root_cause",
            "repair_strategy",
        ]
        missing = [key for key in required if not result.get(key)]
        if missing:
            raise RuntimeError(
                "Diagnosis is missing required fields: "
                + ", ".join(missing)
            )

        return {
            "summary": str(result.get("summary", "")),
            "root_cause": str(result.get("root_cause", "")),
            "confidence": cls._confidence(result.get("confidence")),
            "affected_files": [str(x) for x in cls._as_list(result.get("affected_files"))],
            "evidence": [str(x) for x in cls._as_list(result.get("evidence"))],
            "repair_strategy": str(result.get("repair_strategy", "")),
            "risk_notes": [str(x) for x in cls._as_list(result.get("risk_notes"))],
        }

    def diagnose(self, context: str) -> dict[str, Any]:
        system = """
You are HEALFORGE's software failure diagnosis engine.

Analyze the supplied pull-request evidence and identify the smallest root
cause supported by evidence. Trace dependencies and tests when relevant.

Return JSON only with:
summary, root_cause, confidence, affected_files, evidence,
repair_strategy, risk_notes

confidence is 0..1. Arrays may contain strings. Do not invent files,
dependencies, APIs, test results, CI results, or repository state. If the
evidence is insufficient, say that explicitly and recommend a safe refusal.
""".strip()

        raw = self._call_model(system, context, temperature=0.1)
        parsed = self._extract_json(raw)

        if parsed is None:
            # One focused formatting retry is safer than crashing on a model
            # that answered correctly but wrapped its JSON unexpectedly.
            retry_system = system + "\nReturn one JSON object and nothing else."
            raw = self._call_model(retry_system, context, temperature=0.0)
            parsed = self._extract_json(raw)

        if parsed is None:
            raise RuntimeError("Diagnosis model response could not be parsed as JSON")

        return self._normalize_diagnosis(parsed)

    def generate_patch(
        self,
        context: str,
        diagnosis: dict[str, Any],
        verification_feedback: str = "",
    ) -> dict[str, Any]:
        feedback = verification_feedback.strip()

        system = """
You are HEALFORGE's autonomous repair engine.

Produce the smallest safe unified diff that fixes the diagnosed failure.
Only modify files explicitly present in the repository evidence.
Do not invent files or dependencies. Do not rewrite unrelated code.
Prefer a minimal multi-file repair when the root cause genuinely crosses
files. The patch must be standard unified diff syntax with a/ and b/ paths.

Return JSON only:
{
  "patch": "unified diff",
  "explanation": "why this fixes the root cause",
  "touched_files": ["path"],
  "confidence": 0.0
}

If no safe repair can be supported by evidence, return an empty patch.
""".strip()

        user = (
            "DIAGNOSIS:\n"
            + json.dumps(diagnosis, indent=2)
            + "\n\nREPOSITORY EVIDENCE:\n"
            + context
        )

        if feedback:
            user += (
                "\n\nPREVIOUS VERIFICATION FAILURE:\n"
                + feedback
                + "\n\nRepair the diagnosed issue while explicitly addressing this verification failure."
            )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = self._call_once(
                    settings.ai_models[0],
                    system if attempt == 0 else system + "\nReturn ONLY the JSON object; do not wrap it in markdown.",
                    user,
                    temperature=0.0,
                )
                result = self._parse_repair_response(raw, context)
                self._validate_patch(result)
                return result
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"No configured repair model produced a safe unified diff: {last_error}")

    def _parse_repair_response(
        self,
        raw: str,
        context: str,
    ) -> dict[str, Any]:
        parsed = self._extract_json(raw)

        if parsed is not None:
            patch = parsed.get("patch", "")
            if isinstance(patch, str):
                patch = self._extract_diff(patch)
                if patch:
                    touched = parsed.get("touched_files")
                    if not isinstance(touched, list):
                        touched = self._files_from_patch(patch)
                    return {
                        "patch": patch,
                        "explanation": str(
                            parsed.get(
                                "explanation",
                                "Minimal evidence-backed repair.",
                            )
                        ),
                        "touched_files": [str(x) for x in touched],
                        "confidence": self._confidence(
                            parsed.get("confidence"),
                            0.8,
                        ),
                    }

        diff = self._extract_diff(raw)
        if diff:
            return {
                "patch": diff,
                "explanation": "Model returned a unified diff directly.",
                "touched_files": self._files_from_patch(diff),
                "confidence": 0.8,
            }

        raise RuntimeError("The repair model did not produce a usable unified diff")

    @staticmethod
    def _extract_diff(content: str) -> str:
        text = content.strip()
        start = re.search(r"(?m)^---\s+a/\S+", text)
        if not start:
            return ""

        diff = text[start.start():].strip()
        diff = re.sub(r"\n```(?:diff)?\s*$", "", diff, flags=re.IGNORECASE).strip()

        if not re.search(r"(?m)^\+\+\+\s+b/\S+", diff):
            return ""
        if not re.search(r"(?m)^@@", diff):
            return ""

        return diff + "\n"

    @staticmethod
    def _files_from_patch(patch: str) -> list[str]:
        result: list[str] = []
        for match in re.finditer(r"(?m)^\+\+\+\s+b/(.+)$", patch):
            path = match.group(1).strip()
            if path not in result:
                result.append(path)
        return result

    @staticmethod
    def _validate_patch(result: dict[str, Any]) -> None:
        patch = result.get("patch", "")
        if not isinstance(patch, str):
            raise RuntimeError("Repair patch must be a string")
        if len(patch) > settings.max_patch_chars:
            raise RuntimeError("Generated patch exceeds configured safety limit")
        if not isinstance(result.get("touched_files"), list):
            raise RuntimeError("Repair touched_files must be an array")
        if not 0 <= float(result.get("confidence", 0.8)) <= 1:
            raise RuntimeError("Repair confidence must be between 0 and 1")

        if not patch:
            return

        if not re.search(r"(?m)^---\s+a/[^\s]+", patch):
            raise RuntimeError("Repair is not a valid unified diff")
        if not re.search(r"(?m)^\+\+\+\s+b/[^\s]+", patch):
            raise RuntimeError("Repair does not contain a valid target file")
        if not re.search(r"(?m)^@@", patch):
            raise RuntimeError("Repair does not contain a unified-diff hunk")

        for path in AIEngine._files_from_patch(patch):
            normalized = path.replace("\\", "/")
            if not is_safe_repo_path(normalized):
                raise RuntimeError("Repair contains an unsafe path")
            if is_sensitive_path(normalized):
                raise RuntimeError("Repair attempts to modify a sensitive file")
