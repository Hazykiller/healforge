import base64
import re
from dataclasses import dataclass
from typing import Any

import httpx

API = "https://api.github.com"
API_VERSION = "2026-03-10"


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    number: int


def parse_pr_url(url: str) -> RepoRef:
    match = re.fullmatch(
        r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?",
        url.strip(),
    )
    if not match:
        raise ValueError(
            "Enter a GitHub pull-request URL such as "
            "https://github.com/owner/repo/pull/42"
        )
    return RepoRef(match.group(1), match.group(2), int(match.group(3)))


class GitHubClient:
    def __init__(self, token: str, timeout: float = 30) -> None:
        self.timeout = timeout
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            **({"Authorization": f"Bearer {token}"} if token else {}),
        }
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        response = await self._client.request(
            method,
            API + path,
            **kwargs,
        )

        if response.status_code >= 400:
            detail = response.text[:700].replace("\n", " ")
            raise RuntimeError(
                f"GitHub API {response.status_code}: {detail}"
            )

        return response.json()

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return await self._request("GET", path, params=params)

    async def pull_request(self, ref: RepoRef) -> dict[str, Any]:
        return await self._get(
            f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}"
        )

    async def files(
        self,
        ref: RepoRef,
        max_pages: int = 30,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            data = await self._get(
                f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/files",
                {"per_page": 100, "page": page},
            )

            if not isinstance(data, list) or not data:
                break

            result.extend(data)

            if len(data) < 100:
                break

        return result

    async def checks(
        self,
        ref: RepoRef,
        sha: str,
        max_pages: int = 10,
    ) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            data = await self._get(
                f"/repos/{ref.owner}/{ref.repo}/commits/{sha}/check-runs",
                {"per_page": 100, "page": page},
            )
            batch = data.get("check_runs", []) if isinstance(data, dict) else []
            runs.extend(batch)

            if len(batch) < 100:
                break

        return {"check_runs": runs}

    async def check_annotations(
        self,
        ref: RepoRef,
        run_id: int,
    ) -> list[dict[str, Any]]:
        data = await self._get(
            f"/repos/{ref.owner}/{ref.repo}/check-runs/{run_id}/annotations",
            {"per_page": 100},
        )
        return data if isinstance(data, list) else []

    async def tree(
        self,
        ref: RepoRef,
        sha: str,
    ) -> list[dict[str, Any]]:
        data = await self._get(
            f"/repos/{ref.owner}/{ref.repo}/git/trees/{sha}",
            {"recursive": "1"},
        )

        if not isinstance(data, dict):
            return []

        return data.get("tree", [])

    async def content(
        self,
        ref: RepoRef,
        path: str,
        sha: str,
        max_bytes: int = 1_000_000,
    ) -> str:
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("Unsafe repository path")

        data = await self._get(
            f"/repos/{ref.owner}/{ref.repo}/contents/{path}",
            {"ref": sha},
        )

        if isinstance(data, list):
            return ""

        size = int(data.get("size") or 0)
        if size > max_bytes:
            return "[HEALFORGE: file omitted because it exceeds the size limit]"

        encoded = data.get("content", "")
        if not encoded:
            return ""

        try:
            return base64.b64decode(encoded).decode(
                "utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise RuntimeError(f"Could not decode repository file: {path}") from exc

    async def create_branch(
        self,
        ref: RepoRef,
        base_sha: str,
        branch: str,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/repos/{ref.owner}/{ref.repo}/git/refs",
            json={
                "ref": f"refs/heads/{branch}",
                "sha": base_sha,
            },
        )

    async def update_file(
        self,
        ref: RepoRef,
        path: str,
        message: str,
        content: str,
        sha: str,
        branch: str,
    ) -> dict[str, Any]:
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "sha": sha,
            "branch": branch,
        }
        return await self._request(
            "PUT",
            f"/repos/{ref.owner}/{ref.repo}/contents/{path}",
            json=payload,
        )

    async def create_pr(
        self,
        ref: RepoRef,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/repos/{ref.owner}/{ref.repo}/pulls",
            json={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
            },
        )
