import base64
import re
from dataclasses import dataclass
from typing import Any
import httpx

API = "https://api.github.com"

@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str
    number: int


def parse_pr_url(url: str) -> RepoRef:
    match = re.fullmatch(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)/?", url.strip())
    if not match:
        raise ValueError("Enter a GitHub pull-request URL such as https://github.com/owner/repo/pull/42")
    return RepoRef(match.group(1), match.group(2), int(match.group(3)))

class GitHubClient:
    def __init__(self, token: str):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        }

    async def _get(self, path: str, params: dict[str, Any] | None = None):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(API + path, headers=self.headers, params=params)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub API {response.status_code}: {response.text[:500]}")
        return response.json()

    async def pull_request(self, ref: RepoRef):
        return await self._get(f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}")

    async def files(self, ref: RepoRef):
        return await self._get(f"/repos/{ref.owner}/{ref.repo}/pulls/{ref.number}/files", {"per_page": 100})

    async def checks(self, ref: RepoRef, sha: str):
        return await self._get(f"/repos/{ref.owner}/{ref.repo}/commits/{sha}/check-runs", {"per_page": 100})

    async def content(self, ref: RepoRef, path: str, sha: str):
        data = await self._get(f"/repos/{ref.owner}/{ref.repo}/contents/{path}", {"ref": sha})
        if isinstance(data, list):
            return ""
        raw = data.get("content", "")
        if not raw:
            return ""
        return base64.b64decode(raw).decode("utf-8", errors="replace")

    async def create_branch(self, ref: RepoRef, base_sha: str, branch: str):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(API + f"/repos/{ref.owner}/{ref.repo}/git/refs", headers=self.headers, json={"ref": f"refs/heads/{branch}", "sha": base_sha})
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub branch creation failed: {response.text[:500]}")
        return response.json()

    async def update_file(self, ref: RepoRef, path: str, message: str, content: str, sha: str, branch: str):
        payload = {"message": message, "content": base64.b64encode(content.encode()).decode(), "sha": sha, "branch": branch}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(API + f"/repos/{ref.owner}/{ref.repo}/contents/{path}", headers=self.headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub file update failed: {response.text[:500]}")
        return response.json()

    async def create_pr(self, ref: RepoRef, head: str, base: str, title: str, body: str):
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(API + f"/repos/{ref.owner}/{ref.repo}/pulls", headers=self.headers, json={"title": title, "head": head, "base": base, "body": body})
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub PR creation failed: {response.text[:500]}")
        return response.json()
