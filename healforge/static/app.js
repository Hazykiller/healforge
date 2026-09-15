const $ = (id) => document.getElementById(id);

let sessionId = null;
let maxRepairAttempts = 2;
let currentStage = "stageInspect";

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    }
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail ||
      data.message ||
      `Request failed (${response.status})`
    );
  }

  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function setStage(id, state, label) {
  const element = $(id);
  if (!element) return;

  element.className = `stage ${state}`;
  const status = element.querySelector("em");
  if (status) status.textContent = label;
}

function showError(message) {
  const element = $("config");
  if (!element) return;
  if (!element.dataset.defaultText) element.dataset.defaultText = element.textContent || "";
  element.textContent = message;
  element.style.color = "#ff6b6b";
}

function clearError() {
  const element = $("config");
  if (!element) return;
  element.textContent = element.dataset.defaultText || element.textContent;
  element.style.color = "";
}

async function inspect() {
  const input = $("prUrl");
  if (!input) return;

  const url = input.value.trim();
  if (!url) {
    showError("Paste a GitHub pull-request URL.");
    return;
  }

  const button = $("inspect");
  button.disabled = true;
  button.textContent = "Inspecting…";
  currentStage = "stageInspect";
  clearError();
  setStage("stageInspect", "active", "RUNNING");

  try {
    const data = await api("/api/inspect", {
      method: "POST",
      body: JSON.stringify({ pr_url: url })
    });

    sessionId = data.session_id;
    if (!sessionId) throw new Error("GitHub inspection returned no session ID.");

    renderPullRequest(data);
    $("workspace").classList.remove("hidden");
    setStage("stageInspect", "done", "READY");
    $("config").textContent =
      `Evidence loaded · ${data.evidence_files || 0} files · ${Number(data.context_chars || 0).toLocaleString()} chars`;
  } catch (error) {
    setStage("stageInspect", "fail", "ERROR");
    showError(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Inspect PR <span>→</span>';
  }
}

function renderPullRequest(data) {
  const pr = data.pr || {};
  const files = Array.isArray(data.files) ? data.files : [];
  const checks = Array.isArray(data.checks) ? data.checks : [];

  const title = escapeHtml(pr.title || "Untitled pull request");
  const number = escapeHtml(pr.number ?? "—");
  const url = escapeHtml(pr.url || "#");
  const sha = escapeHtml((pr.head_sha || "").slice(0, 12));

  $("prMeta").innerHTML = `
    <div class="result-title">
      <h2>${title}</h2>
      <span class="pill">PR #${number}</span>
    </div>
    <div class="hint">
      <a href="${url}" target="_blank" rel="noopener noreferrer">Open on GitHub ↗</a>
      ${sha ? `<br>HEAD ${sha}` : ""}
    </div>
  `;

  $("fileCount").textContent = files.length;
  $("checkCount").textContent = checks.length;
  $("contextChars").textContent = Number(data.context_chars || 0).toLocaleString();
  $("files").innerHTML = files.length
    ? files.map(renderFile).join("")
    : `<div class="hint">No changed files returned.</div>`;
}

function renderFile(file) {
  const path = escapeHtml(file.path || "unknown file");
  const status = escapeHtml(file.status || "modified");
  const additions = Number(file.additions || 0);
  const deletions = Number(file.deletions || 0);

  return `
    <div class="file">
      <span>${path}</span>
      <em>${status} · +${additions}/-${deletions}</em>
    </div>
  `;
}

async function recover() {
  if (!sessionId) {
    showError("Inspect a pull request before running recovery.");
    return;
  }

  const button = $("runRecovery");
  button.disabled = true;
  button.textContent = "Running recovery…";
  clearError();
  hideResultCards();

  try {
    currentStage = "stageDiagnose";
    setStage("stageDiagnose", "active", "RUNNING");
    const diagnosis = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId })
    });
    setStage("stageDiagnose", "done", "COMPLETE");
    renderDiagnosis(diagnosis);

    const maxAttempts = Math.max(1, Number(maxRepairAttempts) || 2);
    let verified = false;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      currentStage = "stageRepair";
      setStage("stageRepair", "active", attempt === 1 ? "RUNNING" : `RETRY ${attempt}`);

      const repair = await api("/api/repair", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, attempt })
      });

      setStage("stageRepair", "done", `GENERATED #${attempt}`);
      renderRepair(repair, attempt);

      currentStage = "stageVerify";
      setStage("stageVerify", "active", `VERIFY #${attempt}`);

      const verification = await api("/api/verify", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, attempt })
      });

      verified = Boolean(verification.passed);
      renderVerification(verification, attempt);

      if (verified) {
        setStage("stageVerify", "done", "VERIFIED");
        break;
      }

      setStage("stageVerify", "fail", attempt < maxAttempts ? "RETRYING" : "REJECTED");
      if (attempt < maxAttempts) {
        showError("Attempt failed verification. HEALFORGE is feeding the sandbox failure back into the repair engine.");
      }
    }

    if (!verified) {
      showError("HEALFORGE could not produce a verified repair after the available attempts.");
    } else {
      clearError();
    }
  } catch (error) {
    setStage(currentStage, "fail", "ERROR");
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run recovery";
  }
}

function renderDiagnosis(diagnosis) {
  const confidence = Number(diagnosis.confidence || 0);
  const evidence = Array.isArray(diagnosis.evidence) ? diagnosis.evidence : [diagnosis.evidence].filter(Boolean);

  $("diagnosisCard").classList.remove("hidden");
  $("diagnosisCard").innerHTML = `
    <div class="result-title">
      <h2>Root-cause diagnosis</h2>
      <span class="pill good">${Math.round(confidence * 100)}% confidence</span>
    </div>
    <div class="kv">
      <b>Summary</b><span>${escapeHtml(diagnosis.summary)}</span>
      <b>Root cause</b><span>${escapeHtml(diagnosis.root_cause)}</span>
      <b>Repair strategy</b><span>${escapeHtml(diagnosis.repair_strategy)}</span>
    </div>
    ${evidence.length ? `<div class="hint">Evidence</div><div class="content">${evidence.map((item) => `• ${escapeHtml(item)}`).join("<br>")}</div>` : ""}
  `;
}

function renderRepair(repair, attempt) {
  const confidence = Number(repair.confidence || 0);
  $("patchCard").classList.remove("hidden");
  $("patchCard").innerHTML = `
    <div class="result-title">
      <h2>Candidate repair · attempt ${attempt}</h2>
      <span class="pill">${Math.round(confidence * 100)}% confidence</span>
    </div>
    <p class="content">${escapeHtml(repair.explanation)}</p>
    <pre class="code">${escapeHtml(repair.patch || "No safe patch generated.")}</pre>
    <div class="actions">
      <a class="secondary" href="/api/session/${encodeURIComponent(sessionId)}/patch">Download patch</a>
    </div>
  `;
}

function renderVerification(verification, attempt) {
  const verified = Boolean(verification.passed);
  const command = escapeHtml(verification.command || "Verification command unavailable");
  const output = escapeHtml(verification.output || verification.reason || "No verification output returned.");
  const exitCode = verification.exit_code == null ? "—" : escapeHtml(verification.exit_code);

  $("verifyCard").classList.remove("hidden");
  $("verifyCard").innerHTML = `
    <div class="result-title">
      <h2>Sandbox verification · attempt ${attempt}</h2>
      <span class="pill ${verified ? "good" : "bad"}">${verified ? "VERIFIED" : "REJECTED"}</span>
    </div>
    <div class="kv">
      <b>Test command</b><span>${command}</span>
      <b>Exit code</b><span>${exitCode}</span>
    </div>
    <pre class="code">${output}</pre>
    ${verified ? `<div class="actions"><a class="secondary" href="/api/session/${encodeURIComponent(sessionId)}/patch">Download verified patch</a></div>` : `<div class="hint">The sandbox rejected this candidate. The next attempt receives this failure output.</div>`}
  `;
}

function hideResultCards() {
  ["diagnosisCard", "patchCard", "verifyCard"].forEach((id) => {
    const element = $(id);
    if (element) {
      element.classList.add("hidden");
      element.innerHTML = "";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  $("inspect")?.addEventListener("click", inspect);
  $("runRecovery")?.addEventListener("click", recover);
  $("prUrl")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      inspect();
    }
  });

  api("/api/health")
    .then((health) => {
      const config = $("config");
      if (!config) return;
      config.textContent = health.ai_configured
        ? `AI ready · ${health.ai_models?.length || 1} model route(s)`
        : "AI key missing";
      config.style.color = "";
    })
    .catch((error) => showError(error.message));
});
