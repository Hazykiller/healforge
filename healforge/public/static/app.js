const $ = (id) => document.getElementById(id);

let sessionId = null;
let currentStage = "stageInspect";
let maxRepairAttempts = 2;
let defaultHealthText = "";
let defaultHealthColor = "";

function resetWorkspaceUI(keepUrl = false) {
  sessionId = null;
  currentStage = "stageInspect";
  clearError();
  hideResultCards();

  // Reset stage indicators
  setStage("stageInspect", "active", keepUrl ? "RUNNING" : "READY");
  setStage("stageDiagnose", "", "WAITING");
  setStage("stageRepair", "", "WAITING");
  setStage("stageVerify", "", "WAITING");

  // Reset Run Recovery button
  const runBtn = $("runRecovery");
  if (runBtn) {
    runBtn.disabled = false;
    runBtn.textContent = "Run recovery";
  }

  // Clear evidence pane
  const prMeta = $("prMeta");
  if (prMeta) prMeta.innerHTML = "";
  const fileCount = $("fileCount");
  if (fileCount) fileCount.textContent = "0";
  const checkCount = $("checkCount");
  if (checkCount) checkCount.textContent = "0";
  const contextChars = $("contextChars");
  if (contextChars) contextChars.textContent = "0";
  const files = $("files");
  if (files) files.innerHTML = "";

  if (!keepUrl) {
    const input = $("prUrl");
    if (input) {
      input.value = "";
      input.focus();
    }
    const workspace = $("workspace");
    if (workspace) workspace.classList.add("hidden");

    const config = $("config");
    if (config && defaultHealthText) {
      config.textContent = defaultHealthText;
      config.style.color = defaultHealthColor;
    }
  }
}

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
    let errorMsg = `Request failed (${response.status})`;
    if (data && typeof data.detail === "object" && data.detail !== null) {
      if (data.detail.error === "AI_QUOTA_EXHAUSTED") {
        let msg = "AI SERVICE LIMITED: OpenRouter's current free-model quota has been exhausted. No repair request was attempted further to avoid wasting quota.";
        if (data.detail.reset_timestamp) {
          try {
            const resetDate = new Date(Number(data.detail.reset_timestamp) * 1000);
            if (!isNaN(resetDate.getTime())) {
              msg += ` (Reset expected at ${resetDate.toLocaleTimeString()})`;
            }
          } catch (_) {}
        }
        if (data.detail.remedy_hint) {
          msg += ` Hint: ${data.detail.remedy_hint}`;
        }
        errorMsg = msg;
      } else {
        errorMsg = data.detail.message || JSON.stringify(data.detail);
      }
    } else if (typeof data.detail === "string") {
      errorMsg = data.detail;
    } else if (data.message) {
      errorMsg = data.message;
    }
    throw new Error(errorMsg);
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

  element.className = `stage ${state}`.trim();
  const status = element.querySelector("em");
  if (status) status.textContent = label;
}

function showError(message) {
  const element = $("notice");
  if (!element) return;
  element.textContent = message;
  element.classList.remove("hidden");
  element.classList.remove("info");
  element.classList.add("error");
}

function showNotice(message, type = "info") {
  const element = $("notice");
  if (!element) return;
  element.textContent = message;
  element.classList.remove("hidden");
  element.classList.remove("error");
  element.classList.remove("info");
  element.classList.add(type);
}

function clearError() {
  const element = $("notice");
  if (!element) return;
  element.textContent = "";
  element.classList.add("hidden");
  element.classList.remove("error");
  element.classList.remove("info");
}

async function inspect() {
  const input = $("prUrl");
  if (!input) return;

  const url = input.value.trim();
  if (!url) {
    showError("Paste a GitHub pull-request URL.");
    return;
  }

  // Refresh everything immediately except the entered link
  resetWorkspaceUI(true);

  const button = $("inspect");
  button.disabled = true;
  button.textContent = "Inspecting…";
  currentStage = "stageInspect";

  const config = $("config");
  if (config) {
    config.textContent = "Connecting to GitHub & loading evidence…";
    config.style.color = "var(--muted)";
  }

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
    if (config) {
      config.textContent =
        `Evidence loaded · ${data.evidence_files || 0} files · ${Number(data.context_chars || 0).toLocaleString()} chars`;
      config.style.color = "";
    }
  } catch (error) {
    setStage("stageInspect", "fail", "ERROR");
    showError(error.message);
    $("workspace")?.classList.add("hidden");
    if (config && defaultHealthText) {
      config.textContent = defaultHealthText;
      config.style.color = defaultHealthColor;
    }
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

  // Reset stages 2-4 to clean initial state for recovery
  setStage("stageDiagnose", "active", "RUNNING");
  setStage("stageRepair", "", "WAITING");
  setStage("stageVerify", "", "WAITING");

  try {
    currentStage = "stageDiagnose";
    const diagnosis = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId })
    });
    setStage("stageDiagnose", "done", "COMPLETE");
    renderDiagnosis(diagnosis);

    const maxAttempts = Math.max(1, Math.min(3, maxRepairAttempts));
    let verified = false;
    let isSandboxUnavailable = false;

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
      isSandboxUnavailable = verification.status === "SANDBOX_UNAVAILABLE" || verification.verifier_type === "none";
      renderVerification(verification, attempt);

      if (verified) {
        setStage("stageVerify", "done", "VERIFIED");
        break;
      }

      if (isSandboxUnavailable) {
        setStage("stageVerify", "done", "PATCH READY");
        showNotice("Candidate patch generated! Sandbox testing was skipped because Docker Desktop is not running locally. You can review and download the patch below.", "info");
        break;
      }

      setStage("stageVerify", "fail", attempt < maxAttempts ? "RETRYING" : "REJECTED");
      if (attempt < maxAttempts) {
        showError("Attempt failed verification. HEALFORGE is feeding the sandbox failure back into the repair engine.");
      }
    }

    if (!verified && !isSandboxUnavailable) {
      showError("HEALFORGE could not produce a verified repair after the available attempts.");
    } else if (verified) {
      clearError();
    }
  } catch (error) {
    setStage(currentStage, "fail", "ERROR");
    let msg = error.message || "An unexpected error occurred.";
    if (msg.includes("AI_QUOTA_EXHAUSTED") || msg.includes("free-models-per-day") || msg.includes("daily quota")) {
      msg = "AI SERVICE LIMITED: OpenRouter's current free-model quota has been exhausted. No repair request was attempted further to avoid wasting quota.";
    } else if (msg.includes("No safe edit plan was produced") || msg.includes("missing valid 'edits'")) {
      const cleanDetails = msg.replace(/^.*No safe edit plan was produced after all repair model attempts:\s*/i, "");
      msg = cleanDetails ? `PATCH GENERATION FAILED: ${cleanDetails}` : "PATCH GENERATION FAILED: No valid semantic edit plan was produced by the configured repair models.";
    } else if (msg.includes("Rate limit exceeded") || msg.includes("429")) {
      msg = "AI SERVICE RATE LIMITED: The configured AI provider rate limit has been reached.";
    }
    // Defense-in-depth: strip any token/key patterns
    msg = msg.replace(/bearer\s+[A-Za-z0-9_\-\.]+/gi, "bearer [REDACTED]");
    msg = msg.replace(/(?:sk-|ghp_|github_pat_)[A-Za-z0-9_\-\.]+/gi, "[REDACTED]");
    msg = msg.replace(/key=[A-Za-z0-9_\-\.]+/gi, "key=[REDACTED]");
    showError(msg);
  } finally {
    button.disabled = false;
    button.textContent = "Run recovery";
  }
}

function renderDiagnosis(diagnosis) {
  const confidence = Number(diagnosis.confidence || 0);
  const evidence = Array.isArray(diagnosis.evidence) ? diagnosis.evidence : [diagnosis.evidence].filter(Boolean);
  const duration = diagnosis.metrics?.diagnosis_seconds ? ` · ${diagnosis.metrics.diagnosis_seconds}s` : "";

  $("diagnosisCard").classList.remove("hidden");
  $("diagnosisCard").innerHTML = `
    <div class="result-title">
      <h2>Root-cause diagnosis</h2>
      <span class="pill good">${Math.round(confidence * 100)}% confidence${duration}</span>
    </div>
    <div class="kv">
      <b>Summary</b><span>${escapeHtml(diagnosis.summary || "Diagnosis complete")}</span>
      <b>Root cause</b><span>${escapeHtml(diagnosis.root_cause || "Identified from repository evidence")}</span>
      <b>Repair strategy</b><span>${escapeHtml(diagnosis.repair_strategy || "Synthesizing minimal patch")}</span>
    </div>
    ${evidence.length ? `<div class="hint">Evidence</div><div class="content">${evidence.map((item) => `• ${escapeHtml(item)}`).join("<br>")}</div>` : ""}
  `;
}

function renderRepair(repair, attempt) {
  const confidence = Number(repair.confidence || 0);
  const duration = repair.metrics?.generate_seconds ? ` · ${repair.metrics.generate_seconds}s` : "";
  const hypothesis = repair.hypothesis ? `<b>Hypothesis</b><span>${escapeHtml(repair.hypothesis)}</span>` : "";
  const strategy = repair.strategy ? `<b>Strategy</b><span>${escapeHtml(repair.strategy)}</span>` : "";
  const whyFailed = repair.why_previous_failed ? `
    <div class="hint" style="color: var(--accent); margin-top: 10px;">
      <b>Prior attempt analysis:</b> ${escapeHtml(repair.why_previous_failed)}
    </div>` : "";

  const patchText = (repair.patch || "").trim();
  const downloadButton = patchText ? `
    <div class="actions">
      <a class="secondary" href="/api/session/${encodeURIComponent(sessionId)}/patch">Download patch</a>
    </div>` : "";

  $("patchCard").classList.remove("hidden");
  $("patchCard").innerHTML = `
    <div class="result-title">
      <h2>Candidate repair · attempt ${attempt}</h2>
      <span class="pill">${Math.round(confidence * 100)}% confidence${duration}</span>
    </div>
    <div class="kv">
      ${hypothesis}
      ${strategy}
      <b>Summary</b><span>${escapeHtml(repair.explanation || repair.summary || "Structured repair generated")}</span>
    </div>
    ${whyFailed}
    <pre class="code" style="margin-top: 12px;">${escapeHtml(patchText || "No safe patch generated.")}</pre>
    ${downloadButton}
  `;
}

function renderVerification(verification, attempt) {
  const verified = Boolean(verification.passed);
  const isSandboxUnavailable = verification.status === "SANDBOX_UNAVAILABLE" || verification.verifier_type === "none";
  const duration = verification.metrics?.verify_seconds ? ` · ${verification.metrics.verify_seconds}s` : "";
  const command = escapeHtml(verification.command || "Verification command unavailable");
  const output = escapeHtml(verification.output || verification.reason || "No verification output returned.");
  const exitCode = verification.exit_code == null ? "—" : escapeHtml(verification.exit_code);

  let pillClass = "bad";
  let pillText = "REJECTED";
  if (verified) {
    pillClass = "good";
    pillText = "VERIFIED";
  } else if (isSandboxUnavailable) {
    pillClass = "warn";
    pillText = "UNVERIFIED (NO DOCKER)";
  }

  let hintHtml = "";
  if (verified) {
    hintHtml = `<div class="actions"><a class="secondary" href="/api/session/${encodeURIComponent(sessionId)}/patch">Download verified patch</a></div>`;
  } else if (isSandboxUnavailable) {
    hintHtml = `
      <div class="hint">The AI successfully synthesized the candidate patch above. Local Docker Desktop is not running, so sandbox execution was skipped. The generated patch is complete and available to download below.</div>
      <div class="actions" style="margin-top: 10px;"><a class="secondary" href="/api/session/${encodeURIComponent(sessionId)}/patch">Download candidate patch</a></div>
    `;
  } else {
    hintHtml = `<div class="hint">The sandbox rejected this candidate. The next attempt receives this failure output and re-evaluates from clean repository state.</div>`;
  }

  $("verifyCard").classList.remove("hidden");
  $("verifyCard").innerHTML = `
    <div class="result-title">
      <h2>Sandbox verification · attempt ${attempt}</h2>
      <span class="pill ${pillClass}">${pillText}${duration}</span>
    </div>
    <div class="kv">
      <b>Test command</b><span>${command}</span>
      <b>Exit code</b><span>${exitCode}</span>
    </div>
    <pre class="code">${output}</pre>
    ${hintHtml}
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
  $("newRun")?.addEventListener("click", () => {
    resetWorkspaceUI(false);
  });
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
      const providerName = (health.ai_provider === "gemini") ? "Gemini" : "OpenRouter";
      if (!health.ai_configured) {
        defaultHealthText = `${providerName} key missing`;
        defaultHealthColor = "#ff6b6b";
      } else if (health.ai_status === "DAILY_QUOTA_EXHAUSTED") {
        defaultHealthText = `${providerName} quota exhausted`;
        defaultHealthColor = "#ffa94d";
      } else if (health.ai_status === "TEMPORARILY_UNAVAILABLE") {
        defaultHealthText = `${providerName} status: Temporarily unavailable`;
        defaultHealthColor = "#ffa94d";
      } else {
        const primaryModel = health.ai_models?.[0] || "default";
        let verifierLabel = "Docker (Local)";
        if (health.verifier_type === "remote") {
          verifierLabel = "Remote Sandbox";
        } else if (health.verifier_type === "none") {
          verifierLabel = "Sandbox Unavailable";
        }
        defaultHealthText = `AI: ${providerName} (${primaryModel}) · VERIFIER: ${verifierLabel}`;
        defaultHealthColor = "";
      }
      config.textContent = defaultHealthText;
      config.style.color = defaultHealthColor;
    })
    .catch((error) => showError(error.message));
});
