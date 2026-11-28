// Scam Call Guard - browser side.
// Builds a cumulative "Agent: ... / Customer: ..." transcript (same format as ICFD-31k) and sends it to the
// server every 3 seconds; the server returns NORMAL / REVIEW / ALERT from the fine-tuned ensemble.
const $ = (id) => document.getElementById(id);
const MESSAGES = {
  NORMAL: "No scam signals detected so far.",
  REVIEW: "Suspicious — stay alert. Do not share OTP, PIN, card or bank details.",
  ALERT: "Likely SCAM — hang up. Never share OTP, PIN, CVV or passwords. Call your bank on its official number.",
};

let session = null, lines = [], speaker = "Agent", t0 = null, history = [], busy = false;
let recog = null, recording = false, tick = null, replaying = false, lastSent = "", wordsWindow = [], confs = [];

// ---------------------------------------------------------------- server
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

async function pollHealth() {
  try {
    const h = await api("/api/health");
    if (h.error) { $("status").textContent = "Model error"; $("status").className = "pill err"; $("msg").textContent = h.error; return; }
    if (h.ready) {
      $("status").textContent = `Model ready · ${h.members.length}-model ensemble${h.intent_checker ? " + intent check" : ""} · ${h.device.toUpperCase()}`;
      $("status").className = "pill ready";
      loadSamples(h.samples_available);
      return;
    }
  } catch (e) { /* server still starting */ }
  setTimeout(pollHealth, 1500);
}

async function newSession() {
  const s = await api("/api/sessions", { policy: $("policySel").value });
  session = s; lines = []; history = []; t0 = performance.now(); lastSent = "";
  renderTranscript(); drawChart(); setVerdict(null);
  $("meterReview").style.left = `${(100 * s.tau_review / s.tau_alert).toFixed(1)}%`;
}

function transcriptText() {
  return lines.map((l) => `${l.speaker}: ${l.text}`).join("\n");
}

async function score(text, ts, audio) {
  if (!session || busy) return;
  busy = true;
  try {
    const st = await api(`/api/sessions/${session.session_id}/update`, { transcript: text, timestamp: ts, audio });
    history.push(st);
    setVerdict(st); drawChart();
  } catch (e) { $("msg").textContent = "Error: " + e.message; }
  busy = false;
}

// ---------------------------------------------------------------- display
function setVerdict(st) {
  const v = $("verdict");
  if (!st) { v.className = "verdict NORMAL"; $("level").textContent = "LISTENING"; $("msg").textContent = MESSAGES.NORMAL;
             $("meterFill").style.width = "0"; $("riskText").textContent = "–"; $("stats").textContent = ""; $("reasons").innerHTML = ""; return; }
  v.className = `verdict ${st.level}`;
  $("level").textContent = st.level === "ALERT" ? "⚠️ ALERT" : st.level === "REVIEW" ? "REVIEW" : "NORMAL";
  $("msg").textContent = MESSAGES[st.level] + (st.alerted && st.first_alert_ts != null
    ? ` (alert raised at ${st.first_alert_ts.toFixed(0)} s by ${{ rules: "the safety rules", intent: "the AI intent check", model: "the AI model" }[st.alert_source] || "the AI model"})` : "");
  const reasons = (st.rule_hits || []).map((h) =>
    `<li><span class="tag ${h.level}">${h.rule.startsWith("I") ? "AI INTENT" : "RULE"} · ${h.level}</span>${escapeHtml(h.reason)} — “${escapeHtml(h.evidence)}”${h.score ? ` (${Math.round(100 * h.score)}%)` : ""}</li>`);
  if (st.model_level && st.model_level !== "NORMAL")
    reasons.unshift(`<li><span class="tag MODEL">AI MODEL · ${st.model_level}</span>The conversation pattern resembles scam calls it was trained on.</li>`);
  $("reasons").innerHTML = reasons.join("");
  const pct = Math.min(100, 100 * st.risk_index);
  $("meterFill").style.width = `${pct}%`;
  $("meterFill").style.background = st.risk_index >= 1 ? "var(--bad)" : st.risk_index >= st.review_index ? "var(--warn)" : "var(--ok)";
  $("riskText").textContent = `${(100 * st.risk_index).toFixed(0)}% of alert threshold`;
  $("stats").innerHTML = `<span>t = ${st.timestamp.toFixed(0)} s</span><span>chunk P(scam) ${(st.score * 100).toFixed(2)}%</span>` +
    `<span>smoothed ${(st.smoothed_score * 100).toFixed(2)}%</span><span>model time ${st.latency_ms} ms</span>`;
}

function drawChart() {
  const svg = $("chart"), W = 600, H = 140, pad = 6;
  if (!history.length) { svg.innerHTML = ""; return; }
  const tMax = Math.max(30, ...history.map((h) => h.timestamp));
  const yMax = Math.max(1.4, ...history.map((h) => h.risk_index));
  const x = (t) => pad + (W - 2 * pad) * t / tMax, y = (v) => H - pad - (H - 2 * pad) * Math.min(v, yMax) / yMax;
  const rv = history[0].review_index;
  const pts = history.map((h) => `${x(h.timestamp).toFixed(1)},${y(h.risk_index).toFixed(1)}`).join(" ");
  const alertAt = history.find((h) => h.alerted);
  svg.innerHTML =
    `<line x1="0" x2="${W}" y1="${y(1)}" y2="${y(1)}" stroke="var(--bad)" stroke-dasharray="5 4" />` +
    `<line x1="0" x2="${W}" y1="${y(rv)}" y2="${y(rv)}" stroke="var(--warn)" stroke-dasharray="5 4" />` +
    `<text x="${W - 4}" y="${y(1) - 4}" text-anchor="end" font-size="11" fill="var(--bad)">alert</text>` +
    `<text x="${W - 4}" y="${y(rv) - 4}" text-anchor="end" font-size="11" fill="var(--warn)">review</text>` +
    (alertAt ? `<line x1="${x(alertAt.first_alert_ts)}" x2="${x(alertAt.first_alert_ts)}" y1="0" y2="${H}" stroke="var(--bad)" />` : "") +
    `<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2.5" />`;
}

function renderTranscript(interim) {
  const box = $("transcript");
  if (!lines.length && !interim) { box.innerHTML = '<p class="empty">Nothing yet.</p>'; return; }
  box.innerHTML = lines.map((l) => `<p class="${l.speaker}"><b>${l.speaker === "Agent" ? "Caller" : "Me"}:</b> ${escapeHtml(l.text)}</p>`).join("");
  box.scrollTop = box.scrollHeight;
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function addLine(spk, text) {
  text = text.trim();
  if (!text) return;
  const last = lines[lines.length - 1];
  if (last && last.speaker === spk) last.text += " " + text; else lines.push({ speaker: spk, text });
  renderTranscript();
}

// ---------------------------------------------------------------- live mic (browser Web Speech API, online)
function audioFeatures() {
  // Approximate the dataset's audio analysis: speaking pace from words/second, recognizer confidence.
  const now = performance.now();
  wordsWindow = wordsWindow.filter((w) => now - w.t < 6000);
  const wps = wordsWindow.reduce((a, w) => a + w.n, 0) / 6;
  const pace = wps < 1.5 ? "Slow" : wps < 3 ? "Medium" : "Fast";
  const conf = confs.length ? confs.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, confs.length) : undefined;
  return conf !== undefined ? { pace, confidence_score: Number(conf.toFixed(3)) } : { pace };
}

async function startCall() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert("This browser has no speech recognition. Use Chrome or Edge, or the 'Type lines' tab."); return; }
  await newSession();
  recog = new SR();
  recog.lang = "en-IN"; recog.continuous = true; recog.interimResults = true;
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) {
        addLine(speaker, r[0].transcript);
        wordsWindow.push({ t: performance.now(), n: r[0].transcript.trim().split(/\s+/).length });
        if (r[0].confidence) confs.push(r[0].confidence);
      } else interim += r[0].transcript;
    }
    $("interim").textContent = interim ? `${speaker === "Agent" ? "Caller" : "Me"}: ${interim}…` : "";
  };
  recog.onerror = (e) => { if (e.error === "not-allowed") { alert("Microphone permission denied."); stopCall(); } };
  recog.onend = () => { if (recording) try { recog.start(); } catch (_) {} };   // Chrome stops after silence
  recording = true; recog.start();
  $("micBtn").textContent = "End call"; $("micBtn").classList.add("recording");
  tick = setInterval(() => {                                                      // one model update per 3 s
    const text = transcriptText();
    if (text) score(text, (performance.now() - t0) / 1000, audioFeatures());
  }, 3000);
}

function stopCall() {
  recording = false; clearInterval(tick);
  try { recog && recog.stop(); } catch (_) {}
  $("micBtn").textContent = "Start call"; $("micBtn").classList.remove("recording"); $("interim").textContent = "";
}

// ---------------------------------------------------------------- typed lines
async function addTyped(spk) {
  if (!session) await newSession();
  addLine(spk, $("lineInput").value); $("lineInput").value = "";
  const ts = 3 * (history.length + 1);                                            // each line = next 3-second chunk
  await score(transcriptText(), ts, {});
}

// ---------------------------------------------------------------- replay held-out calls
async function loadSamples(available) {
  const sel = $("sampleSel");
  if (!available) { sel.innerHTML = "<option>No local dataset (data/icfd-31k) found</option>"; $("replayBtn").disabled = true; return; }
  const list = await api("/api/samples");
  sel.innerHTML = list.map((s, i) => `<option value="${s.id}">Call ${i + 1} · ${s.domain.replace(/_/g, " ")} · ${Math.round(s.duration_s)} s</option>`).join("");
}

async function replay() {
  if (replaying) { replaying = false; return; }
  const s = await api(`/api/samples/${$("sampleSel").value}`);
  await newSession(); replaying = true; $("truth").classList.add("hidden");
  $("replayBtn").textContent = "Stop";
  for (const c of s.chunks) {
    if (!replaying) break;
    lines = c.transcript.split("\n").filter(Boolean).map((l) => {
      const m = l.match(/^(Agent|Customer):\s*(.*)$/);
      return m ? { speaker: m[1], text: m[2] } : { speaker: "Agent", text: l };
    });
    renderTranscript();
    await score(c.transcript, c.timestamp, c.audio);
    await new Promise((r) => setTimeout(r, 600));                               // 3 s of call -> 0.6 s
  }
  replaying = false; $("replayBtn").textContent = "Replay";
  const last = history[history.length - 1];
  const caught = history.some((h) => h.alerted);
  const truth = s.is_scam_scenario
    ? `Ground truth: <b>scam scenario</b> (${s.case_type}). The scam request starts at ~${s.label_flip_s ?? "?"} s. ` +
      (caught ? `Model ALERT at ${last.first_alert_ts.toFixed(0)} s ✔` : "Model did not raise an ALERT ✘")
    : `Ground truth: <b>normal call</b> (${s.case_type}). ` + (caught ? `Model raised a false ALERT at ${last.first_alert_ts.toFixed(0)} s ✘` : "No ALERT raised ✔");
  $("truth").innerHTML = `${truth}<br><small>${s.source}</small>`; $("truth").classList.remove("hidden");
}

// ---------------------------------------------------------------- wiring
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${b.dataset.tab}`));
});
document.querySelectorAll(".spk").forEach((b) => b.onclick = () => {
  speaker = b.dataset.spk; document.querySelectorAll(".spk").forEach((x) => x.classList.toggle("active", x === b));
});
$("micBtn").onclick = () => (recording ? stopCall() : startCall());
$("addCaller").onclick = () => addTyped("Agent");
$("addMe").onclick = () => addTyped("Customer");
$("resetBtn").onclick = () => newSession();
$("replayBtn").onclick = replay;
$("policySel").onchange = () => { if (!recording && !replaying) newSession(); };
pollHealth();
