/* Site logic. All numbers are precomputed (docs/data/*.js); nothing is trained in the browser. */
(function () {
  const C = window.Charts, M = window.SITE_METRICS, G = window.SITE_GRID, S = window.SITE_SAMPLE, ST = window.SITE_STEPPER;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const f2 = (v) => C.fmt(v, 2), f3 = (v) => C.fmt(v, 3), pct = (v, d) => (v == null ? "–" : (100 * v).toFixed(d == null ? 1 : d) + "%");
  const MODEL = () => C.css("--model"), BASE = () => C.css("--baseline");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function human(sec) {
    if (sec < 90) return Math.round(sec) + " s";
    if (sec < 5400) return Math.round(sec / 60) + " min";
    if (sec < 172800) return (sec / 3600).toFixed(sec < 36000 ? 1 : 0) + " h";
    return (sec / 86400).toFixed(1) + " days";
  }
  const hs = (lg) => human(Math.pow(10, lg));
  const ci = (pair, d) => `[${C.fmt(pair[0], d == null ? 3 : d)}, ${C.fmt(pair[1], d == null ? 3 : d)}]`;
  function fill(id, txt) { $$(`[data-fill="${id}"]`).forEach((e) => (e.textContent = txt)); }

  /* ---------------- numbers written into the prose ---------------- */
  function fillNumbers() {
    const r = M.runtime.overall, f = M.failure.overall;
    fill("rt-mae-m", f3(r.model.mae)); fill("rt-mae-b", f3(r.baseline.mae));
    fill("rt-fac-m", C.fmt(r.model.error_factor, 2) + "×"); fill("rt-fac-b", C.fmt(r.baseline.error_factor, 2) + "×");
    fill("rt-diff", f3(r.ci ? r.model.mae - r.baseline.mae : null)); fill("rt-diff-ci", ci(r.ci.mae_diff));
    fill("rt-rmse-m", f3(r.model.rmse)); fill("rt-rmse-b", f3(r.baseline.rmse));
    fill("rt-cov", pct(r.model.coverage_90)); fill("rt-cov-raw", pct(M.interval_raw.coverage_90));
    fill("rt-mse", f3(r.model.mean_signed_error)); fill("rt-ridge", f3(M.runtime.ridge.mae)); fill("rt-global", f3(M.runtime.global.mae));
    fill("rt-n", r.model.n.toLocaleString());
    fill("fl-pr-m", f3(f.model.pr_auc)); fill("fl-pr-b", f3(f.baseline.pr_auc)); fill("fl-base", f3(f.model.base_rate));
    fill("fl-diff", f3(f.model.pr_auc - f.baseline.pr_auc)); fill("fl-diff-ci", ci(f.ci.prauc_diff));
    fill("fl-brier-m", f3(f.model.brier)); fill("fl-brier-b", f3(f.baseline.brier)); fill("fl-brier-ci", ci(f.ci.brier_diff));
    fill("fl-ece-m", f3(f.model.ece)); fill("fl-ece-b", f3(f.baseline.ece)); fill("fl-logit", f3(M.failure.logistic.pr_auc));
    fill("fl-n", f.model.n.toLocaleString()); fill("thr", f2(M.threshold));
    fill("fl-prec", pct(f.model.precision)); fill("fl-rec", pct(f.model.recall));
    fill("test-ran", M.test_run.ran_at_utc.replace("T", " ").replace("+00:00", " UTC"));
    const sg = M.sensitivity_assigned_gpu;
    fill("leak-rt", f3(sg.runtime.mae)); fill("leak-rt0", f3(sg.runtime_tree_without.mae));
    fill("leak-fl", f3(sg.failure.pr_auc)); fill("leak-fl0", f3(sg.failure_without.pr_auc));
  }

  /* ---------------- headline comparison ---------------- */
  function headline() {
    const r = M.runtime.overall, f = M.failure.overall;
    const rows = [
      { id: "hl-rt", domain: [0, Math.max(0.8, r.baseline.mae * 1.25)], label: "Runtime MAE",
        pts: [{ v: r.model.mae, color: MODEL() }, { v: r.baseline.mae, color: BASE() }] },
      { id: "hl-fl", domain: [0, 1], label: "Failure PR-AUC",
        pts: [{ v: f.model.pr_auc, color: MODEL() }, { v: f.baseline.pr_auc, color: BASE() }] },
      { id: "hl-br", domain: [0, Math.max(0.25, f.baseline.brier * 1.3)], label: "Failure Brier score",
        pts: [{ v: f.model.brier, color: MODEL() }, { v: f.baseline.brier, color: BASE() }] }];
    rows.forEach((row) => C.dotRow($("#" + row.id), { domain: row.domain, points: row.pts, label: row.label + ": model vs baseline", tickFmt: (t) => t.toFixed(2) }));
  }

  /* ---------------- 1. Try a job ---------------- */
  function tryJob() {
    const root = $("#try"); if (!root) return;
    const ax = G.axes, order = ["role", "gpu", "gtype", "size", "hour", "day", "user"];
    const state = Object.assign({}, G.preset);
    const labels = { role: "Task type", gpu: "GPUs requested", gtype: "Requested GPU type", size: "CPU / memory per instance",
                     hour: "Submit hour (UTC+8)", day: "Day", user: "User" };
    const ctl = $(".controls", root);
    order.forEach((k) => {
      const w = document.createElement("label"); w.className = "ctl"; w.textContent = labels[k];
      if (k === "day" || k === "user") {
        const seg = document.createElement("div"); seg.className = "seg"; seg.setAttribute("role", "group"); seg.setAttribute("aria-label", labels[k]);
        ax[k].forEach((v) => { const b = document.createElement("button"); b.type = "button"; b.textContent = v; b.dataset.k = k; b.dataset.v = v;
          b.addEventListener("click", () => { state[k] = v; render(); }); seg.appendChild(b); });
        w.appendChild(seg);
      } else {
        const s = document.createElement("select"); s.dataset.k = k;
        ax[k].forEach((v) => { const o = document.createElement("option"); o.value = v;
          o.textContent = k === "size" ? `${v} (${G.sizes[v].cpu_cores} cores, ${G.sizes[v].mem_gb} GB)` : k === "hour" ? `${String(v).padStart(2, "0")}:00` : v; s.appendChild(o); });
        s.addEventListener("change", () => { state[k] = k === "gpu" || k === "hour" ? +s.value : s.value; render(); });
        w.appendChild(s);
      }
      ctl.appendChild(w);
    });
    $("#try-preset").addEventListener("click", () => { Object.assign(state, G.preset); render(); });
    function index() {
      let i = 0;
      order.forEach((k) => { i = i * ax[k].length + ax[k].map(String).indexOf(String(state[k])); });
      return i;
    }
    function render() {
      $$("select", root).forEach((s) => (s.value = String(state[s.dataset.k])));
      $$(".seg button", root).forEach((b) => b.setAttribute("aria-pressed", String(state[b.dataset.k]) === b.dataset.v ? "true" : "false"));
      const i = index();
      const lo = G.lo[i], hi = G.hi[i], p = G.pred[i], br = G.base_rt[i], pf = G.pf[i], bf = G.base_f[i];
      $("#try-rt").textContent = hs(p); $("#try-rng").textContent = `${hs(lo)} to ${hs(hi)}`;
      $("#try-rt-b").textContent = hs(br); $("#try-pf").textContent = pct(pf, 0); $("#try-pf-b").textContent = pct(bf, 0);
      $("#try-width").textContent = `range spans a factor of ${C.fmt(Math.pow(10, hi - lo), 0)}`;
      $("#try-flag").textContent = pf >= M.threshold ? `above the alert threshold (${f2(M.threshold)})` : `below the alert threshold (${f2(M.threshold)})`;
      // range bar on a log axis 1 s .. 30 days
      const svg = $("#try-bar"); const d0 = 0, d1 = Math.log10(30 * 86400);
      const W = Math.round(svg.getBoundingClientRect().width) || 640; svg.setAttribute("viewBox", `0 0 ${W} 56`);
      const x = (v) => 10 + ((Math.max(d0, Math.min(d1, v)) - d0) / (d1 - d0)) * (W - 20);
      svg.innerHTML = "";
      C.el("line", { x1: 10, x2: W - 10, y1: 30, y2: 30, stroke: C.css("--rule") }, svg);
      [[0, "1 s"], [1.78, "1 min"], [3.56, "1 h"], [4.94, "1 day"], [6.41, "30 days"]].forEach(([v, t]) => {
        C.el("line", { x1: x(v), x2: x(v), y1: 26, y2: 34, stroke: C.css("--rule") }, svg);
        C.el("text", { x: x(v), y: 50, "text-anchor": v > 6 ? "end" : v < 0.1 ? "start" : "middle", style: "font: 11px var(--mono); fill: var(--ink-3)" }, svg).textContent = t;
      });
      C.el("rect", { x: x(lo), y: 22, width: Math.max(2, x(hi) - x(lo)), height: 16, rx: 3, fill: C.css("--model-soft"), stroke: MODEL() }, svg);
      C.el("line", { x1: x(br), x2: x(br), y1: 14, y2: 46, stroke: BASE(), "stroke-width": 3 }, svg);
      C.el("circle", { cx: x(p), cy: 30, r: 6, fill: MODEL(), stroke: C.css("--surface"), "stroke-width": 2 }, svg);
    }
    render();
  }

  /* generic stepper wiring: Back / Next / Play / Reset + arrow keys on the panel */
  function stepper(root, n, show) {
    let k = 0, timer = null;
    const back = $(".s-back", root), next = $(".s-next", root), play = $(".s-play", root), reset = $(".s-reset", root), status = $(".status", root);
    function go(j) { k = Math.max(0, Math.min(n - 1, j)); show(k); status.textContent = `step ${k + 1} of ${n}`; back.disabled = k === 0; next.disabled = k === n - 1; }
    function stop() { clearInterval(timer); timer = null; play.textContent = "Play"; }
    back.addEventListener("click", () => { stop(); go(k - 1); });
    next.addEventListener("click", () => { stop(); go(k + 1); });
    reset.addEventListener("click", () => { stop(); go(0); });
    play.addEventListener("click", () => {
      if (timer) return stop();
      if (k === n - 1) go(0);
      play.textContent = "Pause";
      timer = setInterval(() => { if (k >= n - 1) return stop(); go(k + 1); }, reduced ? 2600 : 1800);
    });
    root.addEventListener("keydown", (e) => {
      if (e.target.matches("select, input")) return;
      if (e.key === "ArrowRight") { e.preventDefault(); stop(); go(k + 1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); stop(); go(k - 1); }
    });
    go(0);
    return { go };
  }

  /* ---------------- 2. How one prediction is made ---------------- */
  function walk() {
    const root = $("#walk"); if (!root) return;
    const q = ST.request, h = ST.history, p = ST.prediction;
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    $("#w-req").innerHTML = `<dt>task roles</dt><dd>${q.role_set}</dd><dt>main role</dt><dd>${q.main_role}</dd><dt>tasks / instances</dt><dd>${q.n_tasks} / ${q.inst_total}</dd>
      <dt>GPUs requested</dt><dd>${q.gpu_total}</dd><dt>CPU cores, memory</dt><dd>${C.fmt(q.cpu_total, 1)} cores, ${C.fmt(q.mem_total, 1)} GB</dd>
      <dt>requested GPU type</dt><dd>${q.gpu_type_spec}</dd><dt>submitted</dt><dd>${days[q.submit_weekday]} ${String(q.submit_hour).padStart(2, "0")}:00 (UTC+8), trace day ${q.day}</dd>`;
    $("#w-leak").innerHTML = `<dt>requested resources, roles, requested GPU type, hour, weekday</dt><dd><span class="pill yes">allowed</span></dd>
      <dt>assigned GPU type (${q.gpu_type_assigned || "none"})</dt><dd><span class="pill no">removed</span> chosen by the scheduler after submission</dd>
      <dt>end time, final status</dt><dd><span class="pill no">removed</span> only known after the job ends</dd>`;
    $("#w-hist").innerHTML = `<dt>this user's earlier submissions</dt><dd>${h.earlier_submissions.toLocaleString()}</dd>
      <dt>finished before this submission (used)</dt><dd>${h.finished_before_submit.toLocaleString()}</dd>
      <dt>still running at submission (not used)</dt><dd>${h.still_running_at_submit.toLocaleString()}</dd>
      <dt>user's past median runtime</dt><dd>${hs(h.user_median_logrt)}</dd><dt>user's past failure rate</dt><dd>${pct(h.user_fail_rate)}</dd>
      <dt>same script group: finished runs</dt><dd>${h.group_n_finished == null ? "–" : h.group_n_finished}</dd>
      <dt>same script group: median runtime</dt><dd>${h.group_median_logrt == null ? "–" : hs(h.group_median_logrt)}</dd>`;
    $("#w-pred").innerHTML = `<dt>predicted runtime</dt><dd>${hs(p.point)} (log10 s = ${f2(p.point)})</dd><dt>failure risk</dt><dd>${pct(p.p_fail, 0)}</dd>`;
    $("#w-range").innerHTML = `<dt>5–95% range</dt><dd>${hs(p.lo)} to ${hs(p.hi)}</dd>`;
    const em = Math.abs(p.point - p.truth), eb = Math.abs(p.baseline - p.truth);
    $("#w-truth").innerHTML = `<dt>actual runtime</dt><dd>${hs(p.truth)}</dd><dt>model error</dt><dd>${f2(em)} log10 (×${C.fmt(Math.pow(10, em), 2)})</dd>
      <dt>baseline (user's past median)</dt><dd>${hs(p.baseline)}, error ${f2(eb)} log10 (×${C.fmt(Math.pow(10, eb), 2)})</dd>
      <dt>inside the range?</dt><dd>${p.truth >= p.lo && p.truth <= p.hi ? "yes" : "no"}</dd>`;
    const items = $$(".steps li", root);
    stepper(root, items.length, (k) => items.forEach((li, i) => li.classList.toggle("on", i === k)));
  }

  /* ---------------- 3. Why we split by time ---------------- */
  function splitDemo() {
    const root = $("#split"); if (!root) return;
    const sp = M.split, end = sp.trace_end_day;
    const steps = [
      { text: "The trace: 69 days of submissions. Days 0–5 are a near-empty warm-up.", shade: [] },
      { text: `Development data: the earliest 70% of jobs (to day ${f2(sp.cut_day)}).`, shade: [[0, sp.cut_day, "dev", "development"]] },
      { text: "A 24-hour embargo: jobs submitted here are used by no model and no metric.", shade: [[0, sp.cut_day, "dev", "development"], [sp.cut_day, sp.test_from_day, "gap", "gap"]] },
      { text: `Test: everything after day ${f2(sp.test_from_day)}, scored once after all choices were frozen.`, shade: [[0, sp.cut_day, "dev", "development"], [sp.cut_day, sp.test_from_day, "gap", "gap"], [sp.test_from_day, end, "test", "test"]] }];
    sp.folds_day.forEach((f) => steps.push({ text: `Walk-forward fold ${f.fold}: fit on days before ${f2(f.fit_to)}, validate on days ${f2(f.val_from)}–${f2(f.val_to)}. All tuning used these folds.`,
      shade: [[0, f.fit_to, "dev", "fit"], [f.fit_to, f.val_from, "gap", "gap"], [f.val_from, f.val_to, "val", "validate"], [sp.test_from_day, end, "test-dim", "test (unused)"]] }));
    const fills = { dev: [C.css("--baseline-soft"), 1], gap: [C.css("--rule"), 1], test: [C.css("--model-soft"), 1], val: [C.css("--model-soft"), 1], "test-dim": [C.css("--rule-soft"), 1] };
    const D = M.daily;
    stepper(root, steps.length, (k) => {
      const st = steps[k];
      $("#split-text").textContent = st.text;
      C.lines($("#split-chart"), { x: [0, 70], y: [0, Math.max.apply(null, D.n) / 1000 * 1.05], xLabel: "Day of trace (submit time)", yLabel: "Jobs per day (thousands)",
        label: "Jobs per day with the split regions shaded", height: 240,
        shade: st.shade.map((s) => [s[0], s[1], fills[s[2]][0], fills[s[2]][1], s[3]]),
        series: [{ name: "jobs (k)", color: C.css("--ink-2"), x: D.day, y: D.n.map((v) => v / 1000) }], tipTitle: (d) => "day " + d, tipDigits: 1 });
    });
    const sd = M.split_demo;
    function showCompare(mode) {
      $$("#split-toggle button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === mode ? "true" : "false"));
      const k = mode === "time" ? "time_split" : "random_split";
      const r = sd.y_logrt[k], f = sd.y_fail[k];
      $("#sd-rt").textContent = f3(r.model.mae); $("#sd-rt-b").textContent = f3(r.baseline.mae);
      $("#sd-fl").textContent = f3(f.model.pr_auc); $("#sd-fl-b").textContent = f3(f.baseline.pr_auc);
      $("#sd-note").textContent = mode === "time"
        ? "Time split: validation week comes after all fitting data, as in real use."
        : "Random split: validation jobs are scattered among fitting jobs from the same days and bursts, so the score looks better than it would be in use.";
    }
    $$("#split-toggle button").forEach((b) => b.addEventListener("click", () => showCompare(b.dataset.mode)));
    showCompare("time");
  }

  /* ---------------- 4. What each feature group adds ---------------- */
  function ablation() {
    const root = $("#abl"); if (!root) return;
    const A = M.ablation, rb = M.runtime.overall.baseline.mae, fb = M.failure.overall.baseline.pr_auc;
    const n = A.runtime.length + 1;
    stepper(root, n, (k) => {
      const shown = A.runtime.slice(0, k);
      const rItems = [{ label: "baseline", v: rb, color: BASE() }].concat(shown.map((a) => ({ label: a.rung, v: a.model.mae, color: MODEL(),
        lo: rb + a.ci.mae_diff[0], hi: rb + a.ci.mae_diff[1],
        tip: `${a.rung}: MAE ${f3(a.model.mae)}<br>vs baseline ${f3(a.model.mae - rb)} ${ci(a.ci.mae_diff)}` })));
      const fItems = [{ label: "baseline", v: fb, color: BASE() }].concat(A.failure.slice(0, k).map((a) => ({ label: a.rung, v: a.model.pr_auc, color: MODEL(),
        lo: fb + a.ci.prauc_diff[0], hi: fb + a.ci.prauc_diff[1],
        tip: `${a.rung}: PR-AUC ${f3(a.model.pr_auc)}<br>vs baseline ${f3(a.model.pr_auc - fb)} ${ci(a.ci.prauc_diff)}` })));
      while (rItems.length < n) rItems.push({ label: A.runtime[rItems.length - 1].rung, v: 0, color: "transparent" });
      while (fItems.length < n) fItems.push({ label: A.failure[fItems.length - 1].rung, v: 0, color: "transparent" });
      C.bars($("#abl-rt"), { items: rItems, y: [0, Math.max(rb, A.runtime[0].model.mae) * 1.15], yLabel: "Runtime MAE (log10 s), lower is better", height: 300, width: 480, label: "Runtime ablation" });
      C.bars($("#abl-fl"), { items: fItems, y: [0, 1], yLabel: "Failure PR-AUC, higher is better", height: 300, width: 480, label: "Failure ablation" });
      const last = k ? `${A.runtime[k - 1].rung}: runtime MAE ${f3(A.runtime[k - 1].model.mae)}, failure PR-AUC ${f3(A.failure[k - 1].model.pr_auc)}.` : "Start: the per-user baselines alone.";
      $("#abl-text").textContent = last;
    });
    const tb = $("#abl-table tbody");
    A.runtime.forEach((a, i) => {
      const fa = A.failure[i];
      tb.insertAdjacentHTML("beforeend", `<tr><td>${a.rung}</td><td class="n">${f3(a.model.mae)}</td><td class="n">${ci(a.ci.mae_diff)}</td><td class="n">${f3(fa.model.pr_auc)}</td><td class="n">${ci(fa.ci.prauc_diff)}</td></tr>`);
    });
  }

  /* ---------------- 5. Where it wins and loses ---------------- */
  function explorer() {
    const root = $("#exp"); if (!root) return;
    const c = S.cols, n = S.n, filt = { band: "all", beat: "all", user: "all", gtype: "all" };
    const gtypes = Array.from(new Set(c.gtype)).sort();
    const gsel = $("#exp-gtype"); gtypes.forEach((g) => gsel.insertAdjacentHTML("beforeend", `<option value="${g}">${g}</option>`));
    $$("select", root).forEach((s) => s.addEventListener("change", () => { filt[s.dataset.f] = s.value; draw(); }));
    let sel = null;
    function keep(i) {
      if (filt.band !== "all" && String(c.band[i]) !== filt.band) return false;
      const win = Math.abs(c.pred[i] - c.y[i]) < Math.abs(c.base[i] - c.y[i]);
      if (filt.beat === "yes" && !win) return false;
      if (filt.beat === "no" && win) return false;
      if (filt.user === "seen" && !c.seen[i]) return false;
      if (filt.user === "unseen" && c.seen[i]) return false;
      if (filt.user === "nohist" && c.hist[i]) return false;
      if (filt.gtype !== "all" && c.gtype[i] !== filt.gtype) return false;
      return true;
    }
    function draw() {
      const idx = []; for (let i = 0; i < n; i++) if (keep(i)) idx.push(i);
      const pts = idx.map((i) => {
        const win = Math.abs(c.pred[i] - c.y[i]) < Math.abs(c.base[i] - c.y[i]);
        return { x: c.y[i], y: c.pred[i], i, r: sel === i ? 6 : 3, color: win ? MODEL() : BASE(), alpha: sel === i ? 1 : 0.5,
                 stroke: sel === i ? C.css("--ink") : null, tip: `job ${c.id[i]}<br>actual ${hs(c.y[i])}<br>model ${hs(c.pred[i])}<br>baseline ${hs(c.base[i])}` };
      });
      C.scatter($("#exp-chart"), { x: [0, 6.5], y: [0, 6.5], diagonal: true, xLabel: "Actual runtime (log10 s)", yLabel: "Predicted (log10 s)", height: 380, width: 560,
        points: pts, label: "Predicted vs actual runtime for sampled test jobs", onClick: (p) => { sel = p.i; $("#exp-id").value = c.id[p.i]; detail(p.i); draw(); } });
      let mm = 0, mb = 0, w = 0;
      idx.forEach((i) => { mm += Math.abs(c.pred[i] - c.y[i]); mb += Math.abs(c.base[i] - c.y[i]); w += Math.abs(c.pred[i] - c.y[i]) < Math.abs(c.base[i] - c.y[i]) ? 1 : 0; });
      $("#exp-status").textContent = idx.length
        ? `${idx.length.toLocaleString()} jobs shown · model MAE ${f3(mm / idx.length)} · baseline MAE ${f3(mb / idx.length)} · model closer on ${pct(w / idx.length, 0)}`
        : "No jobs match these filters. Widen one of them.";
    }
    $("#exp-show").addEventListener("click", () => { const i = Math.max(1, Math.min(n, +$("#exp-id").value || 1)) - 1; sel = i; detail(i); draw(); });
    $("#exp-worst").addEventListener("click", () => {
      let best = -1, bi = -1; for (let i = 0; i < n; i++) if (keep(i)) { const e = Math.abs(c.pred[i] - c.y[i]); if (e > best) { best = e; bi = i; } }
      if (bi >= 0) { sel = bi; $("#exp-id").value = c.id[bi]; detail(bi); draw(); }
    });
    function detail(i) {
      const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
      $("#exp-detail").innerHTML = `<h4>Job ${c.id[i]} (test band ${c.band[i]}, day ${c.day[i]})</h4><dl class="kv">
        <dt>roles</dt><dd>${c.role[i]} (${c.ntask[i]} task${c.ntask[i] > 1 ? "s" : ""})</dd><dt>GPUs / type</dt><dd>${c.gpu[i]} / ${c.gtype[i]}</dd>
        <dt>CPU, memory</dt><dd>${c.cpu[i] == null ? "–" : c.cpu[i]} cores, ${c.mem[i] == null ? "–" : c.mem[i]} GB</dd>
        <dt>submitted</dt><dd>${days[c.wday[i]]} ${String(c.hour[i]).padStart(2, "0")}:00</dd>
        <dt>user</dt><dd>${c.seen[i] ? "seen in development" : "new to the test period"}${c.hist[i] ? "" : ", no finished history"}</dd>
        <dt>actual</dt><dd>${hs(c.y[i])}</dd><dt>model</dt><dd>${hs(c.pred[i])} (range ${hs(c.lo[i])}–${hs(c.hi[i])})</dd>
        <dt>baseline</dt><dd>${hs(c.base[i])}</dd><dt>failure risk</dt><dd>model ${pct(c.pf[i], 0)} · baseline ${pct(c.pfb[i], 0)}</dd></dl>`;
    }
    draw();
  }

  /* ---------------- 6. Does it hold up over time ---------------- */
  function overTime() {
    const root = $("#time"); if (!root) return;
    let metric = "mae";
    $$("#time-toggle button").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.m; draw(); }));
    function draw() {
      $$("#time-toggle button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === metric ? "true" : "false"));
      const rd = M.drift.runtime_by_day, fd = M.drift.failure_by_day.filter((d) => d.pr_auc_model != null);
      const edges = M.split.band_edges_day, shade = [];
      for (let b = 0; b < 4; b++) if (b % 2 === 1) shade.push([edges[b], edges[b + 1], C.css("--rule-soft"), 1, "band " + (b + 1)]);
      let o;
      if (metric === "mae") o = { y: [0, 0.9], yLabel: "Runtime MAE (log10 s)", series: [
        { name: "model", color: MODEL(), x: rd.map((d) => d.day), y: rd.map((d) => d.mae_model), label: "model" },
        { name: "baseline", color: BASE(), x: rd.map((d) => d.day), y: rd.map((d) => d.mae_baseline), label: "baseline" }] };
      if (metric === "mse") o = { y: [-0.6, 0.6], yTicks: [-0.6, -0.3, 0, 0.3, 0.6], yLabel: "Mean signed error (log10 s), + = too long", hlines: [{ y: 0, label: "no bias" }], series: [
        { name: "model", color: MODEL(), x: rd.map((d) => d.day), y: rd.map((d) => d.mse_model_signed), label: "model" },
        { name: "baseline", color: BASE(), x: rd.map((d) => d.day), y: rd.map((d) => d.mse_baseline_signed), label: "baseline" }] };
      if (metric === "pr") o = { y: [0, 1], yLabel: "Failure PR-AUC", series: [
        { name: "model", color: MODEL(), x: fd.map((d) => d.day), y: fd.map((d) => d.pr_auc_model), label: "model" },
        { name: "baseline", color: BASE(), x: fd.map((d) => d.day), y: fd.map((d) => d.pr_auc_baseline), label: "baseline" },
        { name: "base rate", color: C.css("--ink-3"), dash: "4 3", x: fd.map((d) => d.day), y: fd.map((d) => d.base_rate), label: "base rate" }] };
      C.lines($("#time-chart"), Object.assign({ x: [56, 69.5], xLabel: "Day of trace (test period)", height: 270, shade, label: "Error per test day, model vs baseline",
        tipTitle: (d) => "day " + d, tipDigits: 3, margin: { r: 64 } }, o));
      const tb = $("#time-table tbody"); tb.innerHTML = "";
      for (let b = 1; b <= 4; b++) {
        const r = M.runtime.slices["band_" + b], f = M.failure.slices["band_" + b];
        tb.insertAdjacentHTML("beforeend", `<tr><td>band ${b} (days ${f2(edges[b - 1])}–${f2(edges[b])})</td>
          <td class="n">${f3(r.model.mae)}</td><td class="n">${f3(r.baseline.mae)}</td><td class="n">${ci(r.ci.mae_diff)}</td>
          <td class="n">${f3(r.model.mean_signed_error)}</td><td class="n">${pct(r.model.coverage_90)}</td>
          <td class="n">${f3(f.model.pr_auc)}</td><td class="n">${f3(f.baseline.pr_auc)}</td><td class="n">${f3(f.model.base_rate)}</td></tr>`);
      }
    }
    draw();
  }

  /* ---------------- 7. Failure threshold ---------------- */
  function threshold() {
    const root = $("#thr"); if (!root) return;
    const T = M.failure.threshold_curve, slider = $("#thr-slider");
    slider.min = 0; slider.max = T.length - 1; slider.step = 1;
    let k = T.findIndex((t) => Math.abs(t.threshold - M.threshold) < 0.006); if (k < 0) k = 12;
    slider.value = k;
    function draw() {
      const t = T[+slider.value];
      $("#thr-val").textContent = f2(t.threshold);
      $("#thr-prec").textContent = pct(t.precision); $("#thr-rec").textContent = pct(t.recall); $("#thr-flag").textContent = pct(t.flag_rate);
      $("#thr-gpu").textContent = pct(t.gpus_flagged_share); $("#thr-gcaught").textContent = pct(t.failed_gpus_caught_share);
      $("#thr-bprec").textContent = pct(t.baseline.precision); $("#thr-brec").textContent = pct(t.baseline.recall);
      $("#thr-cost").textContent = f3(t.cost_per_job); $("#thr-bcost").textContent = f3(t.baseline.cost_per_job);
      slider.setAttribute("aria-valuetext", `threshold ${f2(t.threshold)}, precision ${pct(t.precision)}, recall ${pct(t.recall)}`);
      C.lines($("#thr-chart"), { x: [0, 1], y: [0, 1], xLabel: "Alert threshold", yLabel: "Share", height: 240, label: "Precision and recall against threshold",
        shade: [[t.threshold - 0.004, t.threshold + 0.004, C.css("--ink-3"), 1]],
        series: [{ name: "precision", color: MODEL(), x: T.map((d) => d.threshold), y: T.map((d) => d.precision), label: "precision" },
                 { name: "recall", color: MODEL(), dash: "5 3", x: T.map((d) => d.threshold), y: T.map((d) => d.recall), label: "recall" }],
        tipTitle: (x) => "threshold " + f2(x), margin: { r: 64 } });
    }
    slider.addEventListener("input", draw);
    $("#thr-preset").addEventListener("click", () => { slider.value = k; draw(); });
    draw();
    // calibration plot
    const cb = M.failure.calibration_bins;
    const pts = cb.model.map((b) => ({ x: b.mean_pred, y: b.frac_pos, r: 5, color: MODEL(), alpha: 1, tip: `model bin<br>predicted ${pct(b.mean_pred)}<br>observed ${pct(b.frac_pos)}<br>n = ${b.n.toLocaleString()}` }))
      .concat(cb.baseline.map((b) => ({ x: b.mean_pred, y: b.frac_pos, r: 4, color: BASE(), alpha: 1, tip: `baseline bin<br>predicted ${pct(b.mean_pred)}<br>observed ${pct(b.frac_pos)}<br>n = ${b.n.toLocaleString()}` })));
    C.scatter($("#cal-chart"), { x: [0, 1], y: [0, 1], diagonal: true, xLabel: "Predicted failure probability", yLabel: "Observed failure rate", height: 300, width: 400, points: pts, label: "Calibration: predicted vs observed" });
    const tb = $("#cal-table tbody");
    cb.model.forEach((b) => tb.insertAdjacentHTML("beforeend", `<tr><td class="n">${pct(b.mean_pred)}</td><td class="n">${pct(b.frac_pos)}</td><td class="n">${b.n.toLocaleString()}</td></tr>`));
  }

  /* ---------------- tables built from data ---------------- */
  function tables() {
    const lt = $("#leak-table tbody");
    if (lt) M.leakage.forEach((r) => lt.insertAdjacentHTML("beforeend", `<tr><td>${r.table}</td><td><code>${r.column}</code></td>
      <td><span class="pill ${r.known_at_submission === "yes" ? "yes" : String(r.known_at_submission || "").startsWith("no") ? "no" : ""}">${r.known_at_submission}</span></td><td>${r.used_as}</td><td>${r.reason}</td></tr>`));
    const st = $("#slice-table tbody");
    if (st) [["seen_user", "user seen in development"], ["unseen_user", "user new in test period"], ["has_history", "has finished history at submission"],
      ["no_history", "no finished history at submission"], ["heavy_user_excluded", "sensitivity: heaviest user excluded"], ["heavy_user_only", "sensitivity: heaviest user only"]].forEach(([k, lab]) => {
      const r = M.runtime.slices[k], f = M.failure.slices[k];
      if (!r) return;
      st.insertAdjacentHTML("beforeend", `<tr><td>${lab}</td><td class="n">${r.model.n.toLocaleString()}</td><td class="n">${f3(r.model.mae)}</td><td class="n">${f3(r.baseline.mae)}</td>
        <td class="n">${ci(r.ci.mae_diff)}</td><td class="n">${pct(r.model.coverage_90)}</td><td class="n">${f ? f3(f.model.pr_auc) : "–"}</td><td class="n">${f ? f3(f.baseline.pr_auc) : "–"}</td><td class="n">${f ? f3(f.model.base_rate) : "–"}</td></tr>`);
    });
    const cv = $("#cv-table tbody");
    if (cv) {
      M.cv.runtime.forEach((r) => cv.insertAdjacentHTML("beforeend", `<tr><td>runtime</td><td>${r.label}</td><td class="n">MAE ${f3(r.mae)}</td></tr>`));
      M.cv.failure.forEach((r) => cv.insertAdjacentHTML("beforeend", `<tr><td>failure</td><td>${r.label}</td><td class="n">PR-AUC ${f3(r.pr_auc)}</td></tr>`));
    }
    const ec = $("#cases");
    if (ec) M.errors.cases.forEach((c) => {
      const note = (M.error_notes || {})[String(c.case)] || "";
      const isRt = c.kind === "runtime";
      const pred = isRt ? c[M.runtime_model] : c.prob;
      const facts = isRt ? `actual ${hs(c.y)} · model ${hs(pred)} · baseline ${hs(c.baseline)} · range ${hs(c.q05)}–${hs(c.q95)}`
        : `outcome ${c.y ? "failed" : "succeeded"} · model risk ${pct(pred, 0)} · baseline ${pct(c.baseline, 0)}`;
      ec.insertAdjacentHTML("beforeend", `<div class="case"><h4>Case ${c.case}: ${c.kind}, ${c.rule}</h4>
        <div class="facts">${facts}<br>${c.role_set}, ${c.gpu_total} GPU${c.gpu_total === 1 ? "" : "s"} (${c.gpu_type_spec}), user history ${c.user_n_finished || 0} finished jobs, test band ${c.band}</div>
        <p>${note}</p></div>`);
    });
  }

  /* ---------------- sticky nav state (theme follows the OS setting) ---------------- */
  function nav() {
    const links = $$(".rail a[href^='#']"), map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const obs = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { links.forEach((a) => a.removeAttribute("aria-current")); const a = map.get(e.target.id); if (a) a.setAttribute("aria-current", "true"); } }),
      { rootMargin: "-20% 0px -70% 0px" });
    $$("main section[id]").forEach((s) => obs.observe(s));
  }

  /* GitHub links: derive owner/repo from the github.io address, so the page needs no hard-coded username */
  function repoLinks() {
    const m = location.hostname.match(/^([^.]+)\.github\.io$/);
    const repo = m ? `https://github.com/${m[1]}/${location.pathname.split("/").filter(Boolean)[0] || "ES335-gpu-job-prediction"}` : null;
    $$("a[data-repo]").forEach((a) => {
      if (repo) a.href = repo + a.dataset.repo;
      else a.title = "The repository link is set when the site is served from GitHub Pages";
    });
  }

  function init() {
    repoLinks(); fillNumbers(); headline(); tables(); tryJob(); walk(); splitDemo(); ablation(); explorer(); overTime(); threshold(); nav();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
