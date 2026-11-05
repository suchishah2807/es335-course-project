/* Minimal SVG chart helpers (no external libraries). Colours come from CSS variables so light/dark both work. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function scale(d0, d1, r0, r1) {
    const f = (x) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.inv = (y) => d0 + ((y - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, step0 = span / (n || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= (n || 5)) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmt = (v, d) => (v == null || isNaN(v) ? "–" : (+v).toFixed(d == null ? 2 : d));

  /* tooltip shared by all charts */
  let tip;
  function showTip(html, ev) {
    if (!tip) { tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "status"); document.body.appendChild(tip); }
    tip.innerHTML = html; tip.style.display = "block";
    const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hideTip() { if (tip) tip.style.display = "none"; }

  /* frame: axes + grid; returns {svg, g, x, y, W, H} */
  function frame(container, o) {
    container.innerHTML = "";
    const W = Math.round(container.clientWidth) || o.width || 720, H = o.height || 280, m = Object.assign({ t: 12, r: 16, b: 44, l: 56 }, o.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "" }, container);
    const x = scale(o.x[0], o.x[1], m.l, W - m.r), y = scale(o.y[0], o.y[1], H - m.b, m.t);
    const grid = el("g", { class: "grid" }, svg);
    (o.yTicks || niceTicks(o.y[0], o.y[1], 5)).forEach((t) => {
      el("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }, grid);
      el("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = o.yFmt ? o.yFmt(t) : t;
    });
    (o.xTicks || niceTicks(o.x[0], o.x[1], 6)).forEach((t) => {
      el("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = o.xFmt ? o.xFmt(t) : t;
    });
    el("line", { class: "domain", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, svg);
    if (o.xLabel) el("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = o.xLabel;
    if (o.yLabel) el("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle" }, svg).textContent = o.yLabel;
    return { svg, x, y, W, H, m };
  }

  /* line chart with optional shaded x-ranges and hover crosshair */
  function lines(container, o) {
    const f = frame(container, o);
    (o.shade || []).forEach((s) => {
      el("rect", { x: f.x(s[0]), y: f.m.t, width: Math.max(0, f.x(s[1]) - f.x(s[0])), height: f.H - f.m.b - f.m.t,
                   fill: s[2] || css("--model-soft"), opacity: s[3] == null ? 1 : s[3] }, f.svg);
      if (s[4] && f.x(s[1]) - f.x(s[0]) > 7 * s[4].length + 8) el("text", { x: f.x(s[0]) + 4, y: f.m.t + 12 }, f.svg).textContent = s[4];
    });
    o.series.forEach((s) => {
      let d = "";
      s.x.forEach((xv, i) => { const yv = s.y[i]; if (yv == null || isNaN(yv)) return; d += (d && s.y[i - 1] != null ? "L" : "M") + f.x(xv).toFixed(1) + "," + f.y(yv).toFixed(1); });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }, f.svg);
      if (s.label) { const i = s.x.length - 1; el("text", { x: f.x(s.x[i]) + 4, y: f.y(s.y[i]) + 4, fill: s.color }, f.svg).textContent = s.label; }
    });
    (o.hlines || []).forEach((h) => {
      el("line", { x1: f.m.l, x2: f.W - f.m.r, y1: f.y(h.y), y2: f.y(h.y), stroke: h.color || css("--baseline"), "stroke-width": 1 }, f.svg);
      if (h.label) el("text", { x: f.W - f.m.r, y: f.y(h.y) - 5, "text-anchor": "end" }, f.svg).textContent = h.label;
    });
    const cross = el("line", { y1: f.m.t, y2: f.H - f.m.b, stroke: css("--ink-3"), "stroke-width": 1, visibility: "hidden" }, f.svg);
    const hit = el("rect", { x: f.m.l, y: f.m.t, width: f.W - f.m.l - f.m.r, height: f.H - f.m.t - f.m.b, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (ev) => {
      const pt = f.svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      const lx = pt.matrixTransform(f.svg.getScreenCTM().inverse()).x, xv = f.x.inv(lx);
      const s0 = o.series[0]; let i = 0, best = Infinity;
      s0.x.forEach((v, k) => { if (Math.abs(v - xv) < best) { best = Math.abs(v - xv); i = k; } });
      cross.setAttribute("x1", f.x(s0.x[i])); cross.setAttribute("x2", f.x(s0.x[i])); cross.setAttribute("visibility", "visible");
      showTip((o.tipTitle ? o.tipTitle(s0.x[i]) : s0.x[i]) + "<br>" + o.series.map((s) => `${s.name}: ${fmt(s.y[i], o.tipDigits)}`).join("<br>"), ev);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
    return f;
  }

  /* horizontal dot-and-interval rows, e.g. model vs baseline with 95% CI */
  function dotRow(container, o) {
    container.innerHTML = "";
    const W = Math.round(container.clientWidth) || 520, H = 54, m = { l: 8, r: 18 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "" }, container);
    const x = scale(o.domain[0], o.domain[1], m.l, W - m.r);
    el("line", { x1: m.l, x2: W - m.r, y1: 40, y2: 40, stroke: css("--rule") }, svg);
    niceTicks(o.domain[0], o.domain[1], 5).forEach((t) => {
      el("line", { x1: x(t), x2: x(t), y1: 37, y2: 43, stroke: css("--rule") }, svg);
      el("text", { x: x(t), y: 54, "text-anchor": "middle", style: "font: 10px var(--mono); fill: var(--ink-3)" }, svg).textContent = o.tickFmt ? o.tickFmt(t) : t;
    });
    o.points.forEach((p, i) => {
      const yy = 14 + i * 12;
      if (p.lo != null) el("line", { x1: x(p.lo), x2: x(p.hi), y1: yy, y2: yy, stroke: p.color, "stroke-width": 2 }, svg);
      el("circle", { cx: x(p.v), cy: yy, r: 5, fill: p.color, stroke: css("--surface"), "stroke-width": 2 }, svg);
    });
  }

  /* vertical bars with CI whiskers */
  function bars(container, o) {
    const n = o.items.length;
    const crowd = (container.clientWidth || 600) / n < 90;
    const f = frame(container, Object.assign({}, o, { x: [0, n], xTicks: [], margin: crowd ? { b: 78 } : undefined,
                                                       xLabel: undefined }));
    const bw = (f.x(1) - f.x(0)) * 0.55;
    o.items.forEach((it, i) => {
      const cx = f.x(i + 0.5), y0 = f.y(Math.max(o.y[0], 0));
      const top = f.y(it.v);
      el("rect", { x: cx - bw / 2, y: Math.min(top, y0), width: bw, height: Math.abs(y0 - top), rx: 2, fill: it.color,
                   opacity: it.dim ? 0.35 : 1 }, f.svg);
      if (it.lo != null) {
        el("line", { x1: cx, x2: cx, y1: f.y(it.lo), y2: f.y(it.hi), stroke: css("--ink"), "stroke-width": 1.5 }, f.svg);
        [it.lo, it.hi].forEach((v) => el("line", { x1: cx - 5, x2: cx + 5, y1: f.y(v), y2: f.y(v), stroke: css("--ink"), "stroke-width": 1.5 }, f.svg));
      }
      const ty = f.H - f.m.b + 16;
      const t = el("text", crowd ? { x: cx, y: ty, "text-anchor": "end", transform: `rotate(-35 ${cx} ${ty})` }
                                 : { x: cx, y: ty, "text-anchor": "middle" }, f.svg);
      t.textContent = it.label;
      const hit = el("rect", { x: cx - bw / 2 - 6, y: f.m.t, width: bw + 12, height: f.H - f.m.t - f.m.b, fill: "transparent" }, f.svg);
      hit.addEventListener("mousemove", (ev) => showTip(it.tip || `${it.label}: ${fmt(it.v, 3)}`, ev));
      hit.addEventListener("mouseleave", hideTip);
    });
    return f;
  }

  /* scatter with click handler */
  function scatter(container, o) {
    const f = frame(container, o);
    if (o.diagonal) el("line", { x1: f.x(o.x[0]), y1: f.y(o.x[0]), x2: f.x(o.x[1]), y2: f.y(o.x[1]), stroke: css("--baseline"), "stroke-width": 1 }, f.svg);
    const g = el("g", {}, f.svg);
    o.points.forEach((p) => {
      const c = el("circle", { cx: f.x(p.x), cy: f.y(p.y), r: p.r || 3, fill: p.color, "fill-opacity": p.alpha == null ? 0.55 : p.alpha,
                               stroke: p.stroke || "none", "stroke-width": p.stroke ? 2 : 0 }, g);
      if (o.onClick) { c.style.cursor = "pointer"; c.addEventListener("click", () => o.onClick(p)); }
      if (p.tip) { c.addEventListener("mousemove", (ev) => showTip(p.tip, ev)); c.addEventListener("mouseleave", hideTip); }
    });
    return f;
  }

  window.Charts = { lines, dotRow, bars, scatter, frame, el, css, fmt, niceTicks, showTip, hideTip };
})();
