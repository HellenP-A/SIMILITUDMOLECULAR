/* Similitud molecular multi-métrica — inferencia 100% en el navegador (RDKit.js + ONNX Runtime Web). */
"use strict";

const RDKIT_WASM = "https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2026.9.1/dist/RDKit_minimal.wasm";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const METRICS = ["tanimoto_2d", "cosine_2d", "shape_3d", "combo_3d"];
const LABELS = { tanimoto_2d: "Tanimoto 2D", cosine_2d: "Coseno 2D", shape_3d: "Shape 3D", combo_3d: "Combo 3D" };
const COLORS = { tanimoto_2d: "var(--m1)", cosine_2d: "var(--m2)", shape_3d: "var(--m3)", combo_3d: "var(--m4)" };
const EXAMPLES = {
  "Albendazol": "CCCSc1ccc2[nH]c(NC(=O)OC)nc2c1",
  "Mebendazol": "COC(=O)Nc1nc2cc(C(=O)c3ccccc3)ccc2[nH]1",
  "BZ-6 (referencia)": "CCCCC1=CC=C(C=C1)C(=O)NC1=NC2=C(N1)C=CC(Cl)=C2",
  "Análogo de BZ-4": "Cc1cccnc1C(=O)Nc1nc2cc(Br)ccc2[nH]1",
  "2-Fenilbencimidazol": "c1ccc(-c2nc3ccccc3[nH]2)cc1",
  "Miltefosina": "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C",
  "Aspirina (control negativo)": "CC(=O)Oc1ccccc1C(=O)O",
};
const DOMAIN_TEXT = {
  alto: "Dentro del dominio: la molécula se parece a los datos de entrenamiento.",
  medio: "Dominio intermedio: predicción razonable, verificar los valores cercanos.",
  bajo: "Fuera del dominio: extrapolación. Un score bajo aquí es el comportamiento esperado de un filtro; interpretar con cautela.",
};

const S = { rdkit: null, session: null, tok: null, re: null, card: null, refs: [], domain: null, nDomain: 0, report: null,
  weights: {}, nAug: 8, last: null, batch: null };
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const POP = new Uint8Array(256).map((_, i) => { let c = 0, x = i; while (x) { c += x & 1; x >>= 1; } return c; });

// ------------------------------------------------------------------ química (RDKit.js)
function withMol(smiles, fn) {
  const m = S.rdkit.get_mol(smiles);
  if (!m || !m.is_valid()) { m && m.delete(); return null; }
  try { return fn(m); } finally { m.delete(); }
}

function standardize(smiles) {
  // fragmento mayor (quita sales/solventes) + SMILES canónico
  const frags = smiles.trim().split(".").filter(Boolean);
  let best = null, bestN = -1;
  for (const f of frags) {
    const r = withMol(f, (m) => ({ can: m.get_smiles(), n: JSON.parse(m.get_descriptors()).NumHeavyAtoms }));
    if (r && r.n > bestN) { best = r.can; bestN = r.n; }
  }
  return best;
}

function randomSmiles(can, n) {
  // Igual que en Python (RenumberAtoms + canonical=False): se permuta el orden de átomos del
  // molblock y se escribe el SMILES sin canonizar, partiendo del nuevo primer átomo.
  if (n <= 0) return [];
  return withMol(can, (m) => {
    const out = new Set();
    const mb = m.get_molblock().split("\n");
    const na = parseInt(mb[3].slice(0, 3)), nb = parseInt(mb[3].slice(3, 6));
    for (let t = 0; t < n * 4 && out.size < n; t++) {
      const perm = [...Array(na).keys()];
      for (let i = na - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      const inv = new Array(na); perm.forEach((p, i) => { inv[p] = i; });
      const atoms = perm.map((p) => mb[4 + p]);
      const bonds = mb.slice(4 + na, 4 + na + nb).map((l) =>
        String(inv[parseInt(l.slice(0, 3)) - 1] + 1).padStart(3) + String(inv[parseInt(l.slice(3, 6)) - 1] + 1).padStart(3) + l.slice(6));
      const props = mb.slice(4 + na + nb).map((l) => {
        if (!/^M  (CHG|ISO|RAD)/.test(l)) return l;
        const k = parseInt(l.slice(6, 9));
        let r = l.slice(0, 9);
        for (let e = 0; e < k; e++) {
          const a = parseInt(l.slice(9 + e * 8, 13 + e * 8)), v = l.slice(13 + e * 8, 17 + e * 8);
          r += String(inv[a - 1] + 1).padStart(4) + v;
        }
        return r;
      });
      const block = [...mb.slice(0, 4), ...atoms, ...bonds, ...props].join("\n");
      const smi = withMol(block, (q) => q.get_smiles(JSON.stringify({ canonical: false })));
      if (smi && smi !== can && withMol(smi, (q) => q.get_smiles()) === can) out.add(smi);
    }
    return [...out];
  }) || [];
}

function fpBits(smiles) {
  // bit i -> byte i>>3, máscara 0x80>>(i&7)  (mismo orden que numpy.packbits)
  return withMol(smiles, (m) => {
    const s = m.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 2048 }));
    const bytes = new Uint8Array(256);
    for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) === 49) bytes[i >> 3] |= 0x80 >> (i & 7);
    return bytes;
  });
}
function popcount(a) { let c = 0; for (let i = 0; i < a.length; i++) c += POP[a[i]]; return c; }
function fpSims(a, b, offset = 0) {
  let inter = 0, nb = 0;
  for (let i = 0; i < 256; i++) { const x = b[offset + i]; inter += POP[a[i] & x]; nb += POP[x]; }
  const na = popcount(a);
  const uni = na + nb - inter;
  return { tanimoto: uni ? inter / uni : 0, cosine: na && nb ? inter / Math.sqrt(na * nb) : 0 };
}

function descriptors(can) {
  return withMol(can, (m) => {
    const d = JSON.parse(m.get_descriptors());
    const mw = d.amw, logp = d.CrippenClogP, hbd = d.NumHBD, hba = d.NumHBA, tpsa = d.tpsa, rot = d.NumRotatableBonds;
    const viol = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10);
    return [["Peso molecular", fmt(mw, 2)], ["LogP (Crippen)", fmt(logp, 2)], ["Donadores H", hbd], ["Aceptores H", hba],
      ["TPSA (Å²)", fmt(tpsa, 2)], ["Enlaces rotables", rot], ["Anillos aromáticos", d.NumAromaticRings],
      ["Violaciones Lipinski", viol], ["Cumple Veber", rot <= 10 && tpsa <= 140 ? "Sí" : "No"]];
  });
}

function svg(smiles, w = 320, h = 220, highlightSmarts = null) {
  return withMol(smiles, (m) => {
    const opts = { width: w, height: h, bondLineWidth: 1.4, addStereoAnnotation: false, clearBackground: false };
    if (highlightSmarts) {
      const q = S.rdkit.get_qmol(highlightSmarts);
      if (q && q.is_valid()) {
        const match = JSON.parse(m.get_substruct_match(q));
        if (match.atoms) Object.assign(opts, match, { highlightColour: [0.42, 0.63, 1.0] });
      }
      q && q.delete();
    }
    return m.get_svg_with_highlights(JSON.stringify(opts));
  }) || "";
}

function mcsSmarts(a, b) {
  try {
    if (typeof S.rdkit.get_mcs_as_smarts !== "function") return null;
    const list = new S.rdkit.MolList();
    const ma = S.rdkit.get_mol(a), mb = S.rdkit.get_mol(b);
    list.append(ma); list.append(mb);
    const s = S.rdkit.get_mcs_as_smarts(list, JSON.stringify({ RingMatchesRingOnly: true, CompleteRingsOnly: true, Timeout: 1 }));
    ma.delete(); mb.delete(); list.delete();
    return s || null;
  } catch { return null; }
}

// ------------------------------------------------------------------ tokenizador + modelo
function encode(smiles) {
  const L = S.tok.max_len, out = new BigInt64Array(L);
  const toks = smiles.match(S.re) || [];
  out[0] = 2n; // [BOS]
  let k = 1;
  for (const t of toks) { if (k >= L - 1) break; out[k++] = BigInt(S.tok.stoi[t] ?? 1); }
  out[k] = 3n; // [EOS]
  return out;
}
function unknownTokens(smiles) { return [...new Set((smiles.match(S.re) || []).filter((t) => !(t in S.tok.stoi)))]; }

async function runPairs(tokA, tokB) {
  const n = tokA.length, L = S.tok.max_len;
  const a = new BigInt64Array(n * L), b = new BigInt64Array(n * L);
  tokA.forEach((t, i) => a.set(t, i * L));
  tokB.forEach((t, i) => b.set(t, i * L));
  const out = await S.session.run({ tok_a: new ort.Tensor("int64", a, [n, L]), tok_b: new ort.Tensor("int64", b, [n, L]) });
  return out.similarity.data;
}

function score(row) {
  let s = 0, tw = 0;
  for (const m of METRICS) { s += row[m] * S.weights[m]; tw += S.weights[m]; }
  return tw ? s / tw : 0;
}

function domainOf(fp) {
  let best = 0;
  for (let r = 0; r < S.nDomain; r++) { const t = fpSims(fp, S.domain, r * 256).tanimoto; if (t > best) best = t; }
  const th = S.card.applicability_domain.thresholds;
  const level = best >= th.alto ? "alto" : best >= th.medio ? "medio" : "bajo";
  return { nn: best, level, mae: S.card.applicability_domain.mae_by_level[level] };
}

async function predict(smiles, nAug = S.nAug) {
  const can = standardize(smiles);
  if (!can) throw new Error("SMILES inválido");
  const variants = [can, ...randomSmiles(can, nAug - 1)];
  const q = variants.map(encode);
  const nR = S.refs.length;
  const tokA = [], tokB = [];
  for (const v of q) for (const r of S.refs) { tokA.push(v); tokB.push(r.tokens); }
  const y = await runPairs(tokA, tokB);
  const fp = fpBits(can);
  const rows = S.refs.map((r, j) => {
    const row = { ref: r, sd: 0 };
    for (let k = 0; k < 4; k++) {
      const vals = variants.map((_, v) => y[(v * nR + j) * 4 + k]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      row[METRICS[k]] = mean;
      row.sd += Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) / 4;
    }
    row.exact = fpSims(fp, r.fp);
    return row;
  });
  rows.forEach((r) => { r.score = score(r); });
  rows.sort((a, b) => b.score - a.score);
  return { can, variants: variants.length, rows, domain: domainOf(fp), unk: unknownTokens(can) };
}

// ------------------------------------------------------------------ UI: análisis individual
function bars(row) {
  return `<div class="bars">${METRICS.map((m) => `<div class="bar"><span>${LABELS[m]}</span>
    <div class="track"><div class="fill" style="width:${(row[m] * 100).toFixed(1)}%;background:${COLORS[m]}"></div></div>
    <span class="v">${fmt(row[m])}</span></div>`).join("")}</div>`;
}

function renderResult(res) {
  S.last = res;
  $("result").classList.remove("hidden");
  $("query-svg").innerHTML = svg(res.can, 340, 240);
  $("query-smiles").textContent = res.can;
  const d = res.domain;
  $("domain").innerHTML = `<span class="badge ${d.level}">${{ alto: "Dentro del dominio", medio: "Dominio intermedio", bajo: "Fuera del dominio" }[d.level]}</span>
    <p class="small">Similitud con el vecino más cercano del entrenamiento: <b>${fmt(d.nn, 3)}</b>${d.mae != null ? ` · error medio esperado ±${fmt(d.mae, 3)}` : ""}</p>
    <p class="small muted">${DOMAIN_TEXT[d.level]}</p>
    ${res.unk.length ? `<p class="small" style="color:var(--warn)">Tokens no vistos en entrenamiento: ${esc(res.unk.join(", "))}</p>` : ""}`;
  const top = res.rows[0];
  const mae2d = res.rows.reduce((a, r) => a + Math.abs(r.tanimoto_2d - r.exact.tanimoto) + Math.abs(r.cosine_2d - r.exact.cosine), 0) / (2 * res.rows.length);
  $("verify2d").innerHTML = `<table class="kv">
    <tr><td>Tanimoto vs ${esc(top.ref.id)} (modelo / exacto)</td><td>${fmt(top.tanimoto_2d, 3)} / ${fmt(top.exact.tanimoto, 3)}</td></tr>
    <tr><td>Coseno vs ${esc(top.ref.id)} (modelo / exacto)</td><td>${fmt(top.cosine_2d, 3)} / ${fmt(top.exact.cosine, 3)}</td></tr>
    <tr><td>Error medio 2D (12 referencias)</td><td>${fmt(mae2d, 4)}</td></tr></table>`;
  $("descriptors").innerHTML = descriptors(res.can).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  $("top3").innerHTML = res.rows.slice(0, 3).map((r, i) => {
    const mcs = mcsSmarts(res.can, r.ref.canonical);
    return `<div class="card top-card">
      <div class="rank-n">#${i + 1} · ${esc(r.ref.species)} · ${r.ref.activity ? esc(r.ref.activity) : `IC50 ${r.ref.ic50_um ?? "NC"} µM`}</div>
      <div class="ref-id">${esc(r.ref.id)}</div>
      <div class="score">${fmt(r.score)} <small>± ${fmt(r.sd)}</small></div>
      <div class="svg-box">${svg(r.ref.canonical, 260, 170, mcs)}</div>
      ${bars(r)}
    </div>`;
  }).join("");

  $("ranking").innerHTML = `<thead><tr><th>#</th><th class="l">Referencia</th>${METRICS.map((m) => `<th>${LABELS[m]}</th>`).join("")}
    <th>Score</th><th>±</th><th class="l">Especie</th><th>IC50 (µM)</th></tr></thead><tbody>${res.rows.map((r, i) => `
    <tr class="${i === 0 ? "best" : ""}"><td>${i + 1}</td><td class="l"><b>${esc(r.ref.id)}</b></td>${METRICS.map((m) => `<td>${fmt(r[m])}</td>`).join("")}
    <td class="score">${fmt(r.score)}</td><td class="pm">${fmt(r.sd)}</td><td class="l"><i>${esc(r.ref.species)}</i></td><td>${r.ref.ic50_um ?? "—"}</td></tr>`).join("")}</tbody>`;
}

function csv(rows) {
  return rows.map((r) => r.map((v) => (typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)).join(",")).join("\n");
}
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function analyze() {
  const smi = $("smiles").value.trim();
  if (!smi) return;
  $("run").disabled = true;
  try { renderResult(await predict(smi)); } catch (e) {
    $("result").classList.add("hidden");
    alert(`No se pudo procesar el SMILES: ${e.message}`);
  } finally { $("run").disabled = false; }
}

// ------------------------------------------------------------------ UI: lote
async function runBatch() {
  const lines = $("batch").value.split("\n").map((l) => l.trim()).filter(Boolean);
  $("batch-run").disabled = true;
  const out = [];
  for (const [i, line] of lines.entries()) {
    const [s, ...name] = line.split(/[\s,;]+/);
    try {
      const r = await predict(s, Math.min(S.nAug, 4));
      const by = Object.fromEntries(r.rows.map((x) => [x.ref.id, x.score]));
      out.push({ name: name.join(" ") || `cmp_${i + 1}`, smiles: r.can, best: r.rows[0].ref.id, max: r.rows[0].score, domain: r.domain.level, by });
    } catch (e) { out.push({ name: name.join(" ") || `cmp_${i + 1}`, smiles: s, best: "SMILES inválido", max: null, domain: "—", by: {} }); }
    $("batch-run").textContent = `Calculando ${i + 1}/${lines.length}…`;
  }
  out.sort((a, b) => (b.max ?? -1) - (a.max ?? -1));
  S.batch = out;
  $("batch-card").classList.remove("hidden");
  $("batch-dl").classList.remove("hidden");
  $("batch-table").innerHTML = `<thead><tr><th>#</th><th class="l">Nombre</th><th class="l">Mejor referencia</th><th>Score máx.</th><th class="l">Dominio</th>
    ${S.refs.map((r) => `<th>${esc(r.id)}</th>`).join("")}</tr></thead><tbody>${out.map((o, i) => `<tr><td>${i + 1}</td><td class="l" title="${esc(o.smiles)}">${esc(o.name)}</td>
    <td class="l"><b>${esc(o.best)}</b></td><td class="score">${fmt(o.max)}</td><td class="l">${o.domain === "—" ? "—" : `<span class="badge ${o.domain}">${o.domain}</span>`}</td>
    ${S.refs.map((r) => `<td>${fmt(o.by[r.id], 3)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("batch-run").disabled = false;
  $("batch-run").textContent = "Calcular lote";
}

// ------------------------------------------------------------------ UI: referencias y desempeño
function renderRefs() {
  $("ref-grid").innerHTML = S.refs.map((r) => `<div class="ref-card"><div class="svg-box">${svg(r.canonical, 240, 170)}</div>
    <h3>${esc(r.id)}</h3><div class="small"><i>${esc(r.species)}</i> · ${r.activity ? esc(r.activity) : `IC50 ${r.ic50_um ?? "NC"} µM`} · MW ${r.mw}</div>
    <div class="mono small muted">${esc(r.canonical)}</div></div>`).join("") +
    `<p class="small muted" style="grid-column:1/-1">Fuente: tabla de benzimidazoles ensayados (USAL). BZ-3 corregido: el SMILES original omitía el doble enlace C=N (forma 2,3-dihidro, MW 274.71); se usa el benzimidazol aromático de la figura (MW 272.69). 141: 80 % de inhibición a 10 µM, sin IC50. NC = no calculado.<br>
    <b>Fármacos de referencia clínica</b> (<i>L. mexicana</i>): Miltefosina IC50 8.6 µM · Anfotericina B 1.3 µM · Glucantime 35 µM.</p>`;
}

function table(el, head, rows) {
  $(el).innerHTML = `<thead><tr>${head.map((h, i) => `<th class="${i === 0 ? "l" : ""}">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((v, i) => `<td class="${i === 0 ? "l" : ""}">${typeof v === "number" ? fmt(v) : esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function renderPerf() {
  const c = S.card, ev = c.evaluation, rep = S.report, p = c.pia02_reference;
  $("kpis").innerHTML = [["R² (test externo)", ev.test_tta.r2], ["Spearman ρ", ev.test_tta.spearman], ["MAE", ev.test_tta.mae],
    ["NDCG@10 ranking", ev.ranking_test.model["ndcg@10"]], ["Parámetros", c.parameters.toLocaleString("es")], ["Pares de entrenamiento", rep.model.pairs.toLocaleString("es")]]
    .map(([l, v]) => `<div class="kpi"><div class="v">${typeof v === "number" ? fmt(v, 3) : v}</div><div class="l">${l}</div></div>`).join("");
  table("perf-compare", ["", "PIA-02 (documento)", "Esta versión"], [
    ["Validación", p.validation, "externa por scaffold (moléculas nunca vistas)"],
    ["Pares", p.pairs.toLocaleString("es"), rep.model.pairs.toLocaleString("es")],
    ["Datos 3D", "OpenEye ROCS (licencia)", "RDKit rdShapeAlign (abierto, reproducible)"],
    ["R² promedio", p.r2, ev.test_tta.r2], ["Spearman promedio", p.spearman, ev.test_tta.spearman], ["MAE promedio", p.mae, ev.test_tta.mae],
    ["Identidad (score mínimo)", "≈0.62 (BZ-6 vs sí mismo)", ev.sanity.identity_score_min],
  ]);
  table("perf-metric", ["Métrica", "R²", "Spearman", "Kendall τ", "Pearson", "MAE", "RMSE"],
    METRICS.map((m) => { const x = ev.test_per_metric[m]; return [LABELS[m], x.r2, x.spearman, x.kendall, x.pearson, x.mae, x.rmse]; }));
  const rk = ev.ranking_test, keys = Object.keys(rk.model);
  table("perf-rank", ["Método", ...keys.map((k) => k.replace("precision", "Precision").replace("ndcg", "NDCG").replace("spearman", "Spearman"))],
    [["Modelo (4 métricas, TTA)", ...keys.map((k) => rk.model[k])], ["Solo Tanimoto 2D exacto", ...keys.map((k) => rk.baseline_tanimoto_only[k])]]);
  const b = rep.baseline_3d_from_2d;
  table("perf-3d", ["Métrica", "R² regresión lineal desde 2D", "R² red neuronal", "Spearman lineal", "Spearman red"],
    ["shape_3d", "combo_3d"].map((m) => [LABELS[m], b[m].linear_from_2d.r2, b[m].model.r2, b[m].linear_from_2d.spearman, b[m].model.spearman]));
}

// ------------------------------------------------------------------ arranque
function setupUI() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${t.dataset.tab}`));
  }));
  $("examples").innerHTML = Object.keys(EXAMPLES).map((k) => `<button class="chip" data-s="${esc(EXAMPLES[k])}">${esc(k)}</button>`).join("");
  $("examples").addEventListener("click", (e) => { const s = e.target.dataset.s; if (s) { $("smiles").value = s; analyze(); } });
  $("run").addEventListener("click", analyze);
  $("smiles").addEventListener("keydown", (e) => { if (e.key === "Enter") analyze(); });
  $("weights").innerHTML = METRICS.map((m) => `<div><label>${LABELS[m]}: <b id="w-${m}">${S.weights[m].toFixed(2)}</b></label>
    <input type="range" min="0" max="1" step="0.05" value="${S.weights[m]}" data-m="${m}"></div>`).join("");
  $("weights").addEventListener("input", (e) => {
    const m = e.target.dataset.m; if (!m) return;
    S.weights[m] = parseFloat(e.target.value); $(`w-${m}`).textContent = S.weights[m].toFixed(2);
    if (S.last) { S.last.rows.forEach((r) => { r.score = score(r); }); S.last.rows.sort((a, b) => b.score - a.score); renderResult(S.last); }
  });
  $("naug").addEventListener("input", (e) => { S.nAug = +e.target.value; $("naug-v").textContent = S.nAug; });
  $("dl").addEventListener("click", () => {
    const r = S.last; if (!r) return;
    download("ranking_similitud.csv", csv([["rank", "referencia", ...METRICS, "score", "incertidumbre", "tanimoto_exacto", "coseno_exacto", "smiles_consulta"],
      ...r.rows.map((x, i) => [i + 1, x.ref.id, ...METRICS.map((m) => x[m].toFixed(4)), x.score.toFixed(4), x.sd.toFixed(4), x.exact.tanimoto.toFixed(4), x.exact.cosine.toFixed(4), r.can])]));
  });
  $("batch").value = Object.entries(EXAMPLES).map(([k, s]) => `${s} ${k.split(" (")[0].replace(/ /g, "_")}`).join("\n");
  $("batch-run").addEventListener("click", runBatch);
  $("batch-dl").addEventListener("click", () => {
    if (!S.batch) return;
    download("ranking_lote.csv", csv([["nombre", "smiles", "mejor_referencia", "score_max", "dominio", ...S.refs.map((r) => `score_${r.id}`)],
      ...S.batch.map((o) => [o.name, o.smiles, o.best, o.max == null ? "" : o.max.toFixed(4), o.domain, ...S.refs.map((r) => (o.by[r.id] ?? "").toString())])]));
  });
  $("batch-file").addEventListener("change", async (e) => {
    const f = e.target.files[0]; if (!f) return;
    const lines = (await f.text()).split(/\r?\n/).filter(Boolean);
    const head = lines[0].split(",").map((h) => h.trim().toLowerCase());
    const si = head.indexOf("smiles"), ni = head.findIndex((h) => h === "name" || h === "nombre");
    $("batch").value = si >= 0 ? lines.slice(1).map((l) => { const c = l.split(","); return `${c[si]} ${ni >= 0 ? c[ni] : ""}`.trim(); }).join("\n") : lines.join("\n");
  });
}

async function init() {
  try {
    const get = (f) => fetch(`assets/${f}`).then((r) => { if (!r.ok) throw new Error(`${f}: ${r.status}`); return r; });
    ort.env.wasm.wasmPaths = ORT_WASM;
    const [rdkit, tok, card, refs, report, dom] = await Promise.all([
      window.initRDKitModule({ locateFile: () => RDKIT_WASM }),
      get("tokenizer.json").then((r) => r.json()), get("model_card.json").then((r) => r.json()),
      get("references.json").then((r) => r.json()), get("metrics.json").then((r) => r.json()),
      get("domain_fps.bin").then((r) => r.arrayBuffer()),
    ]);
    S.rdkit = rdkit; S.card = card; S.report = report;
    S.tok = { ...tok, stoi: Object.fromEntries(tok.vocab.map((t, i) => [t, i])) };
    S.re = new RegExp(tok.regex, "g");
    S.domain = new Uint8Array(dom); S.nDomain = S.domain.length / 256;
    S.weights = { ...card.default_weights };
    S.refs = refs.map((r) => ({ ...r, tokens: encode(r.canonical), fp: fpBits(r.canonical) }));
    S.session = await ort.InferenceSession.create("assets/similarity.onnx", { executionProviders: ["wasm"] });
    setupUI(); renderRefs(); renderPerf();
    $("status").className = "status ready";
    $("status-text").textContent = `Modelo listo · ${card.parameters.toLocaleString("es")} parámetros`;
    $("run").disabled = false; $("batch-run").disabled = false;
    analyze();
  } catch (e) {
    console.error(e);
    $("status").className = "status error";
    $("status-text").textContent = `Error al cargar: ${e.message}`;
  }
}
init();
