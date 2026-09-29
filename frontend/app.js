/* IWNET single-page UI. Vanilla JS, no build step, no dependencies. */
'use strict';

const $ = (id) => document.getElementById(id);

const el = {
  health: $('health'),
  drop: $('drop'),
  file: $('file'),
  preview: $('preview'),
  clear: $('clear'),
  result: $('result'),
  spinner: $('spinner'),
  sampleGrid: $('sampleGrid'),
  classGrid: $('classGrid'),
  showProcessing: $('showProcessing'),
  viz: $('viz'),
  vizSteps: $('vizSteps'),
  vizDisclaimer: $('vizDisclaimer'),
  vizTech: $('vizTech'),
};

let modelReady = false;
let activeSample = null;
let busy = false;

/* ── health ─────────────────────────────────────────────────── */
async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    modelReady = !!data.model_loaded;
    el.health.className = 'pill ' + (modelReady ? 'pill--ok' : 'pill--bad');
    el.health.textContent = modelReady
      ? 'model ready' + (data.device ? ' · ' + data.device : '')
      : 'no model — train first';
    el.health.title = modelReady ? '' : (data.detail || '');
  } catch {
    el.health.className = 'pill pill--bad';
    el.health.textContent = 'API unreachable';
  }
}

/* ── classes + samples ──────────────────────────────────────── */
async function loadClasses() {
  try {
    const classes = await (await fetch('/api/classes')).json();
    el.classGrid.innerHTML = '';
    for (const c of classes) {
      const card = document.createElement('div');
      card.className = 'class-card' + (c.is_rejection ? ' class-card--reject' : '');
      card.innerHTML =
        '<h4></h4><p></p>';
      card.querySelector('h4').textContent = c.label;
      card.querySelector('p').textContent = c.description;
      el.classGrid.appendChild(card);
    }
  } catch { /* reference panel is non-critical */ }
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples?per_class=1');
    if (!res.ok) { el.sampleGrid.replaceChildren(); return; }
    const samples = await res.json();
    el.sampleGrid.innerHTML = '';
    for (const s of samples) {
      const btn = document.createElement('button');
      btn.className = 'sample';
      btn.type = 'button';
      btn.title = `${s.class} — ${s.filename} (ground truth)`;
      const img = document.createElement('img');
      img.src = s.url;
      img.alt = s.class;
      img.loading = 'lazy';
      const tag = document.createElement('span');
      tag.className = 'sample__label';
      tag.textContent = s.class;
      btn.append(img, tag);
      btn.addEventListener('click', () => {
        activeSample = s;
        el.file.value = '';
        el.preview.src = s.url;
        el.preview.hidden = false;
        el.clear.hidden = false;
        el.sampleGrid.querySelectorAll('.sample')
          .forEach((n) => n.classList.remove('is-active'));
        btn.classList.add('is-active');
        predict({ path: s.path });
      });
      el.sampleGrid.appendChild(btn);
    }
  } catch { /* gallery is non-critical */ }
}

/* ── prediction ─────────────────────────────────────────────── */
function setBusy(on) {
  busy = on;
  el.spinner.hidden = !on;
  el.drop.setAttribute('aria-busy', String(on));
}

function showAlert(message, info) {
  el.result.className = 'result';
  el.result.innerHTML = '';
  const box = document.createElement('div');
  box.className = 'alert' + (info ? ' alert--info' : '');
  box.textContent = message;
  el.result.appendChild(box);
}

function pct(x) { return (x * 100).toFixed(1) + '%'; }

/* ── inference visualization ────────────────────────────────────
   Renders the optional `visualization` payload returned when the request
   asked for ?visualize=true. Everything here is presentation only: it reads
   values the server already produced and never recomputes or rescales a
   probability. Where the server could not expose a stage, the step says so
   rather than showing a placeholder.
   ---------------------------------------------------------------- */
const VIZ_STEPS = [
  ['original',          'Original image'],
  ['resized',           'Resize'],
  ['normalized',        'Normalized input'],
  ['features',          'Feature representation'],
  ['channel_attention', 'Channel attention'],
  ['spatial_attention', 'Spatial attention'],
  ['multi_scale',       'Multi-scale fusion'],
  ['probabilities',     'Class probabilities'],
  ['final',             'Final prediction'],
];

const VIZ_DEFAULT_DISCLAIMER =
  'Interpretability aid. These tensors show how the network weighted its own ' +
  'features. They are not a causal explanation of the prediction and should ' +
  'not be read as a diagnosis.';

function vizStepShell(n, title, caption) {
  const li = document.createElement('li');
  li.className = 'viz__step';
  const head = document.createElement('div');
  head.className = 'viz__head';
  const num = document.createElement('span');
  num.className = 'viz__num';
  num.textContent = String(n);
  const h = document.createElement('h4');
  h.textContent = title;
  head.append(num, h);
  li.appendChild(head);
  if (caption) {
    const cap = document.createElement('p');
    cap.className = 'viz__caption';
    cap.textContent = caption;
    li.appendChild(cap);
  }
  return li;
}

function vizNa() {
  const p = document.createElement('p');
  p.className = 'viz__na';
  p.textContent = 'Not available for this image.';
  return p;
}

function vizFigure(url, caption) {
  const fig = document.createElement('figure');
  fig.className = 'viz__figure';
  const img = document.createElement('img');
  img.className = 'viz__img';
  img.src = url;
  img.alt = caption || '';
  img.setAttribute('loading', 'lazy');
  fig.appendChild(img);
  if (caption) {
    const cap = document.createElement('figcaption');
    cap.className = 'viz__figcaption';
    cap.textContent = caption;
    fig.appendChild(cap);
  }
  return fig;
}

function vizBars(rows) {
  const wrap = document.createElement('div');
  wrap.className = 'viz__bars';
  for (const r of rows) {
    const row = document.createElement('div');
    row.className = 'viz__bar' + (r.top ? ' viz__bar--top' : '');
    const name = document.createElement('span');
    name.className = 'viz__barname';
    name.textContent = r.name;
    const track = document.createElement('span');
    track.className = 'viz__track';
    const fill = document.createElement('span');
    fill.className = 'viz__fill';
    fill.style.width = Math.max(0, Math.min(100, r.value * 100)).toFixed(2) + '%';
    track.appendChild(fill);
    const val = document.createElement('span');
    val.className = 'viz__barval';
    val.textContent = r.label !== undefined ? r.label : pct(r.value);
    row.append(name, track, val);
    wrap.appendChild(row);
  }
  return wrap;
}

function vizFinalStep(n, data) {
  const li = vizStepShell(
    n,
    'Final prediction',
    'The class the model returned, with the timings measured for this request.'
  );
  const box = document.createElement('div');
  box.className = 'viz__final';
  const t = (data && data.timing_ms) || {};
  const rows = [
    ['Predicted class', data ? data.prediction : '—'],
    ['Confidence', data ? pct(data.confidence) : '—'],
    ['Inference', (t.inference !== undefined ? t.inference : '?') + ' ms'],
    ['Preprocessing', (t.preprocessing !== undefined ? t.preprocessing : '?') + ' ms'],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement('div');
    row.className = 'viz__kvp';
    const key = document.createElement('span');
    key.className = 'viz__kvkey';
    key.textContent = k;
    const val = document.createElement('span');
    val.className = 'viz__kvval';
    val.textContent = v;
    row.append(key, val);
    box.appendChild(row);
  }
  li.appendChild(box);
  return li;
}

function renderVisualization(viz, data) {
  if (!el.viz) return;
  if (!viz) {
    el.viz.hidden = true;
    el.vizSteps.replaceChildren();
    el.vizTech.replaceChildren();
    return;
  }

  el.viz.hidden = false;
  el.vizSteps.replaceChildren();
  el.vizTech.replaceChildren();
  el.vizDisclaimer.textContent = viz.disclaimer || VIZ_DEFAULT_DISCLAIMER;

  // Optional feature failed server-side: say so, still show the outcome.
  if (viz.available === false) {
    const li = vizStepShell(1, 'Processing breakdown', viz.message || 'Some processing visualizations are unavailable.');
    li.appendChild(vizNa());
    el.vizSteps.appendChild(li);
    el.vizSteps.appendChild(vizFinalStep(2, data));
    return;
  }

  const stages = viz.stages || {};
  const tech = [];

  VIZ_STEPS.forEach((entry, index) => {
    const n = index + 1;
    const key = entry[0];
    if (key === 'final') {
      el.vizSteps.appendChild(vizFinalStep(n, data));
      return;
    }
    const stage = stages[key];
    if (!stage) return;
    const li = vizStepShell(n, stage.title || entry[1], stage.caption || '');

    if (!stage.available) {
      li.appendChild(vizNa());
    } else if (key === 'channel_attention') {
      for (const s of stage.stages || []) {
        const head = document.createElement('p');
        head.className = 'viz__subhead';
        head.textContent =
          'Stage ' + s.stage + ' — top ' + s.top_weights.length + ' of ' + s.channels +
          ' channels (mean ' + s.mean.toFixed(3) + ')';
        li.appendChild(head);
        li.appendChild(vizBars(
          s.top_channels.map((c, i) => ({ name: 'ch ' + c, value: s.top_weights[i], top: i === 0 }))
        ));
        tech.push('Channel attention stage ' + s.stage + ': ' + s.channels +
          ' channels, min ' + s.min + ', max ' + s.max + ', mean ' + s.mean);
      }
    } else if (key === 'multi_scale') {
      li.appendChild(vizBars(
        (stage.scales || []).map((s) => ({
          name: 'Stage ' + s.stage + ' (' + s.height + '×' + s.width + ', ' + s.channels + ' ch)',
          value: s.weight,
          label: pct(s.weight),
        }))
      ));
      if (stage.iw_gates && stage.iw_gates.length) {
        const p = document.createElement('p');
        p.className = 'viz__subhead';
        p.textContent = 'IWAttention per-sample gate: ' + stage.iw_gates.map((g) => g.toFixed(4)).join(', ');
        li.appendChild(p);
        tech.push('IW gate (one scalar per stage): ' + stage.iw_gates.join(', ') +
          ' — ' + (stage.iw_gate_caption || ''));
      }
      tech.push('Adaptive scale weights (sum ' + stage.weight_sum + '): ' +
        (stage.scales || []).map((s) => 'stage' + s.stage + '=' + s.weight).join(', '));
    } else if (key === 'probabilities') {
      li.appendChild(vizBars(
        (stage.rows || []).map((r) => ({ name: r.name, value: r.probability, top: r.predicted }))
      ));
    } else if (key === 'normalized') {
      if (stage.image) li.appendChild(vizFigure(stage.image, 'The normalized tensor, clipped for display.'));
      if (stage.inverse_image) {
        li.appendChild(vizFigure(stage.inverse_image, stage.inverse_caption || ''));
      }
      tech.push('Normalized tensor shape: ' + (stage.height || '?') + '×' + (stage.width || '?') + '×3');
    } else if (key === 'spatial_attention') {
      if (stage.image) li.appendChild(vizFigure(stage.image, stage.overlay_caption || ''));
      if (stage.montage) li.appendChild(vizFigure(stage.montage, stage.montage_caption || ''));
      for (const m of stage.maps || []) {
        tech.push('Spatial attention stage ' + m.stage + ': ' + m.height + '×' + m.width +
          ' map, min ' + m.min + ', max ' + m.max + ', mean ' + m.mean);
      }
    } else {
      if (stage.image) li.appendChild(vizFigure(stage.image, ''));
      if (key === 'features' && stage.stages) {
        for (const s of stage.stages) {
          tech.push('Projected features stage ' + s.stage + ': ' + s.channels + '×' +
            s.height + '×' + s.width + '; top channels ' + s.top_channels.join(', '));
        }
      }
      if (key === 'original' && stage.display_downscaled) {
        tech.push('Original displayed at reduced size for transfer; prediction used the full image.');
      }
    }
    el.vizSteps.appendChild(li);
  });

  tech.push('Capture source: ' + (viz.source || 'unknown'));
  if (viz.unavailable_stages && viz.unavailable_stages.length) {
    tech.push('Stages not exposed: ' + viz.unavailable_stages.join(', '));
  }
  for (const line of tech) {
    const p = document.createElement('p');
    p.className = 'viz__techline';
    p.textContent = line;
    el.vizTech.appendChild(p);
  }
}

function renderResult(data) {
  el.result.className = 'result' + (data.is_rejection ? ' verdict--reject' : '');
  el.result.innerHTML = '';

  const verdict = document.createElement('div');
  verdict.className = 'verdict';

  const badge = document.createElement('div');
  if (data.is_rejection) {
    badge.className = 'badge badge--reject';
    badge.textContent = 'rejection class — not a grape leaf';
  }
  verdict.appendChild(badge);

  const label = document.createElement('div');
  label.className = 'verdict__label';
  label.textContent = data.label;
  verdict.appendChild(label);

  const conf = document.createElement('div');
  conf.className = 'verdict__conf';
  conf.textContent = pct(data.confidence);
  verdict.appendChild(conf);

  if (data.description) {
    const desc = document.createElement('div');
    desc.className = 'verdict__desc';
    desc.textContent = data.description;
    verdict.appendChild(desc);
  }
  el.result.appendChild(verdict);

  // probability bars, sorted high -> low
  const bars = document.createElement('div');
  bars.className = 'bars';
  const sorted = [...data.probabilities].sort((a, b) => b.probability - a.probability);
  for (const p of sorted) {
    const row = document.createElement('div');
    row.className = 'bar__row' + (p.name === data.prediction ? ' bar__row--top' : '');
    const name = document.createElement('div');
    name.className = 'bar__name';
    name.textContent = p.name;
    const track = document.createElement('div');
    track.className = 'bar__track';
    const fill = document.createElement('div');
    fill.className = 'bar__fill';
    fill.style.width = Math.max(0, Math.min(100, p.probability * 100)).toFixed(2) + '%';
    track.appendChild(fill);
    const val = document.createElement('div');
    val.className = 'bar__val';
    val.textContent = pct(p.probability);
    row.append(name, track, val);
    bars.appendChild(row);
  }
  el.result.appendChild(bars);

  if (data.note) {
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = data.note;
    el.result.appendChild(note);
  }

  const timing = document.createElement('div');
  timing.className = 'timing';
  const t = data.timing_ms || {};
  const parts = [
    'inference ' + (t.inference ?? '?') + ' ms',
    'preprocess ' + (t.preprocessing ?? '?') + ' ms',
  ];
  if (data.model_trained_at_utc) {
    parts.push('trained ' + String(data.model_trained_at_utc).slice(0, 10));
  }
  timing.textContent = parts.join(' · ');
  el.result.appendChild(timing);
}

async function predict({ blob, filename, path }) {
  if (busy) return;
  if (!modelReady) {
    showAlert('No trained model is available. Train it with "python grape.py", then restart the server.', true);
    return;
  }
  setBusy(true);
  // Visualization is opt-in and only ever rides along with a prediction. The
  // server returns the same prediction fields either way.
  const withViz = el.showProcessing && el.showProcessing.checked;
  const suffix = withViz ? (path ? '&' : '?') + 'visualize=true' : '';
  try {
    let res;
    if (path) {
      res = await fetch('/api/predict?image_path=' + encodeURIComponent(path) + suffix, { method: 'POST' });
    } else {
      const form = new FormData();
      form.append('file', blob, filename || 'image');
      res = await fetch('/api/predict' + suffix, { method: 'POST', body: form });
    }
    const data = await res.json();
    if (!res.ok) {
      showAlert(data.detail || ('Request failed with status ' + res.status));
      renderVisualization(null, data);
    } else {
      renderResult(data);
      renderVisualization(withViz ? data.visualization : null, data);
    }
  } catch (err) {
    showAlert('Could not reach the server: ' + err.message);
    renderVisualization(null, null);
  } finally {
    setBusy(false);
  }
}

/* ── input handling ─────────────────────────────────────────── */
function useFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showAlert('That file is not an image.');
    return;
  }
  activeSample = null;
  el.sampleGrid.querySelectorAll('.sample').forEach((n) => n.classList.remove('is-active'));
  const url = URL.createObjectURL(file);
  el.preview.src = url;
  el.preview.hidden = false;
  el.clear.hidden = false;
  predict({ blob: file, filename: file.name });
}

function reset() {
  activeSample = null;
  el.file.value = '';
  el.preview.hidden = true;
  el.preview.removeAttribute('src');
  el.clear.hidden = true;
  el.result.className = 'result result--empty';
  el.result.innerHTML = '<p class="muted">No prediction yet. Select an image to begin.</p>';
  el.sampleGrid.querySelectorAll('.sample').forEach((n) => n.classList.remove('is-active'));
  renderVisualization(null, null);
}

el.drop.addEventListener('click', () => el.file.click());
el.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); }
});
el.file.addEventListener('change', () => useFile(el.file.files[0]));
el.clear.addEventListener('click', reset);

['dragenter', 'dragover'].forEach((type) =>
  el.drop.addEventListener(type, (e) => {
    e.preventDefault();
    el.drop.classList.add('is-drag');
  })
);
['dragleave', 'drop'].forEach((type) =>
  el.drop.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === 'dragleave' && el.drop.contains(e.relatedTarget)) return;
    el.drop.classList.remove('is-drag');
  })
);
el.drop.addEventListener('drop', (e) => {
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  useFile(file);
});

/* ── boot ───────────────────────────────────────────────────── */
loadHealth();
loadClasses();
loadSamples();
