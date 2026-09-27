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
  try {
    let res;
    if (path) {
      res = await fetch('/api/predict?image_path=' + encodeURIComponent(path), { method: 'POST' });
    } else {
      const form = new FormData();
      form.append('file', blob, filename || 'image');
      res = await fetch('/api/predict', { method: 'POST', body: form });
    }
    const data = await res.json();
    if (!res.ok) {
      showAlert(data.detail || ('Request failed with status ' + res.status));
    } else {
      renderResult(data);
    }
  } catch (err) {
    showAlert('Could not reach the server: ' + err.message);
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
