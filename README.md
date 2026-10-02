# IWNET — Grape Leaf Disease Classifier

A 7-class grape leaf disease classifier (EfficientNet-B3 backbone with an
inception-style weighted attention block, the "IWNET" head), built as a modular
pipeline: dataset construction, training, evaluation, CLI, REST API and web UI.

**Result on the held-out test split: 96.30% accuracy, 96.33% balanced accuracy,
0.9623 macro F1** — 1,189 images that were never used for training, tuning,
model selection or early stopping.

> This is a research classifier, not a diagnostic device. Reported confidence is
> raw softmax output. It is **not calibrated** and is not a measure of
> diagnostic certainty.

> **The 96.30% above is a held-out test-split result on curated data — it is not
> a real-world accuracy figure.** Those 1,189 images come from the same curated
> source collections as the training split. A separate real-world validation was
> carried out on genuinely external photographs and it did **not** yield a usable
> generalisation estimate: only 1 of 17 external images could be given
> independent ground truth. Read
> [Real-World / External Validation](#real-world--external-validation) before
> quoting the headline number.

---

## The 7 classes

| # | Class | Meaning |
|---|-------|---------|
| 0 | `BacterialSpot` | Bacterial spot |
| 1 | `Black_Rot` | Black rot |
| 2 | `DownyMildew` | Downy mildew |
| 3 | `Esca` | Esca (black measles) |
| 4 | `Healthy` | Healthy leaf |
| 5 | `Irrelavant` | Not a grape leaf (rejection class) |
| 6 | `PowderyMildew` | Powdery mildew |

`Irrelavant` is **intentionally misspelled**. It is the spelling used by the
source folders, and the class *order* is baked into every trained checkpoint —
renaming it or reordering the list would silently reinterpret saved weights.
`iwnet.config.normalize_class_name()` maps the common variants
(`irrelevant`, `Bacterial_spot`, `Grape___healthy`, `Esca_(Black_Measles)`, …)
onto these names, and raises `UnknownClassError` for anything else rather than
guessing.

---

## Quick start

```bash
# 1. Python 3.12 + a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. PyTorch first, from the index matching your hardware
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 3. Check the environment
python grape.py --check-environment
```

Full pipeline:

```bash
python grape.py --build-dataset    # dedup + group-aware split from the sources
python grape.py --dataset-info     # counts + dataset fingerprint
python grape.py --check-dataset    # deep validation: every image decodes
python grape.py --leakage-check    # train/test duplicate audit
python grape.py                    # train, then evaluate the best checkpoint
python grape.py --serve            # web UI + API on http://127.0.0.1:8000
```

---

## Results

Trained on the group-aware split, seed 42, 40 epochs with early stopping
(stopped at epoch 38 after 10 epochs without improvement; best checkpoint from
epoch 28). ~33 s/epoch on an RTX 3050 Laptop GPU, ~25 minutes total.

### Test split (1,189 images, never trained on)

| Class | Precision | Recall | F1 | AUC | Accuracy | N |
|---|---|---|---|---|---|---|
| BacterialSpot | 0.9524 | 1.0000 | 0.9756 | 0.9999 | 100.00% | 140 |
| Black_Rot | 0.9727 | 0.9889 | 0.9807 | 0.9993 | 98.89% | 180 |
| DownyMildew | 0.9551 | 0.8713 | 0.9113 | 0.9935 | 87.14% | 171 |
| Esca | 0.9885 | 0.9773 | 0.9829 | 0.9996 | 97.73% | 176 |
| Healthy | 0.9649 | 0.9880 | 0.9763 | 0.9985 | 98.80% | 167 |
| Irrelavant | 1.0000 | 0.9947 | 0.9973 | 1.0000 | 99.47% | 187 |
| PowderyMildew | 0.9012 | 0.9226 | 0.9118 | 0.9933 | 92.26% | 168 |

**Overall:** accuracy 96.30% · balanced accuracy 96.33% · macro F1 0.9623 ·
weighted F1 0.9627 · macro precision 0.9621 · macro recall 0.9633

The weakest classes are DownyMildew and PowderyMildew, which are also the two
that are most often confused with each other in the field. The rejection class
(`Irrelavant`) is the strongest — unsurprising, since its negatives come from a
visually distinct source collection.

Reproduce with `python grape.py --evaluate`. Figures and CSV reports land in
`Balanced_From_Sources/results/`.

---

## Real-World / External Validation

> **The 96.30% figure above is the held-out test-set result, not the
> external real-world validation result.** The two measure different things and
> must not be compared or interchanged.

| | Held-out test evaluation | External / real-world validation |
|---|---|---|
| Images | 1,189 | 17 |
| Ground-truth labels available | all 1,189 | 1 `VERIFIED`, 16 `UNKNOWN` |
| Images that can be scored | 1,189 | **1** |
| Result | 96.30% accuracy, macro F1 0.9623 | **0 / 1** |

### The workflow: Step 5.1, Step 5.2, Step 5.3

`RealWorldValidation/` holds a separate, read-only real-world validation
workflow, performed against the frozen checkpoint:

| Step | Question | Conclusion |
|---|---|---|
| **Step 5.1** | Hard-example analysis of the two Healthy ↔ BacterialSpot confusion directions in feature space. | Characterises the confusable pairs; no model change. |
| **Step 5.2** | Illumination diagnostic across 43 brightness / contrast / gamma / shadow / correction variants. | Illumination sensitivity is **insufficient** to explain the observed failure. |
| **Step 5.3** | Ground-truth validation on 17 genuinely external images. | **`CATEGORY_A` — insufficient ground truth to judge real-world performance.** |

### Step 5.3: how ground truth was decided

Ground truth came only from **independent evidence**. The model prediction,
filenames, pixel statistics, training-set similarity and visual impression were
all explicitly excluded as evidence. Each image is adjudicated in
`RealWorldValidation/step5_3_external_validation/external_ground_truth.csv`
under a `final_ground_truth_status`:

| `final_ground_truth_status` | Count |
|---|---|
| `VERIFIED` | 1 |
| `SUPPORTED` | 0 |
| `DISPUTED` | 0 |
| `UNKNOWN` | 16 |

**16 of the 17 external images remain `UNKNOWN`.** This is not a search that
gave up early: the provenance attempts are logged in full — including the
failures — in `evidence_log.csv`, and the other 16 files carry no EXIF, XMP,
IPTC, JPEG comment or embedded URL. Reverse-image search and analyst visual
inspection were unavailable and are recorded as tooling gaps rather than
omitted. Only one image has a machine-readable source, and that source confirms
the image's identity while making **no** statement about plant health.

The verified subset therefore contains **exactly one usable ground-truth
example**.

### The verified result — and what it does not mean

The single `VERIFIED` image was misclassified:

| Field | Value |
|---|---|
| File | `grape leaf.jpg` |
| Ground truth | `Healthy` |
| Prediction | `BacterialSpot` |
| `BacterialSpot` probability | 0.8788 |
| `Healthy` probability | 0.0350 |
| Top-1 / top-2 margin | 0.8437 |
| **Verified accuracy** | **0 / 1** |

**0/1 is not the model's real-world accuracy.** It is a single observation about
a single file, and specifically:

- **A sample of n = 1 cannot support a generalisation estimate.** No confidence
  interval is quoted, because any interval on 0/1 spans effectively the entire
  range and would misrepresent the evidence.
- It is **not** an error rate, **not** an accuracy percentage, and says nothing
  about any other image.
- The 16 `UNKNOWN` images are **excluded from every calculation** rather than
  guessed. Assigning labels to them in order to score them would manufacture the
  very ground truth this step exists to obtain.
- Only `Healthy` has a verified example. There are **zero** verified
  `BacterialSpot` images, so the opposite error direction is entirely
  unmeasured and cannot be compared.
- The verified error was the model's *most confident* external prediction
  (margin 0.8437), not a borderline call near the decision boundary.

The practical conclusion of Step 5.3 is that the **binding constraint is the
availability of independent ground truth, not the model**. On this evidence,
**retraining is not justified** — and no automatic relabelling of any image was
performed.

Full method, evidence log, per-image adjudication and analysis:
[`RealWorldValidation/step5_3_external_validation/STEP5_3_REPORT.md`](RealWorldValidation/step5_3_external_validation/STEP5_3_REPORT.md).

---

## How the dataset is built

This is the part that determines whether the number above means anything.

**Sources (read-only, never modified):**
- `Balanced_Final_Split/Balanced_Final_Split` — 7,200 images, 6 disease classes
- `Irrelevant/Irrelevant` — 900 images, the rejection class

**Pipeline:**

1. **Collect** — 8,100 images.
2. **Global SHA-256 dedup** — 2,171 exact duplicates removed. All 2,171 were
   *within* a single class, so no cross-class contradiction was involved (the
   build refuses to silently resolve a cross-class exact duplicate; it reports
   it). → **5,929 unique images**.
3. **Group near-duplicates** — see below. 5,778 perceptual groups; 351 images
   belong to a group of more than one image.
4. **Group-aware stratified split** — 20% test, taken *group by group* so a
   cluster of near-identical images can never straddle the boundary. The
   remaining 80% is stratified per class.
5. **Validation carve-out** — 15% of *train* becomes the validation set
   (4,029 train / 711 val / 1,189 test). Validation never touches the test split.

| Class | Train | Test | Total |
|---|---|---|---|
| BacterialSpot | 560 | 140 | 700 |
| Black_Rot | 720 | 180 | 900 |
| DownyMildew | 682 | 171 | 853 |
| Esca | 700 | 176 | 876 |
| Healthy | 664 | 167 | 831 |
| Irrelavant | 746 | 187 | 933 |
| PowderyMildew | 668 | 168 | 836 |
| **Total** | **4,740** | **1,189** | **5,929** |

Dataset fingerprint (SHA-256 over the full split manifest):
`0c27cf041e35f4f74b8e862ff143ef3975a7387ae77e5436fb4c1bc213fde3bb`

That fingerprint is stored in the dataset report *and* inside every checkpoint,
so any result can be tied back to the exact images that produced it.

### Why perceptual grouping, not just SHA-256

The sources are not 8,100 independent photographs. They contain re-encoded and
lightly augmented copies of the same leaf: distinct files, so SHA-256 dedup
misses them, but near-identical once decoded. A plain random split puts such
copies on both sides of the train/test boundary, and the model then scores well
on a test image it has effectively already memorised.

Grouping uses a 64-bit **perceptual hash** (DCT-based, scale-invariant).
Images within a Hamming distance of `<= 6` are merged with union-find into one
group, and a group is assigned wholly to train or wholly to test.

An earlier attempt used **aHash** at a similar threshold. It produced false
positives: `aHash` collapses to a mean intensity signature, so at that
threshold genuinely unrelated leaves merged into one "group", which then forced
whole clusters onto one side of the split and quietly skewed the class
distribution. pHash was chosen because it keys on structure rather than
brightness. The threshold `6` is recorded in `config.yaml` and reported in every
run, so the number is not an unfalsifiable knob.

Verify it yourself:

```bash
python grape.py --leakage-check
```

Current result: 1,189 candidates actually searched, **0** exact duplicates, **0**
cross-class duplicates, **0** near-duplicates, **0** filename collisions, **0**
within-split hash collisions.

> The audit compares within each class, so a perceptual near-copy that carries
> *two different labels* is not part of the near-duplicate search. Exact
> cross-class duplicates are still caught, by the dedup step.

---

## Training

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B3 (`out_indices=[1,2,3]`) |
| Head | IW attention + adaptive scale fusion, embedding 128, bottleneck 64 |
| Image size | 224 |
| Batch size | 8 |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | 3-epoch linear warmup, then cosine decay to exactly 0 |
| Progressive unfreezing | last 3 backbone stages at epoch 10, at 1% of base lr |
| Regularisation | MixUp (α=0.3, p=0.5), label smoothing 0.1, dropout 0.5/0.3, weight decay |
| Class imbalance | inverse-frequency class weights |
| Early stopping | patience 10, on validation accuracy |
| Precision | AMP, gradient clipping at 1.0 |
| Seed | 42 |

Two details that are easy to get wrong and were fixed here:

- **Frozen BatchNorm.** Setting `requires_grad=False` does *not* stop BatchNorm
  running statistics from updating. The pretrained backbone's running
  means/variances drifted on this small dataset before it was ever trained. The
  backbone's BatchNorm layers are now pinned to `eval()` for as long as it is
  frozen.
- **MixUp must honour the configured loss.** The MixUp loss was calling
  `F.cross_entropy` directly, silently discarding the class weights and label
  smoothing on exactly the batches where the minority classes are hardest. It
  now uses the same criterion module as the clean batches.

---

## CLI

`python grape.py --help` lists every option and the 7 classes.

| Command | Effect |
|---|---|
| `--check-environment` | Python / PyTorch / CUDA / dependency check. Non-zero exit if broken. |
| `--build-dataset` | Rebuild the dataset from the read-only sources. |
| `--dataset-info` | Per-class counts, manifest path, dataset fingerprint. |
| `--check-dataset` | Deep validation: class coverage and full decode of every image. |
| `--leakage-check` | Exact / cross-class / near / filename duplicate audit. |
| *(none)* | Train, then evaluate the best checkpoint on the test split. |
| `--resume` | Continue from `last_iwnet.pth`. |
| `--evaluate` | Evaluate an existing checkpoint. No training, no tuning. |
| `--replot` | Regenerate figures and CSV reports. |
| `--predict IMAGE` | Classify a single image. |
| `--serve` | Run the API and the web UI. |

Any config value can be overridden: `--set train.epochs=60`.

Numeric flags are validated at parse time, so `--epochs 0` fails immediately
rather than training nothing and reporting success. Contradictory combinations
are refused — `--augment` cannot be combined with `--skip-checks`, because
offline augmentation rewrites the training split and the dataset and leakage
checks must run afterwards.

---

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | Always 200. Reports `status`, `model_loaded`, `device`, `classes`. |
| GET | `/api/model` | Architecture, classes, epoch, val accuracy, dataset fingerprint. |
| GET | `/api/classes` | Class order and descriptions. Works without a model. |
| POST | `/api/predict` | `multipart/form-data`, field name `file`. |
| GET | `/api/samples` | A gallery of test images. |
| GET | `/api/samples/{class}/{filename}` | One test image. Path-traversal guarded. |

**Degraded startup is intentional.** With no checkpoint present, the server still
boots: `/api/health` returns 200 with `status: "degraded"`, and `/api/model`,
`/api/predict` and `/api/samples` return 503 with an actionable message. It does
not crash.

`POST /api/predict`:

```bash
curl -X POST http://127.0.0.1:8000/api/predict -F "file=@leaf.jpg"
```

```json
{
  "prediction": "Black_Rot",
  "label": "Black Rot",
  "confidence": 0.8768,
  "is_rejection": false,
  "description": "...",
  "probabilities": [{"name": "Black_Rot", "probability": 0.8768}, "..."],
  "timing_ms": {"preprocessing": 4.1, "inference": 18.7},
  "confidence_note": "Softmax model confidence. Confidence is not calibrated and is not a measure of diagnostic certainty."
}
```

Checks enforced: upload size limit, pixel-count limit, a real image decode
(corrupt bytes are rejected with 400, not 500), and path traversal on the sample
endpoints.

---

## Deploying

The API is stateless and loads the model once at startup. Loading never needs
network access, so the container starts fully offline.

```bash
docker build -t iwnet .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/Balanced_From_Sources:/app/Balanced_From_Sources:ro" \
  iwnet
```

The mount point must be `<project>/Balanced_From_Sources`, which is where
`iwnet/config.py` resolves the dataset and checkpoints. The container runs as a
non-root user and ships a healthcheck.

The base image is CPU-only. For GPU serving, install torch from the CUDA index
during the build and run with `--gpus all`; expect roughly 39 MB per checkpoint
and ~4 GB VRAM during training.

**Do not bind to `0.0.0.0` without a reverse proxy.** The endpoint is
unauthenticated and unauthenticated-by-design: there is no per-user quota, and
`/api/samples` will serve any file inside the dataset directory.

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

161 tests, ~2 min on CPU, no GPU required (they run on tiny synthetic images and
a synthetic B0 model). They cover the parts where a silent wrong answer is
possible rather than a crash:

- **Class contract** — exact names, order, the `Irrelavant` spelling, alias
  mapping, and that unknown folders raise instead of being guessed.
- **Grouped splitting** — a perceptual group can never appear in both splits;
  the split is deterministic and loses no image; planted exact, cross-class and
  resized near-duplicates are all *detected* by the audit.
- **Checkpoints** — round-trip reproduces the original predictions exactly;
  mismatched class order, geometry or architecture are refused; loading
  performs no network request.
- **Frozen BatchNorm** — stays in `eval()` while the backbone is frozen.
- **Scheduler** — first epoch has non-zero LR, warmup rises, cosine decays
  monotonically to exactly 0, and a group added at unfreeze keeps its own
  reduced base LR.
- **MixUp** — the loss is exactly `lam*CE(y_a) + (1-lam)*CE(y_b)`, and class
  weights and label smoothing actually change it.
- **ASCII-safe reporting** — the summary table encodes in cp1252. This crashed a
  completed run *after* the checkpoint was written.
- **API** — degraded startup, 503s, corrupt and oversized uploads, path
  traversal, and that the confidence disclaimer is actually sent.
- **CLI** — numeric validation, mutually exclusive flags, `--help` contents.

Several tests are explicit regression tests for bugs that produced a *plausible
but wrong* number rather than an error: a leakage search that silently skipped
its candidates, `--evaluate` using the current config instead of the
checkpoint's own geometry, and a dataset fingerprint that was written but never
displayed.

---

## Layout

```
iwnet/
  config.py            dataclass config, the 7 classes, the alias table, SEED
  utils.py             pHash/dHash/aHash, SHA-256, class normalisation, device
  data/
    build.py           dedup, perceptual grouping, group-aware split, fingerprint
    leakage.py         exact / cross-class / near / filename duplicate audit
    dataset.py         transforms, validation split, loaders
    validation.py      deep image + class validation
  model/
    architecture.py    IWNET: EfficientNet-B3 + IW attention + scale fusion
    checkpoint.py      self-describing checkpoints, validation, atomic writes
  training/
    trainer.py         training loop, warmup+cosine, MixUp, progressive unfreezing
    metrics.py         classification metrics, ASCII-safe reporting
    plots.py           figures and CSV reports
  inference/
    predictor.py       cached checkpoint inference
  api/
    app.py             FastAPI service, degraded startup
frontend/              no-build web UI (vanilla HTML/CSS/JS)
tests/                 pytest suite
grape.py               CLI entry point
config.yaml            the exact configuration behind the reported result
RealWorldValidation/   Step 5.1-5.3 real-world / external validation (read-only)
```

## Reproducing the reported number

```bash
python grape.py --build-dataset
python grape.py --leakage-check     # must report 0 blocking duplicates
python grape.py                     # 40 epochs, early stopping
python grape.py --evaluate
```

The dataset fingerprint must be
`0c27cf041e35f4f74b8e862ff143ef3975a7387ae77e5436fb4c1bc213fde3bb`. If it
differs, the split changed and the result is not comparable.

Training is stochastic only through CUDA kernel non-determinism; the split,
augmentation and shuffling are seeded from `iwnet.config.SEED = 42`. Expect the
final test accuracy to land within a few tenths of a percent of 96.30%, not
exactly on it.
