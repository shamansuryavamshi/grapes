# STEP 5.1 — HEALTHY ↔ BACTERIALSPOT HARD-EXAMPLE ANALYSIS

**Nature of this step: ANALYSIS ONLY.**

No training. No retraining. No fine-tuning. No model modification. No dataset
modification. No new checkpoint. No change to production inference, thresholds,
class definitions or preprocessing. The verified external image was used only as
a query/reference and was never used as a training sample.

---

## 1. Objective

Step 4 established that one verified real-world **Healthy** image
(`grape leaf.jpg`) is classified **BacterialSpot = 0.878758 / Healthy = 0.035046**,
and attributed the failure to primary category **A — feature-space class overlap**,
with 20/20 nearest training images being BacterialSpot.

Step 5.1 does not re-open that verdict. It asks a narrower, prior question:

> Is there a **systematic Healthy/BacterialSpot feature-space overlap inside the
> training data itself** that could explain why a real-world Healthy leaf lands
> deep inside BacterialSpot space?

Specifically it determines whether the overlap is (a) unique to that one external
image, (b) a systematic property of the class representation, (c) a small thin
tail of hard examples, or (d) a cluster concentrated in particular source
regions — and whether any such population resembles the verified external image.

The result is intended to inform whether a targeted Step 5.2 intervention is
justified. **It does not perform one.**

---

## 2. Frozen baseline

| Item | Value |
|---|---|
| Repository | `https://github.com/shamansuryavamshi/grapes.git` |
| Baseline commit | `e101607c78a3c61dda4e3e0d91dfe1ae9a3901a4` — *"Step 4: complete root-cause validation"* |
| Checkpoint | `Balanced_From_Sources\best_iwnet.pth` |
| Checkpoint SHA-256 | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` |
| Checkpoint size | 41,213,897 bytes |
| Architecture | IWNET, **10,161,307** parameters |
| Preprocessing | `Resize(255, bilinear) → CenterCrop(224,224) → ToTensor → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` |
| Classes | `BacterialSpot, Black_Rot, DownyMildew, Esca, Healthy, Irrelavant, PowderyMildew` |

The baseline commit was **not** amended, reset, squashed, rewritten or otherwise
touched. This step added only new files under
`RealWorldValidation\step5_1_hard_examples\`.

---

## 3. Dataset inventory

Source: `Balanced_From_Sources` (train + test), used **read-only**. Nothing was
added, removed, moved or renamed.

| Split | BacterialSpot | Healthy |
|---|---|---|
| train | 560 | 664 |
| test | 140 | 167 |
| **total analysed** | **700** | **831** |

Whole dataset: 5,929 images across 7 classes. `Balanced_Final_Split`: 7,200
files. Class counts are unchanged — hard examples were **identified, never
altered or removed**.

**Source metadata IS available.** `Balanced_From_Sources\results\split_manifest.csv`
(5,929 rows, sha256 `d25542dab5673810…`) carries per-file
`split, class, filename, source, source_group, perceptual_group, sha256`.

Two caveats found in that metadata:

* `source_group` is a **single constant string per class** (every Healthy image is
  `source_classes/train/Healthy`). It carries no discriminative information and
  was therefore not used for concentration analysis.
* The real source folder is encoded in the **filename stem**
  (`train_772_Healthy Leaves_117.jpg` → folder `train_772`). With **831 distinct
  source folders for 831 Healthy images**, the dataset contains **exactly one
  image per source folder per class**. Folder-level concentration is therefore
  *structurally impossible* to measure in this dataset — a hard methodological
  limitation, not a finding.

The 17 external images in `C:\Users\shaman\Desktop\healthy leaves` are **not
training data** and were never copied into either dataset.

---

## 4. Feature representation

| Item | Value |
|---|---|
| Layer | `head[8]` = `BatchNorm1d(64)` output — the **classifier bottleneck** |
| Dimension | 64 |
| Same as Step 4 | Yes — identical layer, identical checkpoint |
| Metric | cosine similarity on L2-normalised features (dot product) |
| Gallery | 831 Healthy + 700 BacterialSpot (cross-class), self excluded for same-class blocks |

**Extraction method.** A plain read-only `register_forward_hook` on `head[8]`
observes the tensor the forward pass had already computed. The project's existing
`IntermediateTrace` does not expose the 64-D bottleneck, so this is the same
mechanism Step 4 used. No model was rewritten and no second model was built.

**Proof the hook did not alter behaviour:** a probe compared the hooked path
against the production single-image path on 6 images —

> `max probability delta = 0.0` (exact, bit-for-bit)

**Independent reproduction of Step 4.** Recomputing the query's centroid
similarities from scratch reproduced Step 4's published numbers:

| Quantity | Step 4 reported | Step 5.1 recomputed | Match |
|---|---|---|---|
| Query → BacterialSpot centroid | 0.936431 | 0.9364308667 | ✔ |
| Query → Healthy centroid | 0.444106 | 0.4441062774 | ✔ |
| Query → strongest Healthy image | 0.915473 | 0.9154730711 | ✔ |
| BacterialSpot ↔ Healthy centroid | 0.356527 | 0.356527 | ✔ |

This is strong evidence the representation and pipeline are the ones Step 4 used.

---

## 5. Healthy → BacterialSpot analysis

Every one of the **831** Healthy images was scored against all **700**
BacterialSpot images. Full per-image output: `healthy_to_bacterialspot.csv`
(831 rows, sorted descending by cross-class top-1 similarity).

**Per-image max cosine to any BacterialSpot image:**

| statistic | value |
|---|---|
| mean | 0.714028 |
| median | 0.704883 |
| p90 | 0.798760 |
| p99 | 0.909100 |
| p99.9 | 0.950120 |
| **max** | **0.950301** |

**Absolute cross-over — independent of any percentile:**

| condition | count | of 831 |
|---|---|---|
| centroid favours the **cross** class (BacterialSpot) | **2** | 0.24% |
| centroid favours own class (Healthy) | 829 | 99.76% |
| cross-class NN **closer** than own-class NN (gap ≤ 0) | **2** | 0.24% |

The same-class/cross-class gap for Healthy images: mean 0.277956, p0.1 −0.078468,
p5 0.134624, p50 0.288436, **min −0.104485**. Only **2 of 831** Healthy images
have a *negative* gap.

**Strongest candidates (top 10):**

| # | file | cos→BS | cos→H_nn | gap | centroid margin (BS−H) | split |
|---|---|---|---|---|---|---|
| 1 | `train_671_Healthy Leaves_1.jpg` | 0.950301 | 0.999993 | 0.049692 | −0.132020 | train |
| 2 | `train_590_Healthy Leaves_213.jpg` | 0.950083 | 0.999993 | 0.049911 | −0.134259 | train |
| 3 | `train_135_Healthy Leaves_1043.png` | 0.937334 | 0.832849 | **−0.104485** | **+0.344078** | **test** |
| 4 | `train_25_Healthy Leaves_265.jpg` | 0.935886 | 0.973099 | 0.037213 | −0.338644 | train |
| 5 | `train_87_377.jpg` | 0.931443 | 0.939349 | 0.007906 | −0.296581 | train |
| 6 | `train_113_Healthy Leaves_275.jpg` | 0.925341 | 0.950407 | 0.025066 | −0.205832 | train |
| 7 | `train_314_Healthy Leaves_21.jpg` | 0.917795 | 0.985539 | 0.067744 | −0.390966 | train |
| 8 | `train_875_Healthy Leaves_677.jpg` | 0.914436 | 0.990536 | 0.076100 | −0.395272 | train |
| 9 | `train_772_Healthy Leaves_117.jpg` | 0.909101 | 0.835962 | **−0.073139** | **+0.205892** | train |
| 10 | `train_184_Healthy Leaves_1148.jpg` | 0.909097 | 0.990536 | 0.081439 | −0.422895 | train |

Note that only rows 3 and 9 have a *negative* gap. Every other top-ranked sample
still has a same-class neighbour at 0.83–0.9999, i.e. it is far closer to other
Healthy images than to any BacterialSpot image.

**Concentration:** the top-1% (9 samples) come from **9 different source folders**
(one per folder, as expected from the dataset structure); 8 train / 1 test versus
a baseline of 79.9% train. No folder-level concentration is measurable, and the
split skew is not meaningful at n=9.

---

## 6. BacterialSpot → Healthy analysis

Reciprocal analysis: every one of the **700** BacterialSpot images against all
**831** Healthy images. Full output: `bacterialspot_to_healthy.csv`.

**Per-image max cosine to any Healthy image:**

| statistic | value |
|---|---|
| mean | 0.810994 |
| median | 0.812639 |
| p90 | 0.878868 |
| p99 | 0.925174 |
| p99.9 | 0.946039 |
| **max** | **0.950301** |

**Absolute cross-over:**

| condition | count | of 700 |
|---|---|---|
| centroid favours the **cross** class (Healthy) | **2** | 0.29% |
| cross-class NN **closer** than own-class NN (gap ≤ 0) | **0** | 0% |

Gap: mean 0.184307, p0.1 0.029054, p5 0.100263, p50 0.183742, **min +0.015360**.
The gap is **never negative** for BacterialSpot — every BacterialSpot image has a
BacterialSpot neighbour closer than its nearest Healthy neighbour.

BacterialSpot hard examples sit in a *very tight internal cluster*: the top
candidates have own-class neighbours at 0.977–0.9999, so even the strongest
BacterialSpot→Healthy similarity (0.950301) is well below their own-class floor.

**Concentration:** the top-1% (7 samples) come from 7 different source folders;
6 train / 1 test. No concentration.

---

## 7. Mutual nearest neighbours

A pair (Healthy A, BacterialSpot B) is a **mutual nearest neighbour** if A's single
closest BacterialSpot image is B **and** B's single closest Healthy image is A.
This tests two-sided overlap rather than a one-sided coincidence.

**7 mutual pairs** (`mutual_neighbours.csv`):

| pair cos | splits | Healthy | BacterialSpot |
|---|---|---|---|
| **0.950301** | train/train | `train_671_Healthy Leaves_1.jpg` | `train_489_Bacterial Leaf Spot_84.jpg` |
| 0.937334 | **test**/train | `train_135_Healthy Leaves_1043.png` | `train_7_Bacterial Leaf Spot_85.png` |
| 0.935886 | train/train | `train_25_Healthy Leaves_265.jpg` | `train_357_Bacterial Leaf Spot_83.jpg` |
| 0.931443 | train/train | `train_87_377.jpg` | `train_289_Bacterial Leaf Spot_84.jpg` |
| 0.925341 | train/train | `train_113_Healthy Leaves_275.jpg` | `train_217_Bacterial Leaf Spot_2.jpg` |
| 0.909101 | train/train | `train_772_Healthy Leaves_117.jpg` | `train_6_Bacterial Leaf Spot_16.jpg` |
| 0.768996 | train/train | `train_158_cd71e97d…___Mt.N.V_HL 6185_90deg.JPG` | `train_267_Bacterial Leaf Spot_58.jpg` |

**Interpretation.** These are strong *relative to typical cross-class similarity*
(0.950 vs a cross-class mean of 0.315) but **modest relative to within-class
structure**: same-class similarities run 0.88–1.00 with a max of 1.000000
(Healthy) and 0.999940 (BacterialSpot). A 0.950 mutual cross-class pair sits
below the same-class p50. Mutual-NN status here means "closest available
cross-class partner", not "indistinguishable across classes".

---

## 8. Distribution statistics

All pairwise blocks over the 1,531-image Healthy ∪ BacterialSpot population
(`distribution_summary.json`). Self-pairs excluded from same-class blocks.

| block | n pairs | mean | median | std | min | max |
|---|---|---|---|---|---|---|
| Healthy → Healthy | 689,730 | 0.866702 | 0.892011 | 0.101277 | 0.206089 | 1.000000 |
| **Healthy → BacterialSpot** | 581,700 | **0.315020** | 0.297117 | 0.108539 | 0.038796 | **0.950301** |
| **BacterialSpot → Healthy** | 581,700 | **0.315020** | 0.297117 | 0.108539 | 0.038796 | **0.950301** |
| BacterialSpot → BacterialSpot | 489,300 | 0.900477 | 0.927447 | 0.086395 | 0.258177 | 0.999940 |
| same-class pooled | 1,179,030 | 0.880719 | 0.908919 | 0.096824 | 0.206089 | 1.000000 |
| cross-class pooled | 1,163,400 | 0.315020 | 0.297117 | 0.108539 | 0.038796 | 0.950301 |

The two cross blocks are numerically identical by construction (cosine
symmetry), which is itself a correctness check.

**Selected percentiles of the cross-class block:** p1 0.0978, p5 0.1731,
p25 0.2389, p50 0.2971, p75 0.3707, p90 0.4536, p95 0.5249, p99 0.6581,
p99.9 0.8223, max 0.950301.

**Class-centroid similarities**

| pair | cosine |
|---|---|
| Healthy centroid ↔ BacterialSpot centroid | **0.356527** |
| Query → BacterialSpot centroid | 0.936431 |
| Query → Healthy centroid | 0.444106 |

**Cross-class margin summary.** Margin is defined as
`cos(image, cross-class centroid) − cos(image, own-class centroid)`, so a
*positive* value means the cross class is favoured.

| population | mean margin | median margin | images with margin ≤ 0 (own class favoured) | images with margin > 0 (cross favoured) |
|---|---|---|---|---|
| Healthy | −0.599108 | −0.617826 | 829 / 831 | **2 / 831** |
| BacterialSpot | −0.610662 | −0.648040 | 698 / 700 | **2 / 700** |

Note that both classes have a strongly **negative** mean margin — training
images sit much closer to their own centroid than to the other one. The maxima
(+0.344078 Healthy, +0.085065 BacterialSpot) are the only cross-favoured
exceptions, and both maxima remain well below the query's +0.492325 (§11).

The two classes are strongly separated in aggregate: same-class mean 0.880719
versus cross-class mean 0.315020, a gap of **0.565698**.

---

## 9. Hard-example criteria

**Distributions were computed and written to `distribution_summary.json` BEFORE
any threshold was chosen.** No arbitrary threshold was selected and then called
meaningful. All thresholds below are explicitly-defined percentiles of the
already-reported distributions. Multiple definitions are reported, as required.

**Criterion C1 — top-percentile cross-class similarity.** Flag images whose max
cosine to any image of the other class is in the top percentile of their own
population.

**Criterion C2 — worst centroid margin.** Flag images whose cross-minus-own
centroid margin is in the top percentile (largest = most cross-favoured).

**Criterion C3 — small same-vs-cross gap.** Flag images where
`max same-class cos − max cross-class cos` is in the *bottom* percentile, i.e.
least class-separable. This is a deliberately weak/broad criterion (bottom 10%).

**Criterion C4 — absolute cross-over**, percentile-free, for reference.

| criterion | Healthy flagged | BacterialSpot flagged |
|---|---|---|
| C1 top 1% cross-class cosine | 9 / 831 | 7 / 700 |
| C2 top 1% centroid margin | 9 / 831 | 7 / 700 |
| C3 bottom 10% gap | 84 / 831 | 70 / 700 |
| C4 centroid favours cross class | 2 / 831 | 2 / 700 |
| C4 cross-NN closer than same-NN | 2 / 831 | **0 / 700** |

**Two summary sets are reported:**

* **Permissive union** (C1-p99 OR C2-p99 OR C3-p10): Healthy **85** (10.2%),
  BacterialSpot **70** (10.0%). This number is dominated by the deliberately
  broad C3 and should **not** be read as "85 images are confusable".
* **Strict set** (C1-p99.5 **AND** C2-p99.5 **AND** C3-p1 — extreme tail of all
  three simultaneously): Healthy **3** (0.36%), BacterialSpot **2** (0.29%).

**Strict set (the 5 strongest candidates):**

| class | file | cos→cross | cos→own-nn | centroid margin | split |
|---|---|---|---|---|---|
| Healthy | `train_671_Healthy Leaves_1.jpg` | 0.950301 | 0.999993 | −0.132020 | train |
| Healthy | `train_590_Healthy Leaves_213.jpg` | 0.950083 | 0.999993 | −0.134259 | train |
| Healthy | `train_135_Healthy Leaves_1043.png` | 0.937334 | 0.832849 | **+0.344078** | **test** |
| BacterialSpot | `train_489_Bacterial Leaf Spot_84.jpg` | 0.950301 | 0.993675 | −0.028713 | train |
| BacterialSpot | `train_357_Bacterial Leaf Spot_83.jpg` | 0.935886 | 0.977941 | +0.030416 | train |

These 5 are **candidates for manual label review.** No image was deleted,
relabelled, moved or modified.

---

## 10. Hard-example tables

Full tables are in `healthy_to_bacterialspot.csv` (831 rows) and
`bacterialspot_to_healthy.csv` (700 rows). Each row retains the original dataset
label, split, relative path, source folder, perceptual group, the top-1/5/10/20
cross-class similarities, both centroid similarities, the centroid margin, the
same-minus-cross gap, and the identity of the nearest cross-class partner.

Visual inspection sheets (labels and paths preserved, nothing relabelled):

* `visualizations/healthy_to_bacterialspot_contact_sheet.png` — top 15
* `visualizations/bacterialspot_to_healthy_contact_sheet.png` — top 15
* `visualizations/mutual_pairs_contact_sheet.png` — all 7 mutual pairs
* `visualizations/external_query_contact_sheet.png` — verified external reference
  vs the hard-example population
* `visualizations/pca_projection.png` — PCA of the combined 64-D bottleneck

> **Limitation — sheets not visually verified.** This environment cannot display
> images to the analyst. All five PNGs were validated **structurally** (valid
> PNG signature, decodable, dimensions recorded: 1639×1317, 1639×1317, 1969×1317,
> 1327×1317, 1155×1027 RGBA) and every tile's caption is generated directly from
> the CSV records, so labels/paths cannot be mismatched. Their **visual
> appearance has not been human-inspected** and is the one deliverable of Step
> 5.1 that still needs a human eye.

---

## 11. External Healthy query comparison

The verified external image `grape leaf.jpg` (ground truth **VERIFIED Healthy**;
production prediction BacterialSpot 0.878758 / Healthy 0.035046) was used purely
as a query. It was never trained on and its label was never altered.

| query quantity | value |
|---|---|
| cos → BacterialSpot centroid | 0.936431 |
| cos → Healthy centroid | 0.444106 |
| centroid margin (BS − Healthy) | **+0.492325** |
| max cos → any BacterialSpot image | **0.980528** |
| max cos → any Healthy image | 0.915473 |
| nearest BacterialSpot | `train_307_Bacterial Leaf Spot_85.png` (train) |
| nearest Healthy | `train_772_Healthy Leaves_117.jpg` (train) |

### The decisive comparison

| comparison | Healthy training population | query | verdict |
|---|---|---|---|
| max cos → nearest BacterialSpot | **0.950301** (best of 831) | **0.980528** | query **exceeds the entire population** |
| centroid margin (BS − Healthy) | **+0.344078** (best of 831) | **+0.492325** | query **exceeds the entire population** |
| images above the query's margin | — | — | **0 of 831 Healthy**, **0 of 700 BacterialSpot** |

**The verified external Healthy image is not a member of the hard-example
population — it is more extreme than every member of it, on both axes.** Not one
of the 1,531 Healthy/BacterialSpot training images reaches its cross-class reach
or its centroid margin.

Its 20 nearest BacterialSpot partners contain **0** BacterialSpot hard-example
candidates; its 20 nearest Healthy partners contain **7** top-1% Healthy
candidates (`external_query_comparison.json`).

**This does not mean the external image is representative of real-world Healthy
leaves.** Exactly one external image has verified ground truth. A single
verified observation cannot support any population-level claim about real-world
Healthy leaves, and none is made.

---

## 12. Duplicate / near-duplicate findings

23 cross-class pairs were checked — all 7 mutual-NN pairs plus the top-1%
one-directional pairs — reusing the project's own hashing utilities
(`iwnet.utils.sha256_file`, `perceptual_hash(algorithm="phash")`, `hamming`),
i.e. the same methodology used when the dataset was originally built, plus the
`perceptual_group` ids assigned by that build.

| check | result |
|---|---|
| pairs checked | 23 |
| **exact duplicate bytes (cross-class)** | **0** |
| **pairs sharing a perceptual group** | **0** |
| pHash Hamming distance | min **22**, median **32**, max **38** (of 64 bits) |

**Interpretation.** A Hamming distance of 22–38 is far from perceptually
identical (0 = identical). There is **no evidence of exact duplicates,
near-duplicate crops/variants, or suspicious cross-class copying** behind the
cross-class similarities. The overlap is therefore **not** a data-leakage
artefact. Duplicates are reported as findings only; nothing was removed.

`duplicate_findings.json` also records the near-duplicate `perceptual_group` id
for every pair; no group appears on both sides of the class boundary.

---

## 13. Possible visual / source patterns

**Source metadata is available** (§3) but the dataset's one-image-per-folder
structure makes folder-level concentration unmeasurable. Two weaker axes were
examined:

**Split distribution.** Healthy top-1%: 8 train / 1 test (baseline 79.9% train).
BacterialSpot top-1%: 6 train / 1 test (baseline 80% train). At n=9 and n=7 these
are not meaningful deviations.

**Trailing filename index** (`train_489_Bacterial Leaf Spot_84.jpg` → 84). A
deterministic one-sided permutation test (200,000 draws, seed 42,
`source_position_test.json`) gave:

| population | population mean | top-1% mean | n | one-sided p | significant at 0.05? |
|---|---|---|---|---|---|
| BacterialSpot | 50.50 | 72.43 | 7 | **0.0216** | yes (not at 0.01) |
| Healthy | 563.60 | 332.11 | 9 | 0.9756 (reverse direction) | no |

BacterialSpot hard examples skew *later* within their source folder. This is
reported **with a strong caveat**: the trailing filename index is an arbitrary
ordering with **no established semantic meaning**, the effect rests on n=7, it
is not significant at α=0.01, and significance would still not be causal
evidence. It is a lead for manual inspection, not a finding.

**Possible label-noise candidates.** The strongest cross-class examples were
screened for the categories listed in the Step 5.1 brief. The most plausible
candidates for *manual review* are:

* `train_135_Healthy Leaves_1043.png` (Healthy, **test** split) — the single
  strongest absolute cross-over in the dataset: gap −0.104485, centroid margin
  +0.344078, and it is in the held-out test split.
* `train_772_Healthy Leaves_117.jpg` (Healthy) — gap −0.073139, margin +0.205892;
  also the query's nearest Healthy neighbour.
* The BacterialSpot filenames ending `…_83` / `…_84` / `…_85`
  (`train_7_…_85.png`, `train_289_…_84.jpg`, `train_357_…_83.jpg`,
  `train_489_…_84.jpg`), which dominate the mutual-NN list.

> These are **candidates for manual label review**, not mislabelled samples.
> Proximity in feature space is not evidence that a label is wrong. No
> independent ground truth exists for any of them, and none is claimed.
> Disease severity gradation, mild symptom onset, image quality, framing and
> background similarity are all viable benign explanations that feature-space
> analysis alone cannot distinguish.

**Explicitly not concluded:** background causes the problem. Step 4 did not
establish that, and neither does this step. **Not concluded:** adding these
samples will fix the model.

**PCA projection** (`pca_projection.json`, `visualizations/pca_projection.png`).
Deterministic: `numpy.linalg.eigh` on the 64×64 covariance of the mean-centred
combined Healthy+BacterialSpot features, with each component's sign fixed by its
largest-magnitude loading. Explained variance: PC1 0.7310, PC2 0.0804,
PC3 0.0428, PC4 0.0389; **cumulative top-3 = 0.8542**. The three-dimensional
coordinates are in the JSON for reproduction.

> The plot is a **projection, not the actual 64-D geometry.** 14.6% of the
> variance lies outside the top 3 components. Clustering was not used and no
> clustering result is claimed as evidence of class correctness.

---

## 14. Limitations

1. **Contact sheets not visually inspected** — structural validation only (§10).
2. **n=1 verified external image.** No population-level claim about real-world
   Healthy leaves is possible or made. 16 of 17 external images have no ground
   truth and were treated as UNKNOWN; no external accuracy was calculated.
3. **One representation only.** The 64-D bottleneck was used throughout, as in
   Step 4. A different layer could show different overlap geometry. No
   alternative representation was introduced.
4. **Cosine similarity on a BatchNorm bottleneck is not a calibrated
   distance.** Values near 0.95 do not mean "visually identical".
5. **Per-image max-over-700 is a max statistic.** The per-image cross-class mean
   of 0.714 (Healthy) and 0.811 (BacterialSpot) sits far above the pairwise mean
   of 0.315 largely because it is a maximum over 700–831 candidates; it must not
   be read as typical similarity.
6. **Folder-level concentration is unmeasurable** — one image per source folder
   per class (§3).
7. **Source-position result rests on n=7** and on a semantically arbitrary index.
8. **CPU only** (`device = CPU`, `cuda_available = False`); batch size 32,
   `num_workers = 0` per the Windows-safety requirement. No multiprocessing was
   used. Feature extraction 492.75 s; analysis 0.81 s; duplicate check 1.83 s.
9. **Dataset labels are unverified.** Directory names are treated as training
   labels, not ground truth.
10. **PCA is lossy** — 14.6% of variance outside the top 3 components.

### Reproducibility

| item | value |
|---|---|
| random seed | 42, used **only** for the permutation test. Feature extraction and all similarity analysis are deterministic and use **no** randomness (no augmentation, sampling or shuffling). |
| python | 3.12.10 |
| torch | 2.14.0+cpu |
| numpy | 2.5.2 |
| device | **CPU** (`cuda_available = False`) |
| representation | `head[8]` = `BatchNorm1d(64)` output, 64-D |
| preprocessing | `Resize(255, bilinear) → CenterCrop(224,224) → ToTensor → Normalize` |
| gallery size | Healthy 831, BacterialSpot 700; dataset total 5,929 |
| top-k | k = 1, 5, 10, 20 |
| distance metric | cosine similarity on L2-normalised 64-D features |
| self-exclusion | applied to all same-class blocks |
| batch size / workers | 32 / 0 |
| hook probe | max probability delta vs production path = **0.0** |

---

## 15. Integrity verification

Full log: `verification_log.txt`. Machine-readable: `integrity_before.json`,
`integrity_after.json`. **All checks passed; no stop condition was triggered.**

| # | requirement | result |
|---|---|---|
| 1 | `best_iwnet.pth` SHA-256 unchanged | **PASS** `c1fae27c…a27c0b` |
| 2 | `best_iwnet.pth` byte size unchanged | **PASS** 41,213,897 |
| 3 | `Balanced_From_Sources` unchanged | **PASS** fingerprint `35c72a0ae283f7e8…`, 5,947 files, 1,178,341,200 bytes |
| 4 | `Balanced_Final_Split` unchanged | **PASS** fingerprint `389b24a8beac8ad5…`, 7,200 files, 1,310,546,227 bytes |
| 5 | external images unchanged | **PASS** all 17 SHA-256 digests identical |
| 6 | Step 1–4 artefacts unchanged | **PASS** all 29 files byte-identical |
| 7 | production prediction for `grape leaf.jpg` | **PASS** BacterialSpot = 0.878758, Healthy = 0.035046 (bit-for-bit identical) |
| 8 | production preprocessing unchanged | **PASS** |
| 9 | model architecture unchanged | **PASS** 10,161,307 parameters |
| 10 | weights unchanged | **PASS** checkpoint digest identical |
| 11 | class definitions unchanged | **PASS** all 7 classes identical |

Additional guarantees recorded in the log:

* Feature hook is read-only — **max probability delta 0.0**.
* No external image was written into any dataset (proved by the unchanged
  dataset fingerprints).
* The verified image was used only as a query; the other 16 as UNKNOWN.
* All class counts unchanged — hard examples were identified, never removed.
* The dataset tree fingerprints cover `best_iwnet.pth` itself, so a weight
  change could not have escaped detection.

**Test status: 161 passed, 0 failed, 0 errors, 0 skipped** — identical to the
Step 4 baseline, measured before and after the analysis. No test was deleted,
weakened, skipped or rewritten. No new test was required: Step 5.1 is analysis
only and adds no production code path.

---

## 16. Conclusion

### A. Is Healthy/BacterialSpot overlap systematic in the training representation?

**No — not in the sense of a systematic overlap between the two classes. It is a
thin tail.**

The aggregate separation is strong and unambiguous: same-class pooled mean
cosine **0.880719** versus cross-class pooled mean **0.315020**, a gap of 0.566.
Centroids sit at cosine 0.356527. Only **2 of 831** Healthy images and **0 of
700** BacterialSpot images have a cross-class nearest neighbour closer than
their own same-class nearest neighbour. Only 2 per class have a centroid that
favours the other class.

There *is* a genuine but small tail: 7 mutual nearest-neighbour pairs, 5 samples
in the strict hard-example set, and a cross-class p99.9 of 0.8223 with a maximum
of 0.950301.

**This does not overturn Step 4's category A.** Step 4's claim — that the query
lands deep inside BacterialSpot feature space — reproduces exactly here. What
Step 5.1 adds is a refinement of the *mechanism*: the region the query occupies
is **not densely populated by Healthy training data**. It is better described as
a region the Healthy training distribution **does not cover** than as a region
where Healthy and BacterialSpot training features are intermixed.

### B. How many Healthy samples are unusually close to BacterialSpot?

**3 of 831 (0.36%)** on the strict criteria. 9 of 831 (1.1%) on either strong
criterion individually. 85 of 831 (10.2%) on the deliberately broad permissive
union, which is dominated by the weak gap criterion and should not be quoted as
"85 confusable images". The single strongest absolute case is
`train_135_Healthy Leaves_1043.png` (gap −0.104485, centroid margin +0.344078),
in the **test** split.

### C. How many BacterialSpot samples are unusually close to Healthy?

**2 of 700 (0.29%)** on the strict criteria; 7 of 700 (1.0%) on either strong
criterion. **Zero** BacterialSpot images have a negative same-vs-cross gap.

### D. Are there strong mutual cross-class neighbours?

**Yes — 7 pairs**, strongest cosine 0.950301
(`train_671_Healthy Leaves_1.jpg` ↔ `train_489_Bacterial Leaf Spot_84.jpg`).
They are strong relative to the cross-class mean of 0.315 but sit **below the
same-class median of 0.909**, so they indicate tail overlap, not class merger.
None is a duplicate: 0 exact byte matches, 0 shared perceptual groups, pHash
Hamming 22–38.

### E. Does the verified external Healthy image resemble the identified hard-example population?

**No — the opposite. It exceeds it.**

* query max cos → BacterialSpot **0.980528** vs the best of all 831 Healthy images **0.950301**;
* query centroid margin **+0.492325** vs the best of all 831 Healthy images **+0.344078**;
* **0 of 831** Healthy and **0 of 700** BacterialSpot training images exceed the query on either axis;
* 0 of the query's 20 nearest BacterialSpot partners are BacterialSpot hard-example candidates.

The external image is an **outlier relative to the entire training set on the
cross-class axis**, not a member of any identifiable hard-example subpopulation.
This is the single most important result of Step 5.1.

### F. Is there evidence sufficient to justify a targeted intervention in Step 5.2?

**Not for a hard-example-cleaning intervention. The evidence argues against it.**

Removing, relabelling or re-weighting the ~5 strict candidates could not plausibly
move the query, because the query lies **outside the range of every one of them**
on both measured axes. There is no sizable, query-resembling hard-example
population to act on. Duplicates are excluded as a cause (0 found), and label
noise is unproven — 5 candidates out of 1,531 is not a label-noise finding.

What Step 5.1 *does* justify is a **diagnostic** next step, not a training step.
The evidence now points away from "the training set is internally confused about
Healthy vs BacterialSpot" and toward "the Healthy training distribution does not
cover real-world Healthy leaves". That hypothesis is **not yet established** —
it is the leading candidate for Step 5.2 to test, and testing it would require
newly verified real-world Healthy examples, which this project does not yet have
(16 of 17 external images remain UNKNOWN). No intervention should be attempted
on the current evidence.

### G. What should NOT yet be changed?

* **Model weights / architecture** — unchanged and should stay so.
* **Preprocessing** — unchanged; Step 4 did not establish it as the explanation.
* **Training datasets** (`Balanced_From_Sources`, `Balanced_Final_Split`) — unchanged.
* **Labels** — the 5 strict candidates and 7 mutual pairs are *candidates for
  manual label review*, **not** mislabelled samples. Do not relabel, delete or
  move them on the strength of feature proximity.
* **Class counts** — unchanged.
* **Thresholds, the production predictor, the API and the UI** — unchanged.
* **The verified external image** — must remain a held-out diagnostic reference;
  it must not be added to the training set or trained on. The other 16 external
  images must not be treated as ground truth.
* **The baseline commit** `e101607` must not be amended, reset, squashed or
  rewritten.

Nothing has been committed. Committing requires an explicit request.

---

### Artefact index — `RealWorldValidation\step5_1_hard_examples\`

| file | contents |
|---|---|
| `STEP5_1_REPORT.md` | this report |
| `hard_example_summary.json` | criteria, counts, strongest pairs, concentration |
| `distribution_summary.json` | all four similarity blocks, percentiles, centroids, margins |
| `healthy_to_bacterialspot.csv` | 831 rows, per-image Healthy → BacterialSpot analysis |
| `bacterialspot_to_healthy.csv` | 700 rows, per-image BacterialSpot → Healthy analysis |
| `mutual_neighbours.csv` | 7 mutual nearest-neighbour pairs |
| `external_query_comparison.json` | verified query vs both populations |
| `query_vs_population_comparison.json` | the decisive out-of-population comparison |
| `duplicate_findings.json` / `.csv` | duplicate, near-duplicate and source-position findings |
| `source_position_test.json` | permutation test for the source-position pattern |
| `pca_projection.json` | deterministic PCA coordinates + explained variance |
| `feature_extraction_meta.json` | extraction settings and the read-only hook probe |
| `integrity_before.json` / `integrity_after.json` | frozen-baseline capture and re-capture |
| `verification_log.txt` | full pass/fail integrity log |
| `visualizations/` | 4 contact sheets + PCA projection + README |
