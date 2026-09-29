# Step 4 — Root-Cause Validation

**Project:** `C:\Users\shaman\Desktop\suryavamshi`
**Subject:** `C:\Users\shaman\Desktop\healthy leaves\grape leaf.jpg`
**Ground truth:** Healthy (VERIFIED in `RealWorldValidation\manifest.csv` — the only one of the 17 external images with verifiable ground truth)
**Nature of this step:** read-only controlled validation. No retraining, no fine-tuning, no modification of `best_iwnet.pth`, architecture, preprocessing, inference logic, class definitions, thresholds or either dataset. The production prediction was not changed and no heuristic correction was added.

Throughout this report:

- **MEASURED FACT** means a number produced by an instrumented run and reproducible from the artefacts in this directory.
- **POSSIBLE EXPLANATION** means an interpretation that the measurements permit but do not establish.
- The 16 UNKNOWN external images are never called correct or incorrect. Their labels are UNKNOWN and stay UNKNOWN.
- No attention statistic is described as detecting a lesion, a spot or a disease.

---

## 1. Confirmed failure

Reproduced through the exact production path (`iwnet.inference.predictor`, unchanged transform, unchanged forward pass, batch size 1, CPU):

| quantity | value |
|---|---|
| source file | `grape leaf.jpg`, 429 238 bytes, 1920 × 1440 |
| production transform | `Resize(255, bilinear)` → 255 × 340 → `CenterCrop((224,224))` → 224 × 224 → `ToTensor` → `Normalize` |
| source window reaching the network | central 1264.9 × 1264.9 px = **57.87 % of the frame area** |
| input tensor | `(3, 224, 224)` float32, range `[-2.1179, 2.4134]`, mean `0.0901` |
| **predicted class** | **BacterialSpot** |
| **confidence** | **0.878758 (87.8758 %)** |
| **Healthy probability** | **0.035046 (3.5046 %)** |
| probability sum | 1.000000 |
| `is_rejection` | `False` |

Full distribution (MEASURED FACT):

| class | probability |
|---|---|
| BacterialSpot | 0.878758 |
| Healthy | 0.035046 |
| Irrelavant | 0.030412 |
| Black_Rot | 0.024897 |
| DownyMildew | 0.014695 |
| PowderyMildew | 0.010187 |
| Esca | 0.006004 |

Full internal readout for the unmodified original (MEASURED FACT; raw values in `variant_predictions.csv`, row `A`):

| quantity | value |
|---|---|
| feature A — `head[0]` AdaptiveAvgPool2d output, the tensor the classifier consumes | 128-d |
| feature B — `head[8]` BatchNorm1d output, the bottleneck feeding the final classifier | 64-d |
| stage-1 projected feature (128-d after `proj1`) | mean −0.003666, std 0.098352, absmax 0.447009 |
| stage-2 projected feature | mean −0.008035, std 0.428743, absmax 2.694474 |
| stage-3 projected feature | mean 0.001615, std 1.311543, absmax 7.980972 |
| stage-1 channel attention | mean 0.466480, max 0.570571, std 0.056168, entropy 3.458540 nats |
| stage-2 channel attention | mean 0.411549, max 0.839567, std 0.219911, entropy 3.720220 nats |
| stage-3 channel attention | mean 0.495402, max 0.823390, std 0.175416, entropy 4.844210 nats |
| stage-1 spatial attention | mean 0.999995, max 1.000000, entropy 8.050700 nats |
| stage-2 spatial attention | mean 0.848812, max 1.000000, entropy 6.514660 nats |
| stage-3 spatial attention | mean 0.479137, max 0.993745, entropy 4.915960 nats |
| IW gate 1 | 0.014716 |
| IW gate 2 | 0.091832 |
| IW gate 3 | 0.410864 |
| adaptive scale weights | stage 1 = 0.057653, stage 2 = 0.005524, stage 3 = **0.936823** |
| logit BacterialSpot | **+3.395030** |
| logit Healthy | +0.173183 |
| logit Irrelavant | +0.031370 |
| logit Black_Rot | −0.168717 |
| logit DownyMildew | −0.695957 |
| logit PowderyMildew | −1.062353 |
| logit Esca | −1.591018 |
| top-1 / top-2 margin | **+3.221847** (top-2 is Healthy) |
| logit spread (max − min) | +4.98605 |

Observer fidelity: the custom read-only hooks used for the gallery were checked against the shipped `capture_intermediates` mechanism on the same image — maximum absolute scale-weight difference **4.415 × 10⁻⁷**.

---

## 2. Step 3 findings used

Step 3 was completed and its report read in full (`..\step3_analysis\step3_report.txt`) before any Step 4 experiment was run. Its measured findings were transcribed verbatim into `step4_hypotheses.json` as `step3_findings` F1–F12. Those carried into Step 4:

- **F2/F3/F4/F6** — the top 20 nearest training images were 20/20 BacterialSpot (cosine 0.962764–0.980528); the BacterialSpot class centroid sat at +0.936431 against +0.444106 for Healthy; the best Healthy neighbour (0.915473) was below the best BacterialSpot neighbour (0.980528); k-NN was 100 % BacterialSpot for k = 1 … 100.
- **F5** — the BacterialSpot similarity distribution against the query (median 0.895694, p95 0.956668) sits far above the Healthy one (median 0.410659, p95 0.520565, max 0.915473).
- **F7** — the stage-3 weight of 0.936823 is at the **49.87th percentile** of the training distribution; 50.13 % of training images are at or above it, and the Healthy (0.9803) and BacterialSpot (0.9776) class medians are both **higher**. Step 3 had already contradicted a stage-3-domination explanation.
- **F8** — stage-3 attention statistics placed the query nearer the BacterialSpot group than the Healthy group (channel-attention mean 0.49540 against 0.49890 and 0.52097), with a caveat of n = 6 representatives per class.
- **F9/F10/F11** — the image is an extreme outlier in measurable pixel statistics against **both** class distributions, every training image is 256 × 256 while this one is 1920 × 1440, and the production transform shows the network only 57.9 % of the frame.

Step 3's own conclusions that were **not** carried forward as explanations, because Step 3 had not tested them: everything about input framing, background content, and scale. Those are the Step 4 contributions.

No additional explanation was invented in Step 4. The eight hypotheses tested (`H1`–`H8` in `step4_hypotheses.json`) map one-to-one onto Step 3 findings F2–F12 plus the Step 2 observation about `Irrelavant`.

---

## 3. Experiments performed

| Task | What was done | Artefact |
|---|---|---|
| 1 | Re-ran the exact production path; captured feature A, feature B, all three projected stage features, channel attention, spatial attention, IW gates, adaptive scale weights, all 7 logits and probabilities via read-only hooks | `variant_predictions.csv` row `A` |
| 2 | Built 10 new analysis-only variants alongside the unmodified original. Each is a new lossless PNG; the original was read in place and never re-encoded or overwritten | `variants\*.png` |
| 3 | Recorded for every variant: class, confidence, all 7 probabilities, 128-d and 64-d features, scale weights, IW gates, logits — plus agreement, probability deltas, feature cosine similarity to the original, scale-weight delta and logit delta | `variant_predictions.csv` (11 rows × 75 columns) |
| 4 | Compared the background-controlled family (original / centre / tight / wide / leaf-centred / background-reduced) | `variant_predictions.csv` |
| 5 | Retrieved top-20 overall, top-20 Healthy and top-20 BacterialSpot from the 5 929-image Step 3 gallery, same feature and same cosine metric; mean / median / max / min and class composition per group | `feature_and_class_analysis.json` → `task5` |
| 6 | Computed per-class centroid, intra-class spread, full 7 × 7 inter-class centroid matrix, query-to-centroid ranking, and gallery 1-NN self-retrieval | `feature_separability.json`, `feature_separability.csv` |
| 7 | Reported all 7 logits, all 7 probabilities, top-1 / top-2 logit and margin | `feature_and_class_analysis.json` → `task7` |
| 8 | Recorded stage weights, IW gates and attention summaries for all 11 variants; additionally measured whether stage-1 attention varies at all | `variant_predictions.csv`, `attention_anomaly_stage1.json` |
| 9 | Measured 15 nearest genuine Healthy and 15 nearest genuine BacterialSpot training members on the identical path and compared 21 internal quantities | `feature_and_class_analysis.json` → `task9*` |
| 10 | Measured all 17 external images (unlabelled) for confidence, centroid distances, scale weights and rejection score | `feature_and_class_analysis.json` → `task10*` |
| 11 | Measured the `Irrelavant` probability on a 1 050-image stratified training sample, on all 11 variants and on all 17 external images | `task11_rejection.json` |
| 12 | Classified the failure into exactly one primary category | this report, §11 |
| 14 | Verified checkpoint, both datasets, all 17 external images, manifest, predictions, architecture, preprocessing, class list and inference behaviour | §13 and the verification log |
| 15 | Ran the complete test suite | §13 |

**Variants used (Task 2).** `retained` is the fraction of the original frame's pixel area that reaches the 224 × 224 network input, computed analytically by mapping the production crop box back through each variant's geometry.

| id | name | construction | size | retained |
|---|---|---|---|---|
| A | original | the true original file, read in place | 1920 × 1440 | 57.87 % |
| B | centre_square | centre 1:1 square crop (1440 × 1440) | 1440 × 1440 | 57.87 % |
| C | tight_0.60 | centre 60 % linear crop — removes surrounding background, ~1.67× zoom | 1152 × 864 | 20.83 % |
| D | wide_squarepad | padded to 1920 × 2400 by **edge-replicating** rows above and below (no invented content), so the production crop keeps far more of the original frame | 1920 × 2400 | 61.73 % |
| E | resize_only | downscaled 50 % with aspect preserved, then the unchanged production transform | 960 × 720 | 57.87 % |
| F | background_reduced | fixed 14 % border trimmed from each side | 1383 × 1037 | 30.01 % |
| G | leaf_centered | green-dominant mask bounding box + 10 % margin, largest centred square clipped to the frame | 1440 × 1440 | 57.87 % |
| H | hflip | horizontal flip | 1920 × 1440 | 57.87 % |
| I | vflip | vertical flip | 1920 × 1440 | 57.87 % |
| J | rot90 | 90° rotation | 1920 × 1440 | 50.90 % |
| K | rot180 | 180° rotation | 1920 × 1440 | 57.87 % |

Two geometry facts that fall out and are worth stating, because they shape how the rest of the report reads:

- **B is geometrically equivalent to A.** The production transform is `Resize(short side → 255)` then `CenterCrop(224)`, so it always extracts a centred square of side `224/255 = 87.84 %` of the short side. For a 4:3 source that square covers 1264.9 × 1264.9 px; cropping to a centred 1:1 square first yields the same window. B is therefore an **invariance control**, not an independent test.
- **G collapsed onto B.** The green-dominant mask spans essentially the whole frame, so "leaf-centred" and "centre square" produced the identical crop. G therefore adds no information beyond B. This is a measured property of the image — the leaf fills the frame — not a cropping failure.

---

## 4. Controlled variant results

MEASURED FACT. All eleven variants, unchanged production preprocessing and unchanged forward path.

| id | name | retained | agreement | predicted | confidence | cos 128-d | cos 64-d | ΔP(BacterialSpot) | ΔP(Healthy) | max Δscale-w | max Δscale-w stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | original | 57.87 % | — | BacterialSpot | 0.878758 | 1.000000 | 1.000000 | — | — | — | — |
| B | centre_square | 57.87 % | same | BacterialSpot | 0.865333 | 0.997865 | 0.998074 | −0.013425 | +0.003808 | 0.008892 | 0.0079 |
| C | tight_0.60 | 20.83 % | **CHANGED** | **Irrelavant** | 0.530499 | 0.796259 | 0.750432 | **−0.586649** | +0.005427 | **0.464389** | stage 3 → 0.472434 |
| D | wide_squarepad | 61.73 % | **CHANGED** | **Healthy** | 0.529331 | 0.679159 | 0.682600 | **−0.654622** | **+0.494285** | 0.082864 | stage 3 → 0.853959 |
| E | resize_only | 57.87 % | same | BacterialSpot | 0.883231 | 0.999814 | 0.999874 | +0.004473 | −0.001732 | 0.005336 | 0.0053 |
| F | background_reduced | 30.01 % | same | BacterialSpot | 0.684350 | 0.928338 | 0.914773 | **−0.194407** | +0.015031 | **0.273592** | stage 3 → 0.663231 |
| G | leaf_centered | 57.87 % | same | BacterialSpot | 0.865333 | 0.997865 | 0.998074 | −0.013425 | +0.003808 | 0.008892 | 0.0079 |
| H | hflip | 57.87 % | same | BacterialSpot | 0.882025 | 0.975271 | 0.991680 | +0.003267 | +0.007695 | 0.041175 | 0.0412 |
| I | vflip | 57.87 % | same | BacterialSpot | 0.775568 | 0.970727 | 0.966023 | −0.103190 | +0.055312 | 0.045993 | 0.0460 |
| J | rot90 | 50.90 % | same | BacterialSpot | 0.922789 | 0.950013 | 0.976735 | +0.044031 | −0.014784 | 0.022696 | 0.0194 |
| K | rot180 | 57.87 % | same | BacterialSpot | 0.891985 | 0.969299 | 0.989572 | +0.013228 | +0.000371 | 0.023680 | 0.0237 |

**Task 3 headline numbers.** Agreement with the original: **9 of 11 same, 2 changed**. Probability range across variants: BacterialSpot 0.224136 – 0.883231, a spread of **65.9 pp**. 64-d feature cosine to the original: **0.682600 – 0.999874**. Logit L2 delta: 0.056075 – 3.814358. Max absolute scale-weight delta: 0.005336 – 0.464389.

**What the pure-geometry control shows (MEASURED FACT).** The four transformations that preserve the same window — H hflip, I vflip, J rot90, K rot180 — all returned BacterialSpot, with 64-d cosines of 0.991680, 0.966023, 0.976735, 0.989572 and BacterialSpot probabilities of 0.882025, 0.775568, 0.922789, 0.891985. E, a 50 % downscale with aspect preserved, returned 0.883231 at 64-d cosine 0.999874.

**POSSIBLE EXPLANATION (not established):** the model appears robust to orientation, mirroring and input resolution, and unstable to *which window of the frame* is mapped to the 224 × 224 input. The measurements do not identify a mechanism for that asymmetry.

**Task 4 — is the error background-driven?** The task explicitly forbids the claim "background caused the error" unless the measurements support it. They support a weaker and more specific statement:

- The background-reduced variant **F** moved the BacterialSpot probability by **−19.44 pp** (0.878758 → 0.684350) and dropped the 64-d cosine to 0.914773, while the top-1 class stayed BacterialSpot.
- The tight variant **C** (20.83 % retained) changed the top-1 to Irrelavant; the wider variant **D** (61.73 % retained) changed it to Healthy; **D** raised P(Healthy) by **+49.43 pp**.
- The centre-square and leaf-centred variants, which change nothing geometrically, moved the probability by only −1.34 pp.

**A methodological limit that must be stated rather than glossed over.** Because production preprocessing ends in `CenterCrop((224,224))`, the network *always* receives a 224 × 224 square. Any change in the amount of frame retained is therefore simultaneously a change in magnification. With a centre-crop family it is **not possible to separate "background removed" from "scale changed"**. F is background reduction *and* a 0.72× linear reduction relative to B. The honest statement is:

> The quantity that was varied is **which sub-window of the image is mapped onto the fixed 224 × 224 input**, and the model's answer changes with that choice by up to 65.9 pp.

That is a real, reproducible framing sensitivity. It is not, by itself, evidence about background content.

---

## 5. Feature-space results

Feature B (64-d bottleneck, `head[8]` BatchNorm1d output), cosine similarity, identical metric and identical extraction path to Step 3. Training class labels come from dataset directory names only; model output was never used as a label. **Nearest-neighbour class is evidence about the learned representation, not ground truth.**

### Task 5 — neighbour groups

| group | n | mean | median | max | min | composition |
|---|---|---|---|---|---|---|
| top 20 overall | 20 | **0.968469** | 0.967973 | 0.980528 | 0.962764 | BacterialSpot 20 / 20 |
| top 20 Healthy | 20 | 0.626776 | 0.591747 | 0.915473 | 0.556217 | Healthy 20 / 20 |
| top 20 BacterialSpot | 20 | **0.968469** | 0.967973 | 0.980528 | 0.962764 | BacterialSpot 20 / 20 |

- gap, best BacterialSpot − best Healthy = **+0.065055**
- gap, mean(top 20 BacterialSpot) − mean(top 20 Healthy) = **+0.341693**

The query is closer to BacterialSpot training examples than to Healthy ones on **every one of these statistics**. The two groups do not overlap: the lowest of the top 20 BacterialSpot (0.962764) is above the highest of the top 20 Healthy (0.915473).

### Task 6 — class separability of the 64-d bottleneck

| class | n | intra-class mean Euclid to centroid | intra-class mean cos to centroid | spread (sd) | cos(query → centroid) | query rank |
|---|---|---|---|---|---|---|
| BacterialSpot | 700 | 0.2879 | 0.949010 | 0.0554 | **+0.936431** | **1** |
| Irrelavant | 933 | 0.0726 | 0.992826 | 0.0362 | +0.447695 | 2 |
| Healthy | 831 | 0.3430 | 0.931054 | 0.0598 | +0.444106 | 3 |
| Black_Rot | 900 | 0.2326 | 0.967106 | 0.0442 | +0.329726 | 4 |
| DownyMildew | 853 | 0.3991 | 0.898000 | 0.1180 | +0.320779 | 5 |
| PowderyMildew | 836 | 0.3710 | 0.915891 | 0.0919 | +0.220901 | 6 |
| Esca | 876 | 0.1006 | 0.988334 | 0.0492 | +0.220330 | 7 |

Inter-class centroid cosine: off-diagonal mean **0.376486**, off-diagonal maximum **0.509571** (`Black_Rot` vs `Esca`). The `BacterialSpot` ↔ `Healthy` centroid similarity is **0.356527** — one of the *more* separated pairs in the matrix, not a confusable one.

Nearest class BacterialSpot at +0.936431; runner-up Irrelavant at +0.447695; **margin +0.488736**.

**Key comparison (MEASURED FACT).** The query's cosine to the BacterialSpot centroid is **0.936431**, against a mean intra-class cosine-to-centroid of **0.949010** for genuine BacterialSpot members. The query sits *inside* the BacterialSpot cluster, marginally outside its typical core — not in no-man's land.

**Gallery 1-NN self-retrieval (5 929 images, 64-d cosine, self excluded):** overall agreement **0.9649** (5 721 / 5 929). BacterialSpot **0.9986**, Healthy 0.9759, Black_Rot 0.9811, DownyMildew 0.9109, Esca 0.9772, Irrelavant 0.9914, PowderyMildew 0.9211.

**Region occupancy (MEASURED FACT).** All **20 of 20** of the query's nearest images have a BacterialSpot image as their *own* nearest neighbour. The region the query occupies is densely populated by real BacterialSpot training images that are themselves robustly clustered.

### Task 10 — the 16 UNKNOWN images (unlabelled diagnostics only)

No ground truth was assigned. **None of these predictions is called correct or incorrect.**

| | query `grape leaf.jpg` | other 16 |
|---|---|---|
| cosine to BacterialSpot centroid | **0.936431** | max 0.588140, median 0.366940 |
| cosine to Healthy centroid | 0.444106 | max 0.908193, median 0.733853 |
| confidence | 0.8788 | mean 0.5899, max 0.8427 |
| stage-3 weight | 0.9368 | range 0.6113 – 0.9937 |

Nearest-centroid class across all 17: Healthy 9, Black_Rot 4, **BacterialSpot 1 (the query, and only it)**, DownyMildew 1, PowderyMildew 1, Esca 1.

The confirmed failure is the only one of the 17 whose representation lands near the BacterialSpot centroid, and the most confident of the 17. **This is a behavioural observation about where the representations sit. It is not evidence that the other 16 are correctly classified** — that cannot be known, and no claim of that kind is made.

---

## 6. Classifier results

MEASURED FACT, final 64 → 7 linear layer on the original:

| rank | class | logit | probability |
|---|---|---|---|
| 1 | BacterialSpot | **+3.395030** | 0.878758 |
| 2 | Healthy | +0.173183 | 0.035046 |
| 3 | Irrelavant | +0.031370 | 0.030412 |
| 4 | Black_Rot | −0.168717 | 0.024897 |
| 5 | DownyMildew | −0.695957 | 0.014695 |
| 6 | PowderyMildew | −1.062353 | 0.010187 |
| 7 | Esca | −1.591018 | 0.006004 |

- top-1 logit +3.395030, top-2 logit +0.173183, **margin +3.221847**, spread 4.98605.
- **No class is in competition with BacterialSpot.** The runner-up is at 0.035046, and the gap in probability is 84.4 pp. `Healthy` is the runner-up but trails by 0.8437 in probability and by 3.2218 logits.
- Softmax is monotone in the logit, so a margin of 3.22 logits means the 64-d feature sits far from the runner-up decision direction. This is **not** the profile of a thin margin crossed at a boundary.

**Which classes compete:** in the only sense the measurement supports, the ordering after BacterialSpot is `Healthy` 0.035046, `Irrelavant` 0.030412, `Black_Rot` 0.024897. All three are within 0.025 of each other and all are 3+ logits below the top-1. **No semantic similarity between BacterialSpot and any of them is asserted**; the logits do not support such a claim.

**Task 9 comparison against genuine class members (MEASURED FACT).** 15 nearest Healthy training members (cos 0.5769 – 0.9155) and 15 nearest BacterialSpot training members (cos 0.9663 – 0.9805), measured on the identical path:

| | own top-1 | mean P(Healthy) | mean P(BacterialSpot) | mean logit Healthy | mean logit BacterialSpot |
|---|---|---|---|---|---|
| 15 nearest genuine Healthy | Healthy 13 / 15 | **0.663552** | 0.192314 | **+2.598468** | +1.046587 |
| 15 nearest genuine BacterialSpot | BacterialSpot 15 / 15 | 0.031142 | **0.869697** | −0.121636 | **+3.228722** |
| **query `grape leaf.jpg`** | BacterialSpot | **0.035046** | **0.878758** | **+0.173183** | **+3.395030** |

The query's logit profile is nearly the genuine BacterialSpot profile: BacterialSpot logit +3.395 against the group's +3.229 (Δ 0.166), and Healthy logit +0.173 against the group's −0.122 (Δ 0.295). Against the genuine Healthy group the BacterialSpot logit is 2.35 logits away.

**Tally over 21 internal quantities** (3 scale weights, IW gate 3, 3 stage-3 attention statistics, 7 logits, 7 probabilities): the query's value is nearer the genuine **BacterialSpot** group for **19** and nearer the genuine **Healthy** group for **2** (IW gate 3, stage-3 channel-attention max). Restricted to the 14 class-discriminative quantities (7 logits + 7 probabilities): **14 nearer BacterialSpot, 0 nearer Healthy**.

Note that 2 of the 15 nearest genuine Healthy members are themselves predicted BacterialSpot, and one of them reaches 0.5063 on the `Irrelavant` probability (see §8). The neighbourhood is not perfectly clean; the model's own labelling of training data is not treated as an oracle.

---

## 7. Attention / multi-scale results

MEASURED FACT.

### Adaptive scale weights across the variants

| id | stage 1 | stage 2 | stage 3 | IW 1 | IW 2 | IW 3 |
|---|---|---|---|---|---|---|
| A original | 0.057653 | 0.005524 | 0.936823 | 0.014716 | 0.091832 | 0.410864 |
| B centre_square | 0.066545 | 0.004499 | 0.928957 | 0.014615 | 0.093807 | 0.394618 |
| **C tight_0.60** | **0.516605** | 0.010960 | **0.472434** | 0.019448 | 0.111263 | 0.341918 |
| **D wide_squarepad** | 0.137937 | 0.008103 | **0.853959** | 0.013526 | 0.119188 | 0.497662 |
| E resize_only | 0.052317 | 0.005524 | 0.942158 | 0.014696 | 0.093242 | 0.408228 |
| F background_reduced | 0.330168 | 0.006601 | 0.663231 | 0.017366 | 0.097317 | 0.439072 |
| G leaf_centered | 0.066545 | 0.004499 | 0.928957 | 0.014615 | 0.093807 | 0.394618 |
| H hflip | 0.018075 | 0.003927 | 0.977998 | 0.015302 | 0.097324 | 0.431795 |
| I vflip | 0.103646 | 0.003317 | 0.893036 | 0.015570 | 0.110908 | 0.389650 |
| J rot90 | 0.034957 | 0.008844 | 0.956199 | 0.013340 | 0.098514 | 0.449746 |
| K rot180 | 0.035874 | 0.003623 | 0.960503 | 0.016329 | 0.106671 | 0.444209 |

**Directly answering the task's question: no, the model does not consistently rely heavily on Stage 3 across the variants.** The stage-3 weight ranges **0.472434 (C) to 0.977998 (H)** — a 0.51 spread, far wider than its 49.87th-percentile placement in the corpus would suggest for a fixed behaviour. The two variants that changed the class sit at opposite ends: C at 0.472434 and D at 0.853959. **Stage-3 weight does not track the class change.**

IW gate 3 varies over 0.341918 – 0.497662; it does not separate the flipping variants from the stable ones either.

Stage-3 attention summaries across variants: channel-attention mean 0.478014 – 0.504229, channel-attention max 0.750114 – 0.993826, spatial-attention mean 0.362680 – 0.534463. The query's stage-3 channel-attention max of 0.823390 sits inside the range spanned by the 15 nearest genuine BacterialSpot members (mean 0.717036, minimum 0.62222) and below the mean of the 15 nearest genuine Healthy members (0.842841) — reproducing Step 3's finding F8 on a 15-per-class sample rather than 6.

**These are descriptive statistics of attention tensors. Nothing here is described as detecting a lesion.**

### A separate measured anomaly: Stage-1 attention is inert in this checkpoint

A measurement-integrity check was triggered because the `s1_ca_max` column was bit-identical across all 11 variants. It is not a measurement error. Over 51 images (40 sampled training images + the query + 10 variants):

- **Stage-1 channel attention is bit-identical for every image** — constant vector, mean 0.466480, max 0.570571, min 0.376848, 32 channels.
- The stage-1 `ca.fc` hidden ReLU layer had **zero active units on 51 of 51 images**; the pre-ReLU activation is negative for every tested input, so the hidden vector is identically zero and the output reduces to `sigmoid(ca.fc[2].bias)`.
- **Stage-1 spatial attention is saturated at unity** on 41 of 41 images (map mean 0.999828 – 0.999998); on the query its entropy is 8.050700 nats, which is exactly `ln(224 × 224) = ln(50176)`, i.e. a uniform map — the multiplicative identity.
- Stage 2 and stage 3 channel attention had an active hidden unit on **51 of 51** images, and their spatial maps vary (stage-2 mean 0.790546 – 0.864869; stage-3 mean 0.177060 – 0.612227).

Full record: `attention_anomaly_stage1.json`.

**Consequence for this section:** stage-1 channel and spatial attention statistics cannot discriminate between images, because they do not vary. Any per-image statement about them would be meaningless. This does **not** make stage 1 information-free — the stage-1 adaptive weight varies from 0.013326 to 0.516605 across the variants, and the image-wise gate and the backbone stage-1 feature remain input-dependent. **This anomaly is not offered as the cause of the confirmed failure.**

---

## 8. Rejection-class results

MEASURED FACT. The `Irrelavant` probability, from the unchanged production forward pass. No threshold and no rejection rule was inspected, changed or applied.

### Training-set reference distribution (1 050-image stratified sample, 150 per class)

| class | n | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|---|
| BacterialSpot | 150 | 0.0276 | 0.0245 | 0.0431 | 0.0799 | 0.0847 |
| Black_Rot | 150 | 0.0234 | 0.0181 | 0.0336 | 0.0961 | 0.2942 |
| DownyMildew | 150 | 0.0165 | 0.0130 | 0.0301 | 0.0616 | 0.1359 |
| Esca | 150 | 0.0228 | 0.0225 | 0.0254 | 0.0292 | 0.0325 |
| Healthy | 150 | 0.0238 | 0.0165 | 0.0306 | 0.2097 | 0.5063 |
| **Irrelavant** | 150 | **0.9048** | 0.9052 | 0.9118 | 0.9180 | 0.9201 |
| PowderyMildew | 150 | 0.0186 | 0.0152 | 0.0267 | 0.1137 | 0.1567 |

The distribution is strongly **bimodal by design**: genuine `Irrelavant` members cluster at ≈ 0.905; every other class sits at p99 **0.0836**, with an overall maximum of **0.5063** (a Healthy image). 151 of 1 050 (14.4 %) have `Irrelavant` as top-1 — i.e. the 150 genuine members plus one.

### The same image under controlled framing

| id | variant | P(Irrelavant) | stage-3 weight | top-1 |
|---|---|---|---|---|
| A | original | 0.030412 | 0.936823 | BacterialSpot |
| B | centre_square | 0.034932 | 0.928957 | BacterialSpot |
| **C** | **tight_0.60** | **0.530499** | 0.472434 | **Irrelavant** |
| D | wide_squarepad | 0.133051 | 0.853959 | Healthy |
| E | resize_only | 0.028691 | 0.942158 | BacterialSpot |
| F | background_reduced | 0.147837 | 0.663231 | BacterialSpot |
| G | leaf_centered | 0.034932 | 0.928957 | BacterialSpot |
| H | hflip | 0.021944 | 0.977998 | BacterialSpot |
| I | vflip | 0.065087 | 0.893036 | BacterialSpot |
| J | rot90 | 0.016770 | 0.956199 | BacterialSpot |
| K | rot180 | 0.024204 | 0.960503 | BacterialSpot |

### What this shows (MEASURED FACT)

- `Irrelavant` was top-1 for **0 of the 17** external images, yet for **1 of the 11** controlled variants of the same image — and it reached **0.530499**, above the maximum (0.5063) observed across all 900 sampled non-`Irrelavant` training images.
- Its probability on `grape leaf.jpg` ranges **0.016770 to 0.530499**, a **31.6-fold spread**, driven only by crop and geometry.
- Across the 17 external images the mean is 0.068834.

**Conclusion from measurement.** The class behaves as a **visual-content class whose probability is strongly coupled to input framing**, not as a stable out-of-distribution or rejection signal. A detector that fires on 0 of 17 out-of-distribution photographs but reaches 0.53 on a crop of one of those same photographs is not behaving as an OOD detector. This is a statement about the measured coupling only; the intent behind the class is not established, and the rejection logic was not touched.

**Consequence carried into §11:** the `Irrelavant` activation under variant C is a symptom of framing sensitivity, not a separate root cause. It is reported as such.

---

## 9. Evidence supporting each hypothesis

| hypothesis | Step 4 measured evidence **for** |
|---|---|
| **H1 — A. Feature-space class overlap** | Top-20 overall neighbours are **20/20 BacterialSpot** at mean cosine **0.968469**; the lowest of them (0.962764) exceeds the best of the top 20 Healthy (0.915473). Centroid cosine **+0.936431 vs +0.444106**, margin **+0.488736**. The query's 0.936431 is comparable to the genuine BacterialSpot intra-class mean of **0.949010** — it is inside the cluster. BacterialSpot 1-NN self-retrieval is **0.9986** and all 20 nearest neighbours have a BacterialSpot image as their own nearest. The `BacterialSpot`↔`Healthy` centroid similarity is only **0.356527**, so the query is not between the two classes. Query BacterialSpot logit **+3.395** against the genuine BacterialSpot group's **+3.229**. **14 of 14** class-discriminative internal quantities nearer the BacterialSpot group. Survives hflip / vflip / rot90 / rot180 (64-d cosine 0.966 – 0.992) and a 50 % downscale (0.999874) with the class unchanged. |
| **H2 — B. Dataset / domain shift** | The image is an extreme pixel-statistic outlier against **both** class distributions (Step 3 F9: meanSat +4.16 IQR vs Healthy, darkFrac +11.39 IQR vs BacterialSpot). Every training image is 256 × 256; this one is 1920 × 1440. The production transform shows the network only 57.87 % of the frame. It is the only one of the 17 external images nearest to the BacterialSpot centroid, and the most confident of the 17. |
| **H3 — C. Background / composition sensitivity** | Background reduction **F** moved P(BacterialSpot) by **−19.44 pp** and 64-d cosine to 0.914773. Tight crop **C** and wider **D** changed the top-1 class outright. Across all 11 variants P(BacterialSpot) spans 0.224136 – 0.883231 and 64-d cosine spans 0.682600 – 0.999874. |
| **H4 — D. Multi-scale / attention dependence** | The query's stage-3 channel-attention max (0.823390) and mean (0.495402) sit with the genuine BacterialSpot group (0.717036 / 0.497701) rather than the Healthy group (0.842841 / 0.502915). The query's IW gate 3 (0.410864) exceeds both group means. The stage-1 attention modules are measurably inert (§7), so stage 1 contributes no attention signal. |
| **H5 — E. Preprocessing sensitivity** | The production transform discards 42 % of the frame width. Two of 11 controlled variants changed the top-1 class; probability swings reached 65.9 pp. The B control shows a 1.34 pp shift from the resampling path alone at an identical retained window. |
| **H6 — F. Classifier-boundary error** | The top-1/top-2 margin is **+3.221847** logits and the runner-up probability is 0.035046 against 0.878758. |
| **H7 — Rejection-class behaviour** | `Irrelavant` is second-closest to the query at +0.447695, only 0.003589 above `Healthy`. It fires on 0 of 17 external frames but reaches 0.530499 on a crop of one of them, exceeding the 0.5063 maximum over 900 non-`Irrelavant` training images. |
| **H8 — Class specificity** | The query is the only one of 17 external images nearest to the BacterialSpot centroid, while 9 of the other 16 are nearest to the `Healthy` centroid. The BacterialSpot neighbourhood is densely occupied (1-NN self-retrieval 0.9986). |

---

## 10. Evidence against each hypothesis

| hypothesis | Step 4 measured evidence **against** |
|---|---|
| **H1** | The query is an extreme outlier in pixel statistics against the **BacterialSpot** distribution too (darkFrac +11.39 IQR, sharpness +3.08 IQR, Step 3 F9), and at 0.936431 it sits just outside the typical BacterialSpot core of 0.949010. So it is not sitting in the densest part of a well-behaved cluster. **The measurements locate the representation; they do not explain how it got there.** |
| **H2** | The representation is **not** in no-man's land. It lies 0.936431 from the BacterialSpot centroid against 0.444106 from `Healthy`, and within 0.013 of the genuine BacterialSpot intra-class mean. A simple out-of-distribution account would predict a diffuse or arbitrary position, not a well-formed position inside a specific, densely populated class cluster. Stage-2 and stage-3 attention behave normally. |
| **H3** | The variants that change nothing geometrically (B, E, G) stay BacterialSpot with 64-d cosine ≥ 0.998074; flips and rotations stay BacterialSpot with cosine ≥ 0.966023. Only **F**, C and D move substantially. **And "background" cannot be isolated**: because the transform ends in `CenterCrop(224)`, every centre-crop change is simultaneously a magnification change, so this experiment does not test background content as such. |
| **H4** | The query's stage-3 weight of 0.936823 is at the **49.87th percentile** of 5 929 training images and is **below** the Healthy median (0.9803) and the BacterialSpot median (0.9776). Across the variants it ranges 0.472434 – 0.977998 and **does not track the class change** — the Irrelavant flip sits at 0.472434 while the Healthy flip sits at 0.853959, and the four stable BacterialSpot variants span 0.893036 – 0.977998. All 12 genuine class representatives in Step 3 also carried stage-3 weights of 0.864 – 0.998. Stage-3 dominance is shared and cannot discriminate. |
| **H5** | Pure geometry and pure resolution changes do **not** alter the class: hflip, vflip, rot90, rot180 and a 50 % downscale all return BacterialSpot with 64-d cosine ≥ 0.966023, most above 0.975. The sensitivity is specific to *which window* is shown, not to arbitrary perturbation of the input. It is a real but bounded effect, and it does not change which class the model favours in 9 of 11 variants. |
| **H6** | A margin of 3.221847 logits with a runner-up at 0.035046 is not a near-tie. The 64-d representation is unambiguously on the BacterialSpot side: 0.936431 versus 0.444106 to the `Healthy` centroid, and 14 of 14 discriminative quantities nearer the genuine BacterialSpot group. The classifier is faithfully reporting the representation it is given. |
| **H7** | Nothing in Step 3 or Step 4 measured the *intent* of the class. The measurements establish that its probability is framing-coupled; whether it was ever designed as an OOD class is not determined, and the rejection logic was not inspected or modified. |
| **H8** | The framing experiment shows the model is **not** generally fragile on this image: 9 of 11 variants, including all four rotations/flips and a rescale, return BacterialSpot. The failure is specific to this image, but the model's framing sensitivity is a general property that Step 4 has now demonstrated. |

---

## 11. Primary root-cause classification

# **A. Feature-space class overlap**

> The Healthy image lies substantially closer to BacterialSpot representations.

**Exactly one category is chosen. Secondary contributing observations are listed separately below and do not change the primary category.**

### Why A, in five measured points

1. **Neighbourhood.** 20 of the 20 nearest training images are BacterialSpot at mean cosine 0.968469; the weakest of them (0.962764) outranks the strongest of the top 20 Healthy (0.915473). The mean gap between the two groups is **+0.341693**.
2. **Centroid position.** +0.936431 to the BacterialSpot centroid against +0.444106 to `Healthy`, a margin of **+0.488736** — while the BacterialSpot↔Healthy centroid similarity is only 0.356527, so the query is not between the two classes. The query's 0.936431 is within 0.013 of the genuine BacterialSpot intra-class mean of 0.949010.
3. **Region is real, not an artefact.** BacterialSpot 1-NN self-retrieval is 0.9986 over 700 images, and all 20 of the query's nearest neighbours have a BacterialSpot image as their own nearest neighbour. The query sits in a densely, consistently occupied BacterialSpot region.
4. **Classifier profile matches the class, not the boundary.** BacterialSpot logit +3.395030 against the 15 nearest genuine BacterialSpot members' +3.228722 (Δ 0.166); 14 of 14 class-discriminative internal quantities nearer the BacterialSpot group; top-1/top-2 margin +3.221847.
5. **It is stable under every content-preserving perturbation tested.** hflip, vflip, rot90, rot180 and a 50 % downscale all return BacterialSpot with 64-d cosine ≥ 0.966023. The representation does not drift toward `Healthy` when the content is merely re-presented.

The proximate fact is therefore unambiguous: **IWNET's 64-d representation of this image is a BacterialSpot representation, and the final linear layer reports that faithfully.** The question Step 4 was asked — reproduce, class-specific, feature-space related, preprocessing/crop related, attention/fusion related, or isolated — resolves to **feature-space related**, and specifically to overlap rather than to a boundary crossing.

### Why not the alternatives

- **Not F (classifier-boundary error).** Margin 3.22 logits, runner-up 0.035046. The classifier is not making a marginal call.
- **Not D (multi-scale/attention dependence).** The stage-3 weight is at the 49.87th percentile, below both class medians, and ranges 0.472 – 0.978 across variants **without tracking the class change**.
- **Not G (insufficient evidence).** For the *identity* of the prediction and for the primary category, the evidence is strong and multiply redundant. Section 12 separates this from the much weaker question of *why* the representation is where it is.
- **Not C/E as primary.** Framing sensitivity is real and material, but 9 of 11 variants — including all flips, rotations and the rescale — still return BacterialSpot, and the category as defined ("controlled cropping materially changes the classification") describes a genuine secondary effect rather than the reason the answer is BacterialSpot. It is recorded as secondary below.
- **Not B as primary.** Domain shift is measurable at the pixel level, but the network maps this image into a well-formed, densely populated class region rather than into an unrecognised one. Domain shift is a plausible contributor to *why* the representation is unusual, but it is not the proximate cause of the label.

### Secondary contributing observations (measured, not causal)

- **Framing / window-selection sensitivity is real and material.** Two of 11 variants changed the top-1 class; P(BacterialSpot) spans 0.224136 – 0.883231 across variants. Because production ends in `CenterCrop((224,224))`, the varied quantity is *which sub-window is mapped to the fixed 224 × 224 input*; "background" and "magnification" are not separable with a centre-crop family.
- **One variant reaches the correct class.** `D` (wider, 61.73 % retained) returns `Healthy` at 0.529331, and raises P(Healthy) by 49.43 pp. This is a measurement, not a proposed fix: D's padding is edge-replicated and is not a production-legal input.
- **Stage-1 attention is inert in this checkpoint** (constant channel attention, unity-saturated spatial map, 0/51 live hidden units). A measured training artefact, recorded in `attention_anomaly_stage1.json`. It is **not** offered as the cause.
- **`Irrelavant` is not a stable OOD signal** — 0/17 on external frames, 0.530499 on a crop of one of them, above the 0.5063 maximum over 900 non-`Irrelavant` training images.
- **The image is a pixel-statistic outlier against both classes** and is 7.5× larger than any training image, with zero size or aspect variance in the training set to measure it against.

---

## 12. Confidence in the classification

**Primary category A — high confidence.** Five independent measurement families agree and none is contradicted: neighbourhood (20/20, gap 0.341693), centroid (margin 0.488736, within 0.013 of the class's own intra-class mean), region occupancy (1-NN self-retrieval 0.9986, 20/20 own-neighbours), classifier profile (logit Δ 0.166 to the genuine class, margin 3.22), and stability under five content-preserving perturbations (64-d cosine ≥ 0.966023). The competing categories F and D are excluded by direct measurement, not by plausibility.

**Explicitly low confidence — why the representation is where it is.** Step 4 measured *where* the representation sits and *how stably* it sits there. It did not and cannot establish a visual cause. The measurements are consistent with several incompatible accounts — a domain-shift effect, a genuine visual similarity the ground truth does not reflect, a labelling artefact in the training data, or an interaction of these — and **nothing in Step 4 discriminates between them.** No claim of the form "the model classified it as bacterial spot because of <visual property>" is made, because no such link was measured.

**Moderate confidence on the secondary framing factor.** The framing sensitivity is reproducible (2/11 class flips, 65.9 pp swing) but the design cannot cleanly attribute it to background rather than magnification, and 9 of 11 variants retain the class.

**Moderate confidence on the stage-1 attention anomaly.** Verified on 51 images and 41 images respectively with two independent probes, but only 51 images were sampled and the checkpoint's training history was not inspected. It is reported as a property of the checkpoint, not as a diagnosis.

**Scope limit on all of the above.** One VERIFIED image, one run, CPU, deterministic inference, no sampling variance quantified. Nothing here generalises to the model as a whole.

---

## 13. What this does NOT prove

- **It does not prove why the image looks the way it does to the network.** The classification is explained; the representation is only located.
- **It does not prove the leaf has, or does not have, any visible disease.** The only ground truth used was the manifest's VERIFIED label. No visual inspection was performed and none is claimed.
- **It does not prove the model "saw a spot".** Every attention figure in this report is a descriptive statistic of a tensor. No statement is made about what any attention map detected.
- **It does not prove background caused the error.** The centre-crop family cannot separate background from magnification. §4 states the measured relationship instead.
- **It does not establish any accuracy figure for the 16 UNKNOWN images.** They have no ground truth and are never called correct or incorrect anywhere in this report.
- **It does not support a general accuracy claim from the 17-image external set.** That set is not a benchmark; 17 user-supplied photographs with 1 verified label cannot produce one.
- **It does not show the model is generally weak on external input.** 9 of the other 16 external images sit nearest the `Healthy` centroid, and their mean confidence is 0.5899. The confirmed failure is the *only* one nearest to BacterialSpot.
- **It does not show that the stage-1 attention collapse caused the failure**, or that it is harmful.
- **It does not show that retraining, a different architecture, or different data would fix anything.** No such experiment was run.
- **It does not justify replacing the model** on the basis of one failure.
- **It does not treat nearest-neighbour class as ground truth.** All neighbour class labels are dataset directory names, and two of the 15 nearest genuine Healthy members are themselves predicted BacterialSpot.
- **It does not quantify run-to-run variance.** All inference was deterministic, single-image, CPU.

---

## 14. Recommended Step 5

**Do not retrain and do not replace the model.** One VERIFIED image does not justify either, and the primary category (A) points at the representation and the training distribution rather than at the inference code.

Step 5 should proceed in this order:

**5.1 — Fix the measurement base before touching the model (highest priority).** Acquire a labelled set of real-world, full-frame grape-leaf photographs — several dozen, with verified labels — and extend `manifest.csv`. Every subsequent decision currently rests on **one** verified image. This is the single largest limitation in this report and no modelling change should precede it.

**5.2 — Characterise the framing sensitivity offline, without changing production.** The production transform discards 42 % of the frame width and always maps a centred 224 × 224 square. Run an offline comparison of candidate framings (pad-to-square / letterbox, resize-short-side-then-crop, multi-crop aggregation) over the labelled set from 5.1 and over `Balanced_From_Sources`. The question is whether the 65.9 pp variant swing is a property of this image or of the framing policy in general. **No production threshold, transform or inference path should be changed until this is measured on more than one image.**

**5.3 — Investigate the stage-1 attention collapse as a training defect.** Stage-1 channel attention is constant (51/51 images, hidden ReLU 0/51 live) and stage-1 spatial attention is unity-saturated (41/41). Corpus median stage-1 scale weight is 0.0067. Check the training logs for whether stage-1 gradients ever flowed and whether the `scale_w` softmax is starving stages 1 and 2. This is a training-quality question and it should be settled before any architecture discussion.

**5.4 — Settle the `Irrelavant` semantics.** It is bimodal by design (≈0.905 for genuine members, p99 0.0836 for everything else) yet framing-coupled: 0/17 on external frames, 0.530499 on a crop of one of them, above the 0.5063 maximum over 900 non-`Irrelavant` training images. Decide explicitly whether it is meant to be an OOD/rejection class. If it is, it is not currently functioning as one, and the `test/irrelavant` vs `train/Irrelavant` casing split carried forward from the static audit should be resolved at the same time. **Do not modify the rejection logic before that decision is made.**

**5.5 — Only then, consider data work.** The training set is 100 % 256 × 256 square tiles with zero size or aspect variance, while real inputs are high-resolution full frames. A controlled experiment on tiling and crop augmentation is better justified than an architecture change. Any such experiment must be evaluated on the labelled set from 5.1, and must not use the 17-image external set as a benchmark.

**Standing constraint for Step 5:** the external validation set is not a benchmark and must never be used to select a model. Model selection stays on `Balanced_From_Sources` splits.

---

**ANNEX (supporting evidence — not one of the 14 sections; full log in `verification_log.txt`)**

### A.1 Checkpoint, datasets, external images, prior artefacts — **PASS**

| item | result |
|---|---|
| `best_iwnet.pth` SHA-256 | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` — identical to the pre-analysis baseline, 41 213 897 bytes, mtime 2026-09-28 20:50:46 (predates this analysis) |
| `Balanced_From_Sources` | 5 947 files / 5 940 images, aggregate hash `b74ddb7e…` unchanged |
| `Balanced_Final_Split` | 7 200 files / 7 200 images, aggregate hash `5d54ca81…` unchanged |
| 17 external images | all 17 SHA-256 hashes identical; no file added, removed, resized or replaced; `grape leaf.jpg` still 429 238 bytes |
| `manifest.csv` | present, 17 rows, exactly 1 VERIFIED row (`grape leaf.jpg`, Healthy), all 17 SHA-256 values still match the images on disk |
| `predictions.csv` | present, 17 rows, 13 columns, all rows still sum to 1, `grape leaf.jpg` still BacterialSpot 0.878758 |
| `predictions_detail.json` | present, 17 entries, all 7 probabilities for `grape leaf.jpg` still match the Step 1/2 record |
| none of the three was overwritten by Step 4 | confirmed |

### A.2 Architecture, preprocessing, class list, inference behaviour — **PASS**

| item | result |
|---|---|
| parameter count | 10 161 307 — unchanged |
| backbone / head | `efficientnet_b3`, `out_indices=(1,2,3)`; head still 11 modules: `AdaptiveAvgPool2d │ Flatten │ Dropout │ Linear │ BatchNorm1d │ ReLU │ Dropout │ Linear │ BatchNorm1d │ ReLU │ Linear` |
| preprocessing | `Compose(Resize(size=255, interpolation=bilinear, …), CenterCrop(size=(224,224)), ToTensor, Normalize)` — unchanged |
| class list | `['BacterialSpot','Black_Rot','DownyMildew','Esca','Healthy','Irrelavant','PowderyMildew']` — unchanged, order preserved, `Irrelavant` spelling intact |
| live inference | `grape leaf.jpg` still predicts BacterialSpot at **0.878758**, `Healthy` 0.035046, all 7 probabilities matching the Step 1/2 record, sum 1.000000, `is_rejection` still `False` — thresholds unchanged |
| project source | no project `.py`/`.yaml`/`.tsx`/`.js`/`.css`/`.html`/`.txt` file was modified by Step 4 |

### A.3 Step 4 outputs — all inside `RealWorldValidation\step4_analysis\`

`step4_hypotheses.json` · `variant_predictions.csv` · `variant_measurements.json` · `feature_separability.json` · `feature_separability.csv` · `feature_and_class_analysis.json` · `task11_rejection.json` · `attention_anomaly_stage1.json` · `verification_log.txt` · `STEP4_ROOT_CAUSE_REPORT.md` · `variants\` (10 new PNG files) — 20 files, all confined to `step4_analysis\`, verified programmatically.

The original `grape leaf.jpg` was read in place and never re-encoded, resized, moved, renamed or copied over. No training image was added, removed or modified.

### A.4 Regression tests — **PASS**

| | count |
|---|---|
| existing tests (baseline) | 134 |
| new tests added by Step 4 | **0** — Step 4 added no tests and weakened or deleted none |
| total collected | 161 |
| passed | **161** |
| failed | 0 |
| errors | 0 |
| skipped | 0 |

---

*Step 4 was read-only throughout. No retraining, no fine-tuning, no modification of `best_iwnet.pth`, architecture, preprocessing, inference logic, class definitions, thresholds or either dataset. The production prediction remains BacterialSpot at 0.878758 and was not altered.*
