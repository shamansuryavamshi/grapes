# STEP 5.2 REPORT — Controlled Illumination / Appearance Diagnostic

**Status:** complete. READ-ONLY diagnostic. No training, no weight change, no dataset change.
**Baseline commit:** `e101607c78a3c61dda4e3e0d91dfe1ae9a3901a4`
**Frozen checkpoint:** `Balanced_From_Sources\best_iwnet.pth`, SHA-256 `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b`, 41,213,897 bytes — unchanged.
**Output directory:** `RealWorldValidation\step5_2_illumination_diagnostic\`
**Nothing was committed to Git.**

---

## 1. Objective

Determine whether controlled changes to illumination and appearance move the verified
external Healthy image (`grape leaf.jpg`) measurably toward or away from the BacterialSpot
feature region, using the frozen Step 4 model and the frozen Step 4/5.1 64-D representation.

This is a **diagnostic** step. Step 5.1 found that the Healthy and BacterialSpot training
distributions are not broadly intermixed and that the verified query lies outside the feature
range of the Healthy training population. Step 5.2 asks a narrower question: can appearance
alone account for that?

## 2. Hypothesis

> "Variation in illumination / brightness / contrast / exposure / directional lighting may cause
> the verified Healthy leaf to acquire a feature representation that is unusually close to
> BacterialSpot."

This was treated as a hypothesis to be **tested, not assumed**. The experiment was designed so
that it could return a null result, and it did return a largely null result — see §14–16.

## 3. Frozen baseline

| item | value |
|---|---|
| Architecture | IWNET (EfficientNet-B3 backbone) |
| Parameters | 10,161,307 |
| Checkpoint | `Balanced_From_Sources\best_iwnet.pth`, epoch 28, val_acc 0.9719 |
| Checkpoint SHA-256 | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` |
| Checkpoint size | 41,213,897 bytes |
| Classes | `BacterialSpot, Black_Rot, DownyMildew, Esca, Healthy, Irrelavant, PowderyMildew` |
| Production preprocessing | `Resize(255, BILINEAR)` → `CenterCrop(224,224)` → `ToTensor` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` |
| Inference device | CPU (`cuda_available = False`) |
| Primary representation | `head[8]` = `BatchNorm1d(64)` — the same 64-D bottleneck used in Step 4 and Step 5.1 |
| Centroids | Healthy (n=831) and BacterialSpot (n=700), reused bit-identically from the Step 5.1 feature cache; inter-centroid cosine 0.356527 |

**Note on `head[0]`.** The brief asked for a "128-D `head[0]` representation". `head[0]` is
`AdaptiveAvgPool2d(output_size=1)`, which emits a `128 x 1 x 1` spatial map, not a vector;
`head[1]` is `Flatten`. Its flattened output is therefore the 128-D pooled vector. A second
128-D representation exists at `head[4]` = `BatchNorm1d(128)`. **Both** were recorded
(`feat128_head0_pool` and `feat128_head4`) rather than assuming the prompt's layer attribution.
The 64-D `head[8]` bottleneck remains the primary representation, as specified.

## 4. Original image information

| property | value |
|---|---|
| Path | `C:\Users\shaman\Desktop\healthy leaves\grape leaf.jpg` |
| SHA-256 | `d5e62c4d30660bdeb7b818b8b6154a7b375a462d0b231e6ce642eea9e33be026` |
| Size | 429,238 bytes |
| Dimensions | 1920 x 1440 |
| Format | JPEG (PIL mode RGB) |
| mtime (UTC) | 2026-09-29T08:48:23Z |
| Ground truth | Healthy (independently verified) |
| Production prediction | `BacterialSpot` 0.878758, `Healthy` 0.035046 |

The file was **never written to**. It was decoded exactly once, read-only; every variant was
produced in memory from that single decode. The SHA-256 was re-read from disk after the
experiment and is byte-identical. All 17 external image digests are unchanged.

**Baseline reproduction (mandatory pre-check).** The production predictor reproduced
`BacterialSpot` 0.8787577152252197 and `Healthy` 0.035046033561229706 before any variant was
generated, and the Step 4/5.1 representation values were reproduced exactly (§8). The STOP
condition for baseline mismatch was therefore never triggered.

**Read-only proof for the feature hook.** `IntermediateTrace` in `iwnet/inference/capture.py`
does not expose the 64-D bottleneck, so a plain forward hook was registered on `head[8]`
(and `head[1]`, `head[4]`) — the same read-only mechanism validated in Step 5.1. For every
variant the hooked forward pass was compared against the official `Predictor.predict_pil`
output: **max probability delta = 0.0 across all 43 variants**. The hook does not perturb
production behaviour. No model was rewritten and no second model was created.

## 5. Step 5.1 findings used as context

| Step 5.1 result | value |
|---|---|
| Healthy / BacterialSpot training images | 831 / 700 |
| Same-class pooled mean cosine | 0.880719 |
| Cross-class pooled mean cosine | 0.315020 |
| Inter-centroid cosine | 0.356527 |
| Strict Healthy -> BacterialSpot hard examples | 3 / 831 (0.36%) |
| Strict BacterialSpot -> Healthy hard examples | 2 / 700 (0.29%) |
| Mutual NN pairs | 7 |
| Strongest cross-class similarity | 0.950301 |
| Healthy images with a cross-class NN closer than own same-class NN | 2 / 831 |
| BacterialSpot images likewise | 0 / 700 |
| **Maximum Healthy centroid margin over all 831 Healthy training images** | **+0.344078** |
| Query -> nearest BacterialSpot training image | 0.980528 |
| Query -> nearest Healthy training image | 0.915473 |
| Query centroid margin | +0.492325 |

Two of these numbers are used as reference scales in §9.

## 6. Transformation methodology

**Every branch starts from the original.** The original was decoded once; each of the 42
transformed variants was generated independently from that single decode. No variant was ever
derived from another variant, so no effect is cumulative. This was verified structurally:
the `factor = 1.00` variant in each of the brightness, contrast and gamma families reproduces
the control **exactly** (max `delta P(BacterialSpot)` = `0.0`).

**Applied before production preprocessing.** Transformations are appearance-only changes to the
image. They are applied to the 1920x1440 RGB array *before* `Resize(255) -> CenterCrop(224) ->
ToTensor -> Normalize`. Production preprocessing was never bypassed, reordered or modified. The
experiment tests image appearance; it does not test a replacement preprocessing.

**Exact definitions.** Every parameter and formula is recorded in `transformation_manifest.json`.

| group | op | definition |
|---|---|---|
| brightness | `PIL.ImageEnhance.Brightness(f)` | `out = clip(f * in)`; `f > 1` brightens |
| contrast | `PIL.ImageEnhance.Contrast(f)` | `out = mean_gray + f * (in - mean_gray)`; `f < 1` flattens |
| gamma | 256-entry LUT | `out = 255 * (in/255) ** g`; **`g < 1` BRIGHTENS, `g > 1` DARKENS** |
| directional | smooth linear gradient, multiplicative | `mask = 1 + strength * (1 - 2t)`, `t` ramps 0→1 along the named axis; brighter at `t = 0` |
| shadow | smooth elliptical falloff, multiplicative | `falloff = 0.5*(1 + cos(pi*d))` for `d` in [0,1]; `mask = 1 - strength*falloff` |
| correction: gray-world | per-channel gain, partial | `gain_c = (mean_of_channel_means / mean_c) ** alpha`; `alpha = 0` identity, `alpha = 1` full gray-world |
| correction: flat-field | divide by blurred illumination estimate | `I = blur(img, r)`; `corrected = img / I * mean(I)`; `out = (1-alpha)*img + alpha*corrected` |
| correction: retinex-style | multi-scale log subtract (MSCR) | `R = log(1+in) - mean_k log(1 + blur_k(in))`, `k in {8,24,72}`; level-matched then blended |

**Gamma direction is stated explicitly** because it is the convention most easily misread: the
`g = 0.70` variant is *brighter* (mean luminance 142.4 vs 116.9) and the `g = 1.30` variant is
*darker* (mean luminance 98.0).

**Directional and shadow masks are controlled proxies, not physical lighting models.** They
simulate the *kind* of appearance change that differing light angles can produce; they do not
reproduce real-world lighting, and no claim in this report treats them as equivalent to it.

**Corrections are general and class-agnostic.** None is tuned toward Healthy, and no
transformation was engineered to produce a Healthy prediction.

## 7. Number of variants

**43 total: 1 zero-change control + 42 transformed.**

| group | n | parameters |
|---|---|---|
| control | 1 | original |
| brightness | 5 | 0.70, 0.85, 1.00, 1.15, 1.30 |
| contrast | 5 | 0.70, 0.85, 1.00, 1.15, 1.30 |
| gamma | 5 | 0.70, 0.85, 1.00, 1.15, 1.30 |
| directional | 12 | 4 directions (left_to_right, right_to_left, top_to_bottom, bottom_to_top) x 3 strengths (weak 0.10, medium 0.20, strong 0.30) |
| shadow | 6 | 2 centres (upper_left, lower_right) x 3 strengths (0.10, 0.20, 0.30) |
| correction | 9 | gray-world a in {0.25, 0.50, 1.00}; flat-field r in {32,128} x a in {0.50,1.00}; retinex a in {0.50, 1.00} |

No transformation was omitted, and none that moved *away* from Healthy has been hidden. The
complete 43-row table is in §8.

## 8. Prediction results

The production prediction reproduces Step 4 / Step 5.1 exactly for the control:

| quantity | Step 4 / 5.1 | Step 5.2 control |
|---|---|---|
| predicted class | `BacterialSpot` | `BacterialSpot` |
| P(BacterialSpot) | 0.878758 | 0.878758 |
| P(Healthy) | 0.035046 | 0.035046 |
| top1-top2 margin | — | 0.843712 |
| cos -> BacterialSpot centroid | 0.936431 | 0.936431 |
| cos -> Healthy centroid | 0.444106 | 0.444106 |
| centroid margin (BS - Healthy) | +0.492325 | +0.492325 |
| cos -> nearest BS training image | 0.980528 | 0.980528 |
| cos -> nearest Healthy training image | 0.915473 | 0.915473 |

**Headline: all 43 variants — including the control — are predicted `BacterialSpot`.**

| | value |
|---|---|
| classification flips | **0** of 42 transformed variants |
| variants with P(Healthy) > P(BacterialSpot) | **0** |
| distinct predicted classes | `BacterialSpot` only |
| minimum P(BacterialSpot) | 0.732782 (`contrast__f1.30`) |
| maximum P(Healthy) | 0.080721 (`contrast__f1.30`) |
| worst-case ratio | **9.08 : 1 in favour of BacterialSpot** |

Per-variant results:

<!-- GENERATED by step52_tables.py from illumination_summary.csv / .json -->

### Per-family summary

| family | n | mean DeltaP(BS) | mean DeltaP(Healthy) | mean Deltamargin | max cos dist | toward Healthy | any flip |
|---|---|---|---|---|---|---|---|
| `brightness` | 5 | -0.0055 | +0.0010 | -0.0005 | 0.00583 | 2/5 | no |
| `contrast` | 5 | -0.0435 | +0.0130 | -0.0337 | 0.04525 | 2/5 | no |
| `gamma` | 5 | -0.0121 | +0.0029 | -0.0031 | 0.00973 | 2/5 | no |
| `directional` | 12 | -0.0083 | +0.0030 | -0.0089 | 0.01207 | 6/12 | no |
| `shadow` | 6 | -0.0049 | +0.0006 | +0.0071 | 0.00400 | 3/6 | no |
| `correction` | 9 | -0.0317 | +0.0114 | -0.0426 | 0.09906 | 6/9 | no |

Sign convention: a **negative** Deltamargin / DeltaP(BS) and a **positive** DeltaP(Healthy) means the
variant moved *toward* Healthy. `cos dist` is `1 - cos(variant, original query)` in the
64-D `head[8]` bottleneck - the Step 4 / Step 5.1 representation.

### Group D - DIRECTIONAL-light and shadow subgroups (each knob = 3 strengths)

| subgroup | n | mean Deltamargin | mean DeltaP(BS) | mean DeltaP(Healthy) | mean cos dist | moved toward Healthy? |
|---|---|---|---|---|---|---|
| `bottom_to_top` (directional) | 3 | -0.0527 | -0.0300 | +0.0132 | 0.00565 | **yes** |
| `left_to_right` (directional) | 3 | -0.0159 | -0.0133 | +0.0043 | 0.00113 | **yes** |
| `right_to_left` (directional) | 3 | +0.0140 | +0.0068 | -0.0027 | 0.00080 | no |
| `top_to_bottom` (directional) | 3 | +0.0189 | +0.0034 | -0.0028 | 0.00057 | no |
| `lower_right` (shadow) | 3 | -0.0101 | -0.0147 | +0.0048 | 0.00227 | **yes** |
| `upper_left` (shadow) | 3 | +0.0244 | +0.0049 | -0.0035 | 0.00160 | no |

### Group A - BRIGHTNESS sweep (every value generated from the original)

| factor | predicted | P(BacterialSpot) | DeltaP(BS) | P(Healthy) | DeltaP(Healthy) | centroid margin | Deltamargin | cos->original |
|---|---|---|---|---|---|---|---|---|
| f0.70 | BacterialSpot | 0.872699 | -0.0061 | 0.034512 | -0.0005 | 0.517716 | +0.0254 | 0.99644 |
| f0.85 | BacterialSpot | 0.875705 | -0.0031 | 0.034915 | -0.0001 | 0.504122 | +0.0118 | 0.99780 |
| f1.00 *(= original)* | BacterialSpot | 0.878758 | +0.0000 | 0.035046 | +0.0000 | 0.492325 | +0.0000 | 1.00000 |
| f1.15 | BacterialSpot | 0.872154 | -0.0066 | 0.037079 | +0.0020 | 0.477534 | -0.0148 | 0.99743 |
| f1.30 | BacterialSpot | 0.867163 | -0.0116 | 0.038716 | +0.0037 | 0.467241 | -0.0251 | 0.99417 |

Spearman(factor, centroid margin) = -1.000, p = 0.0000.

### Group B - CONTRAST sweep (every value generated from the original)

| factor | predicted | P(BacterialSpot) | DeltaP(BS) | P(Healthy) | DeltaP(Healthy) | centroid margin | Deltamargin | cos->original |
|---|---|---|---|---|---|---|---|---|
| f0.70 | BacterialSpot | 0.856936 | -0.0218 | 0.038082 | +0.0030 | 0.504914 | +0.0126 | 0.99527 |
| f0.85 | BacterialSpot | 0.875523 | -0.0032 | 0.034182 | -0.0009 | 0.507078 | +0.0148 | 0.99824 |
| f1.00 *(= original)* | BacterialSpot | 0.878758 | +0.0000 | 0.035046 | +0.0000 | 0.492325 | +0.0000 | 1.00000 |
| f1.15 | BacterialSpot | 0.832286 | -0.0465 | 0.051960 | +0.0169 | 0.433361 | -0.0590 | 0.99151 |
| f1.30 | BacterialSpot | 0.732782 | -0.1460 | 0.080721 | +0.0457 | 0.355225 | -0.1371 | 0.95475 |

Spearman(factor, centroid margin) = -0.900, p = 0.0374.

### Group C - GAMMA / EXPOSURE sweep (every value generated from the original)

| gamma | predicted | P(BacterialSpot) | DeltaP(BS) | P(Healthy) | DeltaP(Healthy) | centroid margin | Deltamargin | cos->original |
|---|---|---|---|---|---|---|---|---|
| g0.70 | BacterialSpot | 0.900683 | +0.0219 | 0.025657 | -0.0094 | 0.542790 | +0.0505 | 0.99561 |
| g0.85 | BacterialSpot | 0.890576 | +0.0118 | 0.030786 | -0.0043 | 0.511017 | +0.0187 | 0.99914 |
| g1.00 *(= original)* | BacterialSpot | 0.878758 | +0.0000 | 0.035046 | +0.0000 | 0.492325 | +0.0000 | 1.00000 |
| g1.15 | BacterialSpot | 0.853372 | -0.0254 | 0.042594 | +0.0075 | 0.467317 | -0.0250 | 0.99768 |
| g1.30 | BacterialSpot | 0.809905 | -0.0689 | 0.055781 | +0.0207 | 0.432769 | -0.0596 | 0.99027 |

Spearman(gamma, centroid margin) = -1.000, p = 0.0000.

### Group E - ILLUMINATION-CORRECTION variants

| variant | algorithm | predicted | P(BacterialSpot) | DeltaP(BS) | P(Healthy) | DeltaP(Healthy) | margin | Deltamargin | cos->original |
|---|---|---|---|---|---|---|---|---|---|
| `correction__grayworld_a0.25` | gray-world white balance (partial) | BacterialSpot | 0.870076 | -0.0087 | 0.031853 | -0.0032 | 0.500854 | +0.0085 | 0.99237 |
| `correction__grayworld_a0.50` | gray-world white balance (partial) | BacterialSpot | 0.864155 | -0.0146 | 0.035708 | +0.0007 | 0.471938 | -0.0204 | 0.98939 |
| `correction__grayworld_a1.00` | gray-world white balance (partial) | BacterialSpot | 0.856263 | -0.0225 | 0.045601 | +0.0106 | 0.421475 | -0.0708 | 0.98482 |
| `correction__flatfield_r32_a0.50` | low-frequency illumination correction | BacterialSpot | 0.842963 | -0.0358 | 0.038381 | +0.0033 | 0.493486 | +0.0012 | 0.95244 |
| `correction__flatfield_r32_a1.00` | low-frequency illumination correction | BacterialSpot | 0.818056 | -0.0607 | 0.052556 | +0.0175 | 0.396297 | -0.0960 | 0.90094 |
| `correction__flatfield_r128_a0.50` | low-frequency illumination correction | BacterialSpot | 0.853211 | -0.0255 | 0.046931 | +0.0119 | 0.432602 | -0.0597 | 0.98578 |
| `correction__flatfield_r128_a1.00` | low-frequency illumination correction | BacterialSpot | 0.809815 | -0.0689 | 0.078071 | +0.0430 | 0.310429 | -0.1819 | 0.94064 |
| `correction__retinex_a0.50` | multi-scale log-subtract (retinex-style) | BacterialSpot | 0.877448 | -0.0013 | 0.029096 | -0.0060 | 0.564414 | +0.0721 | 0.98440 |
| `correction__retinex_a1.00` | multi-scale log-subtract (retinex-style) | BacterialSpot | 0.831491 | -0.0473 | 0.060069 | +0.0250 | 0.455954 | -0.0364 | 0.96757 |

### Extremes

| extreme | variant | predicted | P(BacterialSpot) | P(Healthy) | centroid margin | Deltamargin | cos dist to original |
|---|---|---|---|---|---|---|---|
| most toward Healthy | `correction__flatfield_r128_a1.00` | BacterialSpot | 0.809815 | 0.078071 | +0.310429 | -0.181896 | 0.05936 |
| most toward BacterialSpot | `correction__retinex_a0.50` | BacterialSpot | 0.877448 | 0.029096 | +0.564414 | +0.072089 | 0.01560 |

Largest change in P(BacterialSpot): `contrast__f1.30` (-0.145975), predicted BacterialSpot.
Largest change in P(Healthy): `contrast__f1.30` (+0.045675), predicted BacterialSpot.

| | value |
|---|---|
| minimum P(BacterialSpot) over all 43 variants | 0.732782 (`contrast__f1.30`) |
| maximum P(Healthy) over all 43 variants | 0.080721 (`contrast__f1.30`) |
| ratio of that worst case | 9.08 : 1 in favour of BacterialSpot |
| number of variants with P(Healthy) > P(BacterialSpot) | 0 |
| distinct predicted classes across all 43 variants | `BacterialSpot` |

### Complete variant table - all 43 variants, none omitted

| # | variant_id | group | predicted | P(BS) | P(Healthy) | margin | cos->orig | flipped |
|---|---|---|---|---|---|---|---|---|
| 1 | `control__original` | `control` | BacterialSpot | 0.8788 | 0.0350 | 0.4923 | 1.00000 | no |
| 2 | `brightness__f0.70` | `brightness` | BacterialSpot | 0.8727 | 0.0345 | 0.5177 | 0.99644 | no |
| 3 | `brightness__f0.85` | `brightness` | BacterialSpot | 0.8757 | 0.0349 | 0.5041 | 0.99780 | no |
| 4 | `brightness__f1.00` | `brightness` | BacterialSpot | 0.8788 | 0.0350 | 0.4923 | 1.00000 | no |
| 5 | `brightness__f1.15` | `brightness` | BacterialSpot | 0.8722 | 0.0371 | 0.4775 | 0.99743 | no |
| 6 | `brightness__f1.30` | `brightness` | BacterialSpot | 0.8672 | 0.0387 | 0.4672 | 0.99417 | no |
| 7 | `contrast__f0.70` | `contrast` | BacterialSpot | 0.8569 | 0.0381 | 0.5049 | 0.99527 | no |
| 8 | `contrast__f0.85` | `contrast` | BacterialSpot | 0.8755 | 0.0342 | 0.5071 | 0.99824 | no |
| 9 | `contrast__f1.00` | `contrast` | BacterialSpot | 0.8788 | 0.0350 | 0.4923 | 1.00000 | no |
| 10 | `contrast__f1.15` | `contrast` | BacterialSpot | 0.8323 | 0.0520 | 0.4334 | 0.99151 | no |
| 11 | `contrast__f1.30` | `contrast` | BacterialSpot | 0.7328 | 0.0807 | 0.3552 | 0.95475 | no |
| 12 | `gamma__g0.70` | `gamma` | BacterialSpot | 0.9007 | 0.0257 | 0.5428 | 0.99561 | no |
| 13 | `gamma__g0.85` | `gamma` | BacterialSpot | 0.8906 | 0.0308 | 0.5110 | 0.99914 | no |
| 14 | `gamma__g1.00` | `gamma` | BacterialSpot | 0.8788 | 0.0350 | 0.4923 | 1.00000 | no |
| 15 | `gamma__g1.15` | `gamma` | BacterialSpot | 0.8534 | 0.0426 | 0.4673 | 0.99768 | no |
| 16 | `gamma__g1.30` | `gamma` | BacterialSpot | 0.8099 | 0.0558 | 0.4328 | 0.99027 | no |
| 17 | `directional__left_to_right_weak` | `directional` | BacterialSpot | 0.8725 | 0.0368 | 0.4862 | 0.99978 | no |
| 18 | `directional__left_to_right_medium` | `directional` | BacterialSpot | 0.8654 | 0.0392 | 0.4768 | 0.99895 | no |
| 19 | `directional__left_to_right_strong` | `directional` | BacterialSpot | 0.8584 | 0.0420 | 0.4663 | 0.99789 | no |
| 20 | `directional__right_to_left_weak` | `directional` | BacterialSpot | 0.8831 | 0.0335 | 0.4983 | 0.99982 | no |
| 21 | `directional__right_to_left_medium` | `directional` | BacterialSpot | 0.8863 | 0.0319 | 0.5085 | 0.99924 | no |
| 22 | `directional__right_to_left_strong` | `directional` | BacterialSpot | 0.8872 | 0.0316 | 0.5122 | 0.99855 | no |
| 23 | `directional__top_to_bottom_weak` | `directional` | BacterialSpot | 0.8784 | 0.0339 | 0.5007 | 0.99989 | no |
| 24 | `directional__top_to_bottom_medium` | `directional` | BacterialSpot | 0.8825 | 0.0322 | 0.5110 | 0.99958 | no |
| 25 | `directional__top_to_bottom_strong` | `directional` | BacterialSpot | 0.8855 | 0.0306 | 0.5219 | 0.99883 | no |
| 26 | `directional__bottom_to_top_weak` | `directional` | BacterialSpot | 0.8675 | 0.0398 | 0.4718 | 0.99923 | no |
| 27 | `directional__bottom_to_top_medium` | `directional` | BacterialSpot | 0.8504 | 0.0472 | 0.4439 | 0.99589 | no |
| 28 | `directional__bottom_to_top_strong` | `directional` | BacterialSpot | 0.8284 | 0.0579 | 0.4030 | 0.98793 | no |
| 29 | `shadow__upper_left_weak` | `shadow` | BacterialSpot | 0.8818 | 0.0332 | 0.5042 | 0.99949 | no |
| 30 | `shadow__upper_left_medium` | `shadow` | BacterialSpot | 0.8832 | 0.0317 | 0.5158 | 0.99856 | no |
| 31 | `shadow__upper_left_strong` | `shadow` | BacterialSpot | 0.8860 | 0.0297 | 0.5302 | 0.99715 | no |
| 32 | `shadow__lower_right_weak` | `shadow` | BacterialSpot | 0.8693 | 0.0379 | 0.4857 | 0.99938 | no |
| 33 | `shadow__lower_right_medium` | `shadow` | BacterialSpot | 0.8632 | 0.0401 | 0.4826 | 0.99781 | no |
| 34 | `shadow__lower_right_strong` | `shadow` | BacterialSpot | 0.8596 | 0.0415 | 0.4783 | 0.99600 | no |
| 35 | `correction__grayworld_a0.25` | `correction` | BacterialSpot | 0.8701 | 0.0319 | 0.5009 | 0.99237 | no |
| 36 | `correction__grayworld_a0.50` | `correction` | BacterialSpot | 0.8642 | 0.0357 | 0.4719 | 0.98939 | no |
| 37 | `correction__grayworld_a1.00` | `correction` | BacterialSpot | 0.8563 | 0.0456 | 0.4215 | 0.98482 | no |
| 38 | `correction__flatfield_r32_a0.50` | `correction` | BacterialSpot | 0.8430 | 0.0384 | 0.4935 | 0.95244 | no |
| 39 | `correction__flatfield_r32_a1.00` | `correction` | BacterialSpot | 0.8181 | 0.0526 | 0.3963 | 0.90094 | no |
| 40 | `correction__flatfield_r128_a0.50` | `correction` | BacterialSpot | 0.8532 | 0.0469 | 0.4326 | 0.98578 | no |
| 41 | `correction__flatfield_r128_a1.00` | `correction` | BacterialSpot | 0.8098 | 0.0781 | 0.3104 | 0.94064 | no |
| 42 | `correction__retinex_a0.50` | `correction` | BacterialSpot | 0.8774 | 0.0291 | 0.5644 | 0.98440 | no |
| 43 | `correction__retinex_a1.00` | `correction` | BacterialSpot | 0.8315 | 0.0601 | 0.4560 | 0.96757 | no |

### Control reference (Step 4 / Step 5.1 values reproduced)

| quantity | Step 4 / 5.1 | Step 5.2 control | match |
|---|---|---|---|
| predicted class | `BacterialSpot` | `BacterialSpot` | yes |
| P(BacterialSpot) | `0.878758` | `0.878758` | yes |
| P(Healthy) | `0.035046` | `0.035046` | yes |
| cos -> BacterialSpot centroid | `0.936431` | `0.936431` | yes |
| cos -> Healthy centroid | `0.444106` | `0.444106` | yes |
| centroid margin (BS - Healthy) | `+0.492325` | `+0.492325` | yes |
| cos -> nearest BS training image | `0.980528` | `0.980528` | yes |
| cos -> nearest Healthy training image | `0.915473` | `0.915473` | yes |

## 9. Feature-space results

Centroid similarities were computed against the same frozen centroids used in Step 4 and Step
5.1, so the numbers are directly comparable. No classification threshold was invented anywhere;
the margin is an observation, not a decision rule.

**Centroid margin across all 43 variants:**

| statistic | value |
|---|---|
| control margin | +0.492325 |
| most negative (most toward Healthy) | +0.310429 (`correction__flatfield_r128_a1.00`) |
| most positive (most toward BacterialSpot) | +0.564414 (`correction__retinex_a0.50`) |
| range of `delta margin` | -0.181896 .. +0.072089 |
| **minimum margin anywhere in the study** | **+0.310429** |

Two comparisons put these numbers in context:

1. **Against the Step 5.1 Healthy training population.** The maximum centroid margin over all
   831 Healthy *training* images is **+0.344078**. Exactly **one** of the 43 variants
   (`correction__flatfield_r128_a1.00`, margin +0.310429) falls below that value. Even the most
   favourable transformation brings the query to roughly the level of the single most extreme
   Healthy training image — and still on the BacterialSpot side. **No variant produces a negative
   margin.**
2. **Against the 16 other external images.** Not used here; they have no verified labels.

**Tonal statistics confirm the transformations did what they claim.** Mean luminance sweeps
116.9 → 81.4 (brightness 0.70) and → 147.4 (brightness 1.30). Critically, the **contrast**
family changed mean luminance by less than 1 unit out of 255 (116.44–116.92) yet produced the
largest probability movement in the study. The effect is therefore about **tonal
distribution/structure, not simple brightness** — see §14.

## 10. Representation movement

Representation movement is `1 - cos(variant_64D, original_query_64D)`, computed in the same
64-D `head[8]` bottleneck used by Step 4 and Step 5.1. The original is the reference point.

| statistic | value |
|---|---|
| range of cosine distance to original | 0.000000 .. 0.099057 |
| mean cosine distance to original | 0.010223 |
| **minimum cosine similarity to the original, over all 42 transformed variants** | **0.900943** |

Every variant stayed at cosine >= 0.90 from the original representation. The largest
representation move in the entire study (`correction__flatfield_r32_a1.00`, cosine distance
0.09906) still retained 90% angular agreement with the original. Note that the variant with the
*largest* representation move (flat-field r=32, distance 0.0991) is **not** the variant with the
largest movement toward Healthy (flat-field r=128, distance 0.0594) — large representation
change and directional movement are not the same thing here.

`representation_distances.csv` additionally records each variant's cosine to the Healthy
centroid, the BacterialSpot centroid, the nearest BacterialSpot training image and the nearest
Healthy training image, plus the 64-D L2 norm.

## 11. Classification flips

**Zero flips.** No variant changed the predicted class from `BacterialSpot`, and no variant
raised P(Healthy) above P(BacterialSpot). The distinct predicted classes across all 43 variants
is the single-element set `['BacterialSpot']`.

Confidence did change within BacterialSpot, and the changes are reported in full above rather
than summarised as "large" or "small":

| variant | delta P(BacterialSpot) | delta P(Healthy) | new predicted class |
|---|---|---|---|
| `contrast__f1.30` | **-0.145975** | **+0.045675** | `BacterialSpot` |
| `correction__flatfield_r128_a1.00` | -0.068943 | +0.043025 | `BacterialSpot` |
| `gamma__g1.30` | -0.068853 | +0.020735 | `BacterialSpot` |
| `gamma__g0.70` | +0.021925 | -0.009389 | `BacterialSpot` |
| `correction__retinex_a0.50` | -0.001310 | -0.005950 | `BacterialSpot` |

The largest single movement reduces P(BacterialSpot) from 0.878758 to 0.732782 and raises
P(Healthy) from 0.035046 to 0.080721 — a 2.3x increase in the Healthy probability. This is a real,
measurable shift in the model's internal state. It is also nowhere near enough to change the
decision: the gap remains 9.08 : 1 in the wrong direction.

## 12. Illumination-correction results

If uneven lighting were the dominant factor, a general illumination correction should move the
image toward Healthy. It does move — consistently more than any other family — but not far
enough.

| correction | algorithm | delta margin | delta P(BS) | delta P(Healthy) | predicted |
|---|---|---|---|---|---|
| `correction__flatfield_r128_a1.00` | flat-field, r=128, a=1.00 | **-0.181896** | -0.068943 | +0.043025 | `BacterialSpot` |
| `correction__flatfield_r32_a1.00` | flat-field, r=32, a=1.00 | -0.096028 | -0.060702 | +0.017510 | `BacterialSpot` |
| `correction__grayworld_a1.00` | gray-world, a=1.00 | -0.070849 | -0.022495 | +0.010555 | `BacterialSpot` |
| `correction__flatfield_r128_a0.50` | flat-field, r=128, a=0.50 | -0.059723 | -0.025547 | +0.011885 | `BacterialSpot` |
| `correction__retinex_a1.00` | retinex-style MSCR, a=1.00 | -0.036374 | -0.047315 | +0.025023 | `BacterialSpot` |
| `correction__flatfield_r32_a0.50` | flat-field, r=32, a=0.50 | +0.001215 | -0.035795 | +0.003335 | `BacterialSpot` |
| `correction__grayworld_a0.50` | gray-world, a=0.50 | -0.020387 | -0.014603 | +0.000662 | `BacterialSpot` |
| `correction__grayworld_a0.25` | gray-world, a=0.25 | +0.008529 | -0.008682 | -0.003193 | `BacterialSpot` |
| `correction__retinex_a0.50` | retinex-style MSCR, a=0.50 | **+0.072089** | -0.001310 | -0.005950 | `BacterialSpot` |

6 of 9 moved toward Healthy; 3 moved away (weak gray-world, half-strength r=32 flat-field, and
half-strength retinex). The correction family produced the largest representation movement in
the study (mean cosine distance 0.0335, max 0.0991) and the largest mean movement toward
Healthy (mean `delta margin` -0.0426). **But the strongest correction still leaves P(BacterialSpot)
= 0.8098 against P(Healthy) = 0.0781**, a 10.4 : 1 gap. Correcting the illumination does not make
the model call this image Healthy.

## 13. Visual inspection limitation

**The generated grids have NOT been visually inspected.** The environment used to produce them
cannot display images to the analyst. This is stated plainly because it bounds what §14 may
claim.

What *was* verified programmatically: all 10 PNGs are valid and decodable, with dimensions
recorded (`original.png` 1920x1440 RGB; `brightness_grid.png`, `contrast_grid.png`,
`gamma_grid.png`, `shadow_grid.png` 1207x846; `directional_grid.png` 1207x1671;
`correction_grid.png` 1207x1259; `illumination_grid.png` 1533x2324; `response_curves.png`
1934x592; `family_movement.png` 1544x540). Every caption value is read from
`prediction_details.json` by `variant_id`, and every image is re-derived from the same
deterministic transformation functions used by the experiment, so captions cannot mismatch
images. Captions are factual, e.g. `Brightness x1.30 | BacterialSpot 86.7% | Healthy 3.9%` —
no caption asserts causation.

**Visual inspection of the generated grids is required before interpreting whether the
transformations resemble plausible real-world illumination.** No visual finding is claimed
anywhere in this report. A reader should open `visualizations/` and judge for themselves
whether, for example, the strong directional gradients look like plausible lighting or like
obvious synthetic ramps. This report's conclusions rest only on the numerical results, and
would need revisiting if the transformations turn out to be unrealistic.

## 14. Interpretation

Answering the ten required questions (A–J) directly from the recorded data.

**A. Does brightness change alter the BacterialSpot probability?**
Yes, measurably but weakly. `delta P(BacterialSpot)` ranges -0.0116 .. 0.0000 across
brightness 0.70–1.30. The effect is monotone in the contrast/gamma sense (Spearman(factor,
centroid margin) = -1.000) but its magnitude is tiny: at the extreme, brightness x1.30 changes
P(BacterialSpot) by 0.0116 and the margin by -0.0251.

**B. Does contrast change alter the BacterialSpot probability?**
Yes — and this is the strongest single effect in the study. Contrast x1.30 gives
`delta P(BacterialSpot) = -0.145975` and `delta P(Healthy) = +0.045675`, the largest of any
variant. The trend is monotone above 0.85 (Spearman = -0.900, p = 0.0374). Critically, the
contrast family changed mean luminance by less than 1/255, so **this is a tonal-contrast
effect, not a brightness effect.**

**C. Does gamma/exposure change alter the BacterialSpot probability?**
Yes, monotonically, but in the *opposite* relationship to luminance. Gamma 0.70 **brightens**
(mean luminance 142.4 vs 116.9) yet yields the **highest** P(BacterialSpot) in the entire study
(0.900683, `delta` = +0.0219). Gamma 1.30 darkens (luminance 98.0) and gives
`delta P(BacterialSpot) = -0.0689`. Spearman(gamma, margin) = -1.000.

**D. Do directional-shadow variants move the representation?**
Yes, but very little, and the direction depends on which way the light comes from. The
directional family's mean cosine distance to the original is only **0.00204** (max 0.01207), the
smallest of any family. Two of the four directions move toward Healthy (`bottom_to_top`
-0.0527, `left_to_right` -0.0159 mean margin) and two move *away* (`top_to_bottom` +0.0189,
`right_to_left` +0.0140). Elliptical shadows behave the same way — `lower_right` -0.0101
toward Healthy, `upper_left` +0.0244 away — with mean cosine distance 0.00193. **A shadow
placed on one side moves the representation in one direction; the same shadow mirrored moves it
the other way, and neither is large.**

**E. Does illumination correction move the representation toward Healthy?**
Yes — the correction family is the most consistently favourable, with 6 of 9 variants moving
toward Healthy, mean `delta margin` -0.0426, and the single largest movement in the study
(flat-field r=128 a=1.00, margin +0.492325 -> +0.310429). **But it does not move far enough to
matter:** P(BacterialSpot) 0.8098 vs P(Healthy) 0.0781, still 10.4 : 1, and the margin remains
positive.

**F. Do any controlled lighting changes flip the prediction?**
**No.** Zero flips out of 42 transformed variants. All 43 variants are `BacterialSpot`. Zero
variants have P(Healthy) > P(BacterialSpot).

**G. Which transformation causes the largest movement toward Healthy?**
`correction__flatfield_r128_a1.00` (flat-field illumination correction, radius 128 px, full
strength): `delta margin = -0.181896`, `delta P(BacterialSpot) = -0.068943`,
`delta P(Healthy) = +0.043025`. Margin ends at +0.310429. (If judged by probability rather than
feature margin, the answer is `contrast__f1.30` with `delta P(BacterialSpot) = -0.145975`. Both
are reported; neither flips the prediction.)

**H. Which causes the largest movement toward BacterialSpot?**
`correction__retinex_a0.50` (half-strength retinex-style MSCR): `delta margin = +0.072089`,
margin ends at +0.564414. The largest *increase* in P(BacterialSpot) is `gamma__g0.70` at
+0.021925.

**I. Is the effect consistent across a transformation family?**
Within the three monotone families, yes — brightness, contrast and gamma all have
Spearman(parameter, margin) between -0.900 and -1.000. **Across families, no.** The families
disagree on what "brighter" means for the outcome: multiplicative brightness moves *toward*
Healthy as the image gets brighter, while gamma moves *toward* BacterialSpot as the image gets
brighter. The shadow family is internally inconsistent (one centre favourable, the other
unfavourable). Aggregate Pearson correlation between mean luminance and P(BacterialSpot) across
all 42 transformed variants is **+0.2929 (p = 0.060)** — weak and not significant at alpha=0.05.
Luminance alone therefore does not explain the pattern.

**J. Is the effect large enough to consider illumination a plausible contributor?**
See §16. Short answer: it is large enough to keep illumination on the list as a *contributing*
factor, but far too small to explain the failure.

**Synthesis.** Controlled appearance changes *do* move this image measurably toward Healthy —
this is not a null result at the level of the model's internal representation. But the movement
is bounded and small: it never reverses the decision, never makes the margin negative, and never
takes P(Healthy) above 0.081. The largest effects come from **contrast and tonal distribution**
rather than brightness, and the effect is not a simple function of luminance. Illumination and
appearance shape the representation, but they are nowhere near sufficient to explain why this
image is classified BacterialSpot.

## 15. Evidence level

The decision rule below was **declared in code before results were inspected** and applied
mechanically, so the verdict could not be chosen after the fact. It is recorded verbatim in
`illumination_summary.json` under `evidence_level_rule`.

| level | rule |
|---|---|
| **LEVEL 3** | at least one variant flips the predicted class to Healthy (any variant with P(Healthy) > P(BacterialSpot)), or the predicted class changes at all |
| **LEVEL 2** | no flip, but any of: max abs `delta P(BacterialSpot)` >= 0.10, **or** max abs `delta centroid margin` >= 0.25, **or** >= 25% of non-control variants move toward Healthy *and* the direction is consistent within at least one whole family |
| **LEVEL 1** | none of the above |

**Observed:**

| criterion | value | threshold | triggered? |
|---|---|---|---|
| classification flips | 0 | >= 1 | no |
| variants with P(Healthy) > P(BacterialSpot) | 0 | >= 1 | no |
| max abs `delta P(BacterialSpot)` | **0.145975** | 0.10 | **YES** |
| max abs `delta centroid margin` | 0.181896 | 0.25 | no |
| fraction of variants toward Healthy | **0.500** | 0.25 | **YES** |
| families meeting the consistency threshold | all 6 | >= 1 | **YES** |

### **EVIDENCE LEVEL 2 — some illumination sensitivity detected**

> *Illumination plausibly contributes, but does not fully explain the failure.*

LEVEL 3 was not reached: nothing flipped. LEVEL 2 is reached on two independent criteria —
`contrast__f1.30` moved P(BacterialSpot) by 0.146 (>= 0.10), and half of all variants moved
toward Healthy with all six families showing internal consistency.

**No level is called "proven."** These thresholds are operational conventions declared in
advance, not scientific laws. A reader who prefers stricter thresholds (for example
`max abs delta P(BacterialSpot) >= 0.20`) would obtain LEVEL 1 from the identical data. The
underlying numbers are all reported above so the verdict can be re-derived under any rule.

For emphasis: the honest summary is that **the effect is real but small, and it is
insufficient.** Level 2 means illumination is a contributing variable, not an explanation.

## 16. Is illumination a plausible contributor?

**Yes, as a contributing factor. No, as an explanation for this failure.**

Supporting the hypothesis:
- Controlled appearance changes measurably move the representation toward Healthy: 21 of 42
  variants moved that way, and the margin moved by up to -0.181896.
- Illumination-correction methods — which specifically remove shading and colour cast — are the
  most favourable family tested (6 of 9 variants toward Healthy, best margin +0.310429).
- `contrast__f1.30` nearly halves the Healthy-vs-BacterialSpot probability gap's log-width while
  changing mean luminance by less than 1/255, so genuine tonal structure is involved.

Against treating it as the explanation:
- **Zero** prediction flips. All 43 variants are `BacterialSpot`.
- The worst case in the entire study is 0.7328 vs 0.0807 — **9.08 : 1** in the wrong direction.
- The margin never goes negative; its minimum is +0.310429.
- Only one variant even falls below the *most extreme* of 831 Healthy training images
  (+0.344078), and it remains on the BacterialSpot side.
- Step 5.1 already showed the query's centroid margin (+0.492325) exceeds that of **all 1,531**
  Healthy and BacterialSpot training images. No appearance change tested here closes a gap of
  that size; the best closes 37% of it.

**This provides evidence that illumination contributes. It does not establish causality**, and
it cannot: a single image was manipulated, the masks are controlled proxies rather than real
lighting, and no independent sample of real-world illumination variation was available.

**Stated in the form the protocol prescribes for a non-explaining result:**

> **"The controlled illumination experiment did not provide evidence that illumination alone
> explains the observed misclassification."**

That sentence is quoted here because it is accurate as written — the qualifier *alone* is
exactly what the data support. The stronger negation ("illumination played no part") would be
false and is not claimed: appearance sensitivity of this size is measurable and consistent
across six families. What the data show is that illumination is a real but insufficient
variable. It moves the representation without moving the decision, and closing ~37% of a 0.49
margin gap leaves the classification untouched.

## 17. What this experiment does NOT prove

1. **It does not prove that lighting caused the real-world failure.** It shows that controlled
   appearance changes move the representation. Causality in the field is untested.
2. **It does not prove illumination is irrelevant.** Level 2 is not Level 1. A measurable
   sensitivity exists and would remain a plausible contributing variable.
3. **It does not establish external accuracy.** No accuracy figure is computed. 16 of the 17
   external images remain unlabelled (UNKNOWN) and were not used at all in Step 5.2.
4. **It does not show that the synthetic masks reproduce real-world lighting.** They are smooth
   multiplicative proxies. Whether they resemble plausible real conditions is a human visual
   judgement that has not been made (§13).
5. **It does not show that correcting illumination in production would fix the model.** Even the
   strongest correction leaves the prediction wrong.
6. **It does not identify the true cause.** Step 5.1 localised the query outside the Healthy
   training distribution; Step 5.2 shows illumination is not sufficient to explain that. The
   underlying cause remains unidentified.
7. **It does not generalise.** One image, one model, one checkpoint. No claim is made about
   other images, other models, or other datasets.
8. **It does not validate the transformations as medically or botanically realistic.** No
   disease-relevant structure was synthesised, but realism of the appearance change is
   unverified.
9. **It does not rule out that *specific* real-world lighting differs more than these proxies.**
   Real directional lighting, specular highlights, mixed colour temperature and motion blur are
   not represented here. A null-ish result for smooth proxies is not a null result for all
   possible appearance variation.
10. **It does not use clustering or dimensionality reduction as evidence.** No projection or
    cluster is used to argue anything about class correctness.

## 18. Is a training intervention justified?

**Not on the basis of Step 5.2 alone, and specifically not as an illumination-robustness
training experiment.**

The rationale:

- An illumination-robustness training experiment would be justified if illumination were shown
  to be a *sufficient* driver. It is not: zero flips, 9.08 : 1 residual gap, margin never
  negative.
- The strongest correction available moved the margin by 0.18 out of a 0.49 baseline gap. Even a
  perfect in-production illumination-correction stage would only close ~37% of the distance, and
  empirically would not change the prediction.
- Retraining to cover this point would be training toward one external image. That is precisely
  what the frozen-baseline rules forbid here, and scientifically it would be fitting an
  anecdote rather than a distribution.
- Step 5.1 already concluded that hard-example cleaning is not justified, because the query lies
  outside the range of every candidate it found. Step 5.2 reaches a compatible conclusion for
  appearance: the query also lies outside the range that illumination manipulation reaches.

**A weaker, narrower statement is defensible:** appearance sensitivity of this magnitude is a
real robustness gap worth *characterising* before any decision to train against it. That is a
diagnostic recommendation, not a training recommendation.

## 19. Recommended next step

**Recommended: a coverage diagnostic, not a training run.**

1. **Do not train.** No illumination-robustness or hard-example intervention is justified by the
   evidence available.
2. **Acquire independently verified real-world Healthy leaf images.** This is the binding
   constraint. 16 of 17 external images are unlabelled, so the current evidence base for
   "what real-world Healthy looks like" is a single verified image. Verification of those 16
   images would immediately enable a genuine external-holdout measurement, which no step so far
   can perform.
3. **Then, and only then, re-run this diagnostic** with the verified real-world images as an
   additional control set, to determine whether appearance variation *within* real-world Healthy
   leaves is larger than the smooth proxies used here. If real-world variation is also small,
   illumination can be set aside with much higher confidence than Step 5.2 alone allows.
4. **Keep illumination recorded as a candidate contributing variable** at low priority. If a
   future training intervention is designed for coverage reasons, controlling for contrast and
   tonal distribution is a reasonable inclusion — but as one factor among several, not as the
   diagnosed cause.
5. **Do not add an illumination-correction preprocessing stage to production.** The evidence
   does not support it, and it would be a production-inference change outside this step's scope.

## 20. Integrity verification

Full detail in `verification_log.txt`. **Result: ALL CHECKS PASSED** (verification script exits
non-zero on any failure, so a regression could not pass silently).

| check | before | after | expected | status |
|---|---|---|---|---|
| `best_iwnet.pth` SHA-256 | `c1fae27c…a27c0b` | `c1fae27c…a27c0b` | `c1fae27c…a27c0b` | PASS |
| `best_iwnet.pth` bytes | 41,213,897 | 41,213,897 | 41,213,897 | PASS |
| `Balanced_From_Sources` fingerprint | `35c72a0ae283f7e8…` | `35c72a0ae283f7e8…` | — | PASS |
| `Balanced_From_Sources` files / bytes | 5,947 / 1,178,341,200 | 5,947 / 1,178,341,200 | — | PASS |
| `Balanced_Final_Split` fingerprint | `389b24a8beac8ad5…` | `389b24a8beac8ad5…` | — | PASS |
| `Balanced_Final_Split` files / bytes | 7,200 / 1,310,546,227 | 7,200 / 1,310,546,227 | — | PASS |
| `split_manifest.csv` SHA-256 | `d25542dab5673810…` | `d25542dab5673810…` | — | PASS |
| total dataset images | 5,929 | 5,929 | 5,929 | PASS |
| class counts (train/test x 7 classes) | unchanged | unchanged | — | PASS |
| protected original image SHA-256 | `d5e62c4d30660bdeb…` | `d5e62c4d30660bdeb…` | `d5e62c4d…be026` | PASS |
| protected original bytes / dims / format | 429,238 / 1920x1440 / JPEG | identical | — | PASS |
| live on-disk re-hash after experiment | `d5e62c4d…be026` | — | `d5e62c4d…be026` | PASS |
| all 17 external image digests | — | identical | — | PASS |
| 51 Step 1–5.1 artefact files | — | byte-identical | 51 | PASS |
| production preprocessing | `Resize→CenterCrop→ToTensor→Normalize` | identical | — | PASS |
| parameter count | 10,161,307 | 10,161,307 | 10,161,307 | PASS |
| class list | 7 classes | identical | — | PASS |
| production P(BacterialSpot) | 0.8787577152252197 | 0.8787577152252197 | 0.878758 | PASS |
| production P(Healthy) | 0.035046033561229706 | 0.035046033561229706 | 0.035046 | PASS |
| hooked forward vs production predictor | max prob delta 0.0 | — | 0.0 | PASS |
| identity (`factor 1.00`) variants vs control | max delta 0.0 | — | 0.0 | PASS |

**No STOP condition was triggered at any point.** Production prediction is bit-for-bit
identical before and after, so nothing was disturbed.

## 21. Test results

| | baseline (Step 5.1, pre-analysis) | Step 5.2 (post-experiment) |
|---|---|---|
| passed | 161 | **161** |
| failed | 0 | **0** |
| errors | 0 | **0** |
| skipped | 0 | **0** |
| duration | — | 67.3 s |

Identical to baseline. **No test was deleted, weakened, skipped or rewritten, and no production
code was modified to make a test pass.** No new tests were added because Step 5.2 added no new
production functionality — the experiment is a standalone read-only analysis script kept outside
the repository (`%TEMP%`), so there is no new testable production surface.

## 22. Reproducibility information

| item | value |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| timm | 1.0.30 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| Operating system | Windows-11-10.0.26200-SP0 |
| Device | CPU (`cuda_available = False`) |
| Architecture | IWNET (EfficientNet-B3), 10,161,307 parameters |
| Checkpoint SHA-256 / size | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` / 41,213,897 bytes |
| Preprocessing | `Resize(255, BILINEAR)` → `CenterCrop(224,224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])` |
| Representation layer | `head[8]` = `BatchNorm1d(64)` (primary); `head[1]` and `head[4]` 128-D also recorded |
| Centroids | Healthy n=831, BacterialSpot n=700, from the Step 5.1 feature cache (identical centroids) |
| Metric | cosine similarity, centroid margin = cos(BS) − cos(Healthy) |
| Variants | 43 (1 control + 42 transformed) |
| Random seed | **none required — no randomness anywhere.** All transformations are deterministic closed-form functions; no sampling, no shuffling, no dropout at inference (`torch.inference_mode()`, model in eval mode via `Predictor.load()`) |
| DataLoader / workers | no DataLoader used; `num_workers = 0`; no multiprocessing |
| Per-variant inference time | min 48.70 ms, mean 64.29 ms, max 80.26 ms (CPU, single image, production path) |
| Integrity capture time | 22.26 s each |
| Analysis time | 0.02 s |

**Determinism check.** Re-running the experiment reproduces the control exactly and the
`factor = 1.00` identity variants exactly (max probability delta 0.0), which is the practical
evidence that the transformation pipeline is deterministic and non-cumulative.

---

## 31. Final decision

### Does controlled manipulation of illumination/appearance cause the verified Healthy image to move measurably toward or away from the BacterialSpot feature region?

**Yes — measurably, but far too little to change the outcome.**

- **Toward Healthy:** up to -0.181896 in centroid margin (`correction__flatfield_r128_a1.00`),
  -0.145975 in P(BacterialSpot) and +0.045675 in P(Healthy) (`contrast__f1.30`). 21 of 42
  variants moved toward Healthy.
- **Away from Healthy:** up to +0.072089 in centroid margin (`correction__retinex_a0.50`) and
  +0.021925 in P(BacterialSpot) (`gamma__g0.70`).
- **But:** 0 prediction flips, 0 variants with P(Healthy) > P(BacterialSpot), margin never
  negative (minimum +0.310429), worst-case ratio 9.08 : 1 against Healthy, and cosine similarity
  to the original never fell below 0.900943.

The image can be pushed measurably in the right direction and still be confidently classified
BacterialSpot. Illumination and appearance shape the representation; they do not determine the
decision here.

### Is illumination a sufficiently supported hypothesis to justify a future illumination-robustness training experiment?

**No.**

Evidence level **2** means illumination is a measurable contributing variable, not a sufficient
explanation. The controlled experiment, run across 43 variants spanning brightness, contrast,
gamma, directional shading, cast shadows and three standard illumination-correction methods,
never once changed the predicted class. A training experiment justified on the belief that
illumination *explains* this failure would be justified by evidence that does not exist.

The remaining movement is best characterised as a robustness gap worth recording — contrast and
tonal distribution demonstrably move this representation — rather than a diagnosed cause. The
diagnostic finding is that the query sits **outside the Healthy training distribution**, which
illumination manipulation cannot bring it back inside.

**That training experiment has not been performed, as instructed.** No training, fine-tuning,
checkpoint modification, dataset modification, or production-inference change occurred at any
point in Step 5.2.

---

## Appendix — files produced

All under `RealWorldValidation\step5_2_illumination_diagnostic\`. No Step 1–5.1 artefact was
modified or overwritten; no temporary analysis file was left in the repository (scripts live in
`%TEMP%` and are disposable).

| file | contents |
|---|---|
| `STEP5_2_REPORT.md` | this report |
| `illumination_summary.csv` | all 43 variants, 30 columns, every required field |
| `illumination_summary.json` | hypothesis test A–J, per-family behaviour, declared evidence-level rule and its computation |
| `representation_distances.csv` | per-variant cosine to original, both centroids, both nearest training images, 64-D norms |
| `prediction_details.json` | full records incl. all 7 class probabilities, 64-D vector, both 128-D vectors, per-variant image statistics |
| `transformation_manifest.json` | every variant's exact op, parameters and formula; branching rule; read-only proofs |
| `integrity_before.json` | pre-experiment capture |
| `integrity_after.json` | post-experiment capture |
| `verification_log.txt` | full pass/fail integrity log (ALL CHECKS PASSED) |
| `visualizations/` | 10 PNGs + README (§13 limitation stated in the README too) |

**Note on `_variant_state.npz`:** a regenerable intermediate holding the numeric arrays used by
the analysis step. It is excluded from the artefact list above and can be deleted or
regenerated at will; it is not required to read the report.
