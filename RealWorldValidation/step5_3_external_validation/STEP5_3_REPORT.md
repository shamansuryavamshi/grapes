# Step 5.3 - External Image Ground-Truth Validation

**Status: complete. READ-ONLY. No model, dataset, label or threshold was modified.**

| | |
|---|---|
| Step | 5.3 external image ground-truth validation |
| Baseline commit | `e101607c78a3c61dda4e3e0d91dfe1ae9a3901a4` (unchanged) |
| Checkpoint | `Balanced_From_Sources\best_iwnet.pth` (unchanged) |
| Training performed | none |
| Commits created | none |
| Date of analysis | 2026-09-30 |

---

## 1. Objective

Step 5.3 asks one question: **is the failure observed on `grape leaf.jpg` an
isolated external example, a recurring Healthy -> BacterialSpot failure, a broader
external domain-shift problem, or is the answer still undecidable because
reliable external ground truth does not exist?**

The step is a validation step, not a modelling step. It runs the frozen
production predictor over the whole external set, then tries hard to obtain
**independent** ground truth for each image, and reports external performance only
where that ground truth genuinely exists.

The central constraint of this step is that the model cannot be used to establish
its own ground truth. Every adjudication below treats the model prediction as an
output to be scored, never as an input to labelling.

**Headline result: 16 of 17 external images remain UNKNOWN. The answer to the
question is CATEGORY_A - insufficient ground truth to decide.**

---

## 2. Frozen Baseline

The frozen scientific baseline was not created, switched, amended, reset or
squashed by this step.

| item | value |
|---|---|
| Baseline commit | `e101607c78a3c61dda4e3e0d91dfe1ae9a3901a4` |
| Checkpoint | `Balanced_From_Sources/best_iwnet.pth` |
| Checkpoint SHA-256 | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` |
| Checkpoint size | 41,213,897 bytes |
| Architecture | IWNET, EfficientNet-B3 backbone |
| Parameter count | 10,161,307 |
| Classes | 7 |
| Device | CPU (`cuda_available = False`) |
| DataLoader workers | 0 (no multiprocessing used in this step) |

Class list:

```python
['BacterialSpot', 'Black_Rot', 'DownyMildew', 'Esca', 'Healthy', 'Irrelavant', 'PowderyMildew']
```

Production preprocessing, read back from the live predictor object:

  1. Resize(255)  interpolation=InterpolationMode.BILINEAR
  2. CenterCrop(224, 224)
  3. ToTensor()
  4. Normalize() mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]

Dataset state at capture time:

| dataset | fingerprint (first 16) | files | bytes |
|---|---|---|---|
| `Balanced_From_Sources` | `35c72a0ae283f7e8` | 5,947 | 1,178,341,200 |
| `Balanced_Final_Split` | `389b24a8beac8ad5` | 7,200 | 1,310,546,227 |

Class counts: train/BacterialSpot 560, train/Black_Rot 720, train/DownyMildew 682, train/Esca 700, train/Healthy 664, train/Irrelavant 746, train/PowderyMildew 668, test/BacterialSpot 140, test/Black_Rot 180, test/DownyMildew 171, test/Esca 176, test/Healthy 167, test/Irrelavant 187, test/PowderyMildew 168
Total dataset images: 5,929
`split_manifest.csv` SHA-256: `d25542dab56738102e3d429b9558890f8a9571e628bc414ec8d8c7fd20103cc2`

**Live production prediction on the verified image, captured before the
experiment and re-captured after it:**

| | before | after |
|---|---|---|
| predicted class | BacterialSpot | BacterialSpot |
| P(BacterialSpot) | 0.878758 | 0.878758 |
| P(Healthy) | 0.035046 | 0.035046 |

The prediction reproduced to full float precision. The pipeline was therefore
reproduced exactly before any Step 5.3 analysis began.

---

## 3. External Dataset

Directory: `C:\Users\shaman\Desktop\healthy leaves`
Distinct external images: **17**
Existing manifest used as the starting point: `RealWorldValidation\manifest.csv`
(SHA-256 `2381b282d32c060d`).

All 17 images are READ-ONLY. Their digests, sizes and dimensions were recorded
before processing (`integrity_before.json`) and re-verified afterwards
(`integrity_after.json`). Every image was opened for reading only; processing ran
in memory; no file was overwritten, resized in place, recompressed, renamed,
moved, cropped, edited, annotated or had metadata altered.

| filename | dimensions | bytes | sha256 (first 16) |
|---|---|---|---|
| `grape leaf.jpg` | 1920x1440 | 429,238 | `d5e62c4d30660bde` |
| `green-leaf-of-grapes-in-the-garden-free-photo.jpg` | 933x700 | 82,790 | `6d1dd8c348d67917` |
| `images (1).jfif` | 533x375 | 24,757 | `d1a95b65ef38dc41` |
| `images (10).jfif` | 547x365 | 42,151 | `38ac45742f2b800b` |
| `images (11).jfif` | 597x335 | 24,257 | `cdcb50419baa50c7` |
| `images (12).jfif` | 400x400 | 47,497 | `4c7d2d4c839af023` |
| `images (13).jfif` | 666x460 | 38,077 | `3b5901d4fa651793` |
| `images (2).jfif` | 300x396 | 11,473 | `5a459dfde9feccda` |
| `images (3).jfif` | 576x347 | 37,308 | `0ce299c5bfd7f147` |
| `images (4).jfif` | 499x375 | 41,329 | `33a07a196f717992` |
| `images (5).jfif` | 335x597 | 41,337 | `7a4111f817ee776c` |
| `images (6).jfif` | 447x447 | 27,839 | `cf4c9a55840766e0` |
| `images (7).jfif` | 480x640 | 33,960 | `1e7834529977ef16` |
| `images (8).jfif` | 439x300 | 21,730 | `bda12b2a404ba8b8` |
| `images (9).jfif` | 612x500 | 53,713 | `99e4000b00c7eb1e` |
| `images.jfif` | 515x388 | 42,105 | `3212ca515690573c` |
| `img1.jfif` | 738x414 | 49,158 | `85215998727635b4` |

All 17 SHA-256 values are distinct. No external image is an exact byte-duplicate
of another.

Only **1 of 17** files carry any machine-readable provenance of any kind. That
single finding is what constrains the whole step.

---

## 4. Ground-Truth Method

### 4.1 The rule

Ground truth for an external image is established **only** from independent
evidence. The following were all available and all excluded as evidence, by
protocol:

- the frozen model's prediction
- the filename
- any folder name
- visual impression or intuition
- similarity to training images
- nearest-neighbour results
- the analyst's interpretation of an image
- the confidence score
- agreement across repeated runs or across models

The last point matters and is stated explicitly: **model agreement is not
verification.** If the only evidence for a label were "IWNET predicts
BacterialSpot", the image stays UNKNOWN.

### 4.2 What was actually done

An exhaustive read-only metadata scan was run over all 17 files. It walked the
JPEG marker chain and extracted every carrier that could plausibly hold a source
or attribution: EXIF across all IFDs (including the Exif and GPS sub-IFDs), the
XMP packet, the Photoshop/IPTC IRB in APP13, JPEG COM segments, the JFIF header,
and a raw byte-level scan for embedded URLs. The full per-file result is in
`external_metadata.json`.

On top of that, provenance was pursued externally: the one URL embedded in the
image's own XMP block was fetched live, and web searches were run against the
descriptive filenames.

### 4.3 Evidence attempts, including the failures

Every attempt is logged, **including the ones that failed**, so a reader can see
the investigation was actually performed rather than assumed. Full detail in
`evidence_log.csv`.

| ID | target | method | outcome |
|---|---|---|---|
| E01 | `all 17 files` | Exhaustive read-only metadata scan: EXIF (all IFDs via PIL + raw marker walk), XMP packet, Photoshop/IPTC IRB (APP13), JPEG COM segments, JFIF header, raw byte scan for URLs | FAILED to establish provenance for 16 images |
| E02 | `grape leaf.jpg` | Live HTTP fetch of the URL embedded in the file's own XMP block | PARTIAL |
| E03 | `grape leaf.jpg` | Web search for the source identifier 'torange.biz grape leaf 39537' | FAILED |
| E04 | `green-leaf-of-grapes-in-the-garden-free-photo.jpg` | Two web searches: descriptive slug terms, then the exact filename string in quotes | FAILED |
| E05 | `images (1..13).jfif, images.jfif, img1.jfif` | Filename analysis | FAILED |
| E06 | `images (1..13).jfif, images.jfif, img1.jfif` | Reverse image search | NOT ATTEMPTED |
| E07 | `all 16 UNKNOWN images` | Perceptual-hash similarity to the training set (carried over from Step 1) | DELIBERATELY NOT USED |
| E08 | `all 17 images` | Model prediction | DELIBERATELY NOT USED FOR GROUND TRUTH |
| E09 | `all 17 images` | Visual inspection by the analyst | NOT ATTEMPTED |
| E10 | `grape leaf.jpg` | Consult the pre-existing manifest (RealWorldValidation/manifest.csv) | CONSISTENT |

Two attempts were not performed at all, and are recorded as gaps rather than
quietly omitted: reverse image search (`E06`) and analyst visual inspection
(`E09`). Neither capability exists in this environment. A human with reverse
image search and the ability to view images could close both gaps, and that is
the single highest-value action available for a future iteration of this step.

---

## 5. Evidence Classification

Status counts:

| status | count | scored? |
|---|---|---|
| VERIFIED | 1 | yes |
| SUPPORTED | 0 | yes, as a separate lower-certainty figure |
| DISPUTED | 0 | no |
| UNKNOWN | 16 | no |
| **total** | **17** | |

| filename | status | label | prior status | evidence level | evidence source |
|---|---|---|---|---|---|
| `green-leaf-of-grapes-in-the-garden-free-photo.jpg` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (1).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (10).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (11).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (12).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (13).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (2).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (3).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (4).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (5).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (6).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (7).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (8).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images (9).jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `images.jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `img1.jfif` | **UNKNOWN** | - | UNVERIFIED | NONE | none - no EXIF, no XMP, no IPTC, no JPEG comment, no embedded URL, no ... |
| `grape leaf.jpg` | **VERIFIED** | Healthy | VERIFIED | A (identity only) / **not A** for disease | Project-owner human visual confirmation, recorded in RealWorldValidati... |

### 5.1 `grape leaf.jpg` - the one image with any provenance

Its XMP block carries the source URL `https://torange.biz/grape-leaf-39537`. That page was fetched live on 2026-09-30 and
matched the file: the page advertises "HD wallpaper size 1920px" and the file is
1920x1440, and the URL in the page matches the URL embedded in the image's own
metadata. **Image identity is confirmed.**

The page title is "Grape leaf #39537", licensed CC-BY 4.0. Its download caption
reads "Grape leaf Green grape vine leaf with droplets and veins in HD wallpaper
size 1920px".

**The source makes no statement about health, disease, or absence of disease.**
A stock caption is written for aesthetics, not pathology. Not mentioning spots is
not the same as certifying that there are none.

The VERIFIED status therefore rests on something else: the project owner's
independent human visual confirmation, recorded in `manifest.csv` as
`ground_truth_source = "Independently confirmed"`. That confirmation is human and
independent of the model, but it is **not** a qualified plant-pathology
determination.

This is recorded honestly rather than smoothed over:

| question | answer |
|---|---|
| Is the image the one the URL points to? | **Yes**, confirmed |
| Does an independent, model-free source establish the condition? | **No** |
| Is the human confirmation independent of the model? | **Yes** |
| Is it an expert/institutional pathology determination? | **No** |
| Status assigned | VERIFIED (the project's established ground truth, stated as a premise of this step) |

Under a strict reading of the declared evidence hierarchy this is **not LEVEL A**
for the disease question, because LEVEL A requires an expert, institutional or
authoritative-source identification of the condition. It is recorded as VERIFIED
because that is the project's established, independently confirmed ground truth.
The discrepancy is documented rather than hidden.

One conflicting-source observation was encountered and resolved: a web-search
snippet rendered this image as "#39537 Snail shell on grape leaf", while the
directly fetched page says "Grape leaf". The fetched page was preferred and the
snippet was **not** used as evidence (`E03`). No image was classified DISPUTED as
a result, because the conflict was between an unreliable search snippet and a
directly fetched page, not between two reliable sources.

### 5.2 The 16 UNKNOWN images

Every one of the 16 UNKNOWN images returned **zero** provenance carriers: no
EXIF tags, no XMP packet, no IPTC/Photoshop IRB, no JPEG comment, no embedded
URL. This was verified independently in Step 5.3 and reproduces the Step 1
finding (`E01`, `E10`).

Their filenames are `images (1..13).jfif`, `images.jfif` and `img1.jfif` -
generic application-save names carrying no source, author, site or condition
information. There is no identifier to search against (`E05`).

For the single file with a descriptive name,
`green-leaf-of-grapes-in-the-garden-free-photo.jpg`, two web searches were run -
one on the descriptive slug, one on the exact filename in quotes. Neither
returned a match. The filename resembles a stock-site slug convention, but no
matching page was located (`E04`). Even had a page been found, it would have
supplied a caption, not a disease determination.

So for all 16:

> **No sufficiently reliable independent ground-truth source established.**

No label is assigned. They are excluded from every accuracy calculation in this
report.

---

## 6. External Prediction Results

The frozen production predictor (`Predictor.predict_pil`) was run on all 17
external images with the same checkpoint, preprocessing, class list, device and
inference procedure. **No accuracy is reported in this section** - accuracy
requires ground truth, which does not exist yet.

Feature vectors were obtained with a read-only forward hook on the frozen
`head[8]` bottleneck. The hook was proved non-perturbing: the maximum absolute
probability difference between the hooked forward pass and the official
production path was **exactly 0.0** across all 17 images. No model was modified
and no second model was created.

| filename | GT | GT status | predicted | top-1 | top-2 | top-2 p | margin | P(BS) | P(Healthy) | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `grape leaf.jpg` | Healthy | VERIFIED | **BacterialSpot** | 0.878758 | Healthy | 0.035046 | 0.843712 | 0.878758 | 0.035046 | 56.56 |
| `green-leaf-of-grapes-in-the-garden-free-photo.jpg` | - | UNKNOWN | **Healthy** | 0.767340 | Black_Rot | 0.084562 | 0.682778 | 0.040220 | 0.767340 | 56.96 |
| `images (1).jfif` | - | UNKNOWN | **Healthy** | 0.595612 | Irrelavant | 0.138600 | 0.457012 | 0.020901 | 0.595612 | 54.25 |
| `images (10).jfif` | - | UNKNOWN | **DownyMildew** | 0.376511 | Black_Rot | 0.228984 | 0.147527 | 0.010526 | 0.188464 | 53.75 |
| `images (11).jfif` | - | UNKNOWN | **Healthy** | 0.723564 | Irrelavant | 0.112115 | 0.611449 | 0.017908 | 0.723564 | 50.59 |
| `images (12).jfif` | - | UNKNOWN | **Black_Rot** | 0.734547 | Healthy | 0.082299 | 0.652248 | 0.013698 | 0.082299 | 51.77 |
| `images (13).jfif` | - | UNKNOWN | **PowderyMildew** | 0.525692 | DownyMildew | 0.258072 | 0.267620 | 0.022738 | 0.099109 | 56.46 |
| `images (2).jfif` | - | UNKNOWN | **Healthy** | 0.270880 | DownyMildew | 0.197395 | 0.073485 | 0.184808 | 0.270880 | 59.61 |
| `images (3).jfif` | - | UNKNOWN | **Black_Rot** | 0.712782 | Healthy | 0.122192 | 0.590590 | 0.039530 | 0.122192 | 53.10 |
| `images (4).jfif` | - | UNKNOWN | **Black_Rot** | 0.504996 | Healthy | 0.318915 | 0.186081 | 0.010810 | 0.318915 | 47.44 |
| `images (5).jfif` | - | UNKNOWN | **Healthy** | 0.842689 | DownyMildew | 0.047923 | 0.794766 | 0.020660 | 0.842689 | 56.90 |
| `images (6).jfif` | - | UNKNOWN | **Healthy** | 0.774728 | PowderyMildew | 0.127639 | 0.647090 | 0.036509 | 0.774728 | 51.55 |
| `images (7).jfif` | - | UNKNOWN | **Healthy** | 0.442244 | Esca | 0.186965 | 0.255279 | 0.056625 | 0.442244 | 48.27 |
| `images (8).jfif` | - | UNKNOWN | **Esca** | 0.392432 | Healthy | 0.368436 | 0.023995 | 0.049075 | 0.368436 | 61.11 |
| `images (9).jfif` | - | UNKNOWN | **Black_Rot** | 0.482182 | Irrelavant | 0.278417 | 0.203765 | 0.098616 | 0.032043 | 69.92 |
| `images.jfif` | - | UNKNOWN | **Healthy** | 0.615791 | Black_Rot | 0.207517 | 0.408275 | 0.063162 | 0.615791 | 74.02 |
| `img1.jfif` | - | UNKNOWN | **Healthy** | 0.677044 | Black_Rot | 0.089694 | 0.587350 | 0.082526 | 0.677044 | 74.40 |

Predicted-class distribution across all 17 images:

| predicted class | count |
|---|---|
| BacterialSpot | 1 |
| Black_Rot | 4 |
| DownyMildew | 1 |
| Esca | 1 |
| Healthy | 9 |
| PowderyMildew | 1 |
| **total** | **17** |

**This distribution is the model's output distribution, not an error
distribution.** With 16 of 17 ground truths UNKNOWN it cannot be read as accuracy,
as a failure rate, or as evidence that any particular image is wrong. Nine images
predicted Healthy does not mean nine images are healthy.

Figures: `visualizations/external_predictions.png`.

---

## 7. Verified External Performance

Computed on the VERIFIED subset only.

| metric | value | contributing classes |
|---|---|---|
| images | 1 | - |
| correct | 0 | - |
| accuracy | 0.0000 | - |
| balanced accuracy | 0.0000 | 1 of 1 |
| macro precision | undefined | 0 of 1 |
| macro recall | 0.0000 | 1 of 1 |
| macro F1 | undefined | 0 of 1 |

The VERIFIED confusion matrix in full:

| ground truth \ predicted | BacterialSpot | Black_Rot | DownyMildew | Esca | Healthy | Irrelavant | PowderyMildew |
|---|---|---|---|---|---|---|---|
| Healthy | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

**Reading this correctly requires care.**

- The VERIFIED subset is **one image**. `0/1` is a statement about one image, not
  a rate.
- Healthy recall is 0.0 because the single verified Healthy image was predicted
  BacterialSpot.
- Healthy **precision is undefined**, because the model never predicted Healthy on
  any verified image. There is no denominator. The same applies to macro F1.
  These are reported as undefined rather than as 0.0, because 0.0 would assert a
  measurement that does not exist.
- Balanced accuracy coincides with macro recall here because only one class has
  ground truth.
- **VERIFIED + SUPPORTED** is the same set, because SUPPORTED is empty. Accuracy
  0/1. No broader lower-certainty evaluation could be produced.

Nothing here supports a generalisation. With `n = 1` there is no confidence
interval worth computing and no claim about real-world external accuracy.

Figures: `visualizations/confusion_matrix.png`.

---

## 8. Healthy <-> BacterialSpot Analysis

| direction | labelled images in that direction | errors |
|---|---|---|
| Healthy -> BacterialSpot | 1 | 1 |
| BacterialSpot -> Healthy | 0 | 0 |

Detailed error records:

| filename | GT | prediction | top-1 | P(BS) | P(Healthy) | margin |
|---|---|---|---|---|---|---|
| `grape leaf.jpg` | Healthy | **BacterialSpot** | 0.878758 | 0.878758 | 0.035046 | 0.843712 |

### 8.1 Is `grape leaf.jpg` the only verified Healthy -> BacterialSpot error?

**It is the only *verified* one. Whether it is the only *actual* one cannot be
determined.**

The question requires at least two verified Healthy images or at least two
verified BacterialSpot images. There is 1 verified Healthy image and **zero**
verified BacterialSpot images. With one labelled image in one direction and none
in the other, neither "isolated" nor "recurring" is a supportable description.
Both would be over-reading a sample of size one.

The pre-declared decision rule recorded this explicitly as
`recurrence_assessable = False` before the outcome was inspected.

### 8.2 Where the answers would come from

This is the binding constraint on the entire investigation, not a technicality.
Resolving it requires one of:

1. an authoritative dataset with documented provenance that these 16 images come
   from, which would bring labels with it;
2. reverse image search performed by a human;
3. a qualified plant-pathology source identifying each image;
4. a documented expert examination.

None of the four is available to this analysis. The step's honest output is
therefore *incomplete ground truth*, not a performance number.

---

## 9. Other Class Errors

The investigation is not restricted to Healthy/BacterialSpot. The purpose is to
separate a *specific* Healthy/BacterialSpot failure from a *general* external
domain shift. All seven classes were checked.

| class | labelled external images | correct | incorrect | assessable? |
|---|---|---|---|---|
| Black_Rot | 0 | 0 | 0 | no |
| DownyMildew | 0 | 0 | 0 | no |
| Esca | 0 | 0 | 0 | no |
| Irrelavant | 0 | 0 | 0 | no |
| PowderyMildew | 0 | 0 | 0 | no |

**No external image with ground truth exists for any of the five other classes.**
Black_Rot, DownyMildew, Esca, Irrelavant and PowderyMildew all have zero labelled
external images, so none of them can be shown to be failing or not failing.

The model does predict five of the seven classes somewhere in the external set
(Black_Rot 4, DownyMildew 1, Esca 1, PowderyMildew 1, and never Irrelavant).
That is a statement about model behaviour on unlabelled inputs, not evidence of
external error on those classes. **Category D (broader domain shift) cannot be
excluded, and cannot be confirmed either** - and this step provides no basis for
preferring one over the other.

---

## 10. Confidence Analysis

| quantity | min | max | mean |
|---|---|---|---|
| top-1 confidence | 0.2709 | 0.8788 | 0.6069 |
| top-1/top-2 margin | 0.0240 | 0.8437 | 0.4372 |

| band | count |
|---|---|
| top-1 >= 0.70 | 7 |
| top-1 < 0.70 | 10 |
| margin < 0.25 (ambiguous) | 5 |

Lowest-confidence images:

| filename | GT status | predicted | top-1 | margin |
|---|---|---|---|---|
| `images (2).jfif` | UNKNOWN | Healthy | 0.2709 | 0.0735 |
| `images (10).jfif` | UNKNOWN | DownyMildew | 0.3765 | 0.1475 |
| `images (8).jfif` | UNKNOWN | Esca | 0.3924 | 0.0240 |
| `images (7).jfif` | UNKNOWN | Healthy | 0.4422 | 0.2553 |
| `images (9).jfif` | UNKNOWN | Black_Rot | 0.4822 | 0.2038 |

For the verified set:

| case | count | detail |
|---|---|---|
| high-confidence incorrect | 1 | `grape leaf.jpg` at top-1 0.878758, margin 0.843712 |
| low-confidence correct | 0 | - |

The single verified error is also the **highest-confidence prediction in the
entire external set** (0.878758 for `grape leaf.jpg`, against a maximum of 0.878758 across all 17). It is
not a borderline call that a slightly different threshold would fix; the model
assigns 0.878758 to BacterialSpot against 0.035046 to Healthy, a margin of 0.843712.

**This is softmax confidence, not a calibrated probability of correctness.** A
high softmax output says the model is internally committed to that class. It is
not evidence that the model is right, and it is not a measure of uncertainty in
any probabilistic sense. No calibration analysis was performed and none is implied.

Figures: `visualizations/verified_error_contact_sheet.png`.

---

## 11. Feature-Space Analysis

Diagnostic only. The model was not changed. The 64-D `head[8]` bottleneck is the
same representation used in Steps 5.1 and 5.2, with centroids taken from the
frozen Step 5.1 cache so the values are bit-identical to those steps
(Healthy n=831, BacterialSpot n=700, inter-centroid cosine 0.356527).

Margin is `cos(BacterialSpot centroid) - cos(Healthy centroid)`; positive means
toward BacterialSpot.

Training-set margin ranges:

| population | min | max | mean |
|---|---|---|---|
| Healthy training images (n=831) | -0.7299 | 0.3441 | -0.5991 |
| BacterialSpot training images (n=700) | -0.0851 | 0.7708 | 0.6107 |

External images, ordered by margin:

| filename | GT status | predicted | cos Healthy | cos BacterialSpot | margin | beyond Healthy train max? |
|---|---|---|---|---|---|---|
| `grape leaf.jpg` | VERIFIED | BacterialSpot | 0.4441 | 0.9364 | +0.4923 | **YES** |
| `images (9).jfif` | UNKNOWN | Black_Rot | 0.4577 | 0.5865 | +0.1288 | no |
| `images (2).jfif` | UNKNOWN | Healthy | 0.6902 | 0.5881 | -0.1021 | no |
| `images (3).jfif` | UNKNOWN | Black_Rot | 0.6084 | 0.4257 | -0.1828 | no |
| `images (13).jfif` | UNKNOWN | PowderyMildew | 0.5563 | 0.3079 | -0.2483 | no |
| `images (12).jfif` | UNKNOWN | Black_Rot | 0.5349 | 0.2764 | -0.2584 | no |
| `images (7).jfif` | UNKNOWN | Healthy | 0.7203 | 0.4412 | -0.2791 | no |
| `img1.jfif` | UNKNOWN | Healthy | 0.7915 | 0.5063 | -0.2852 | no |
| `images (8).jfif` | UNKNOWN | Esca | 0.6775 | 0.3723 | -0.3052 | no |
| `images.jfif` | UNKNOWN | Healthy | 0.8648 | 0.5033 | -0.3615 | no |
| `images (10).jfif` | UNKNOWN | DownyMildew | 0.5225 | 0.1376 | -0.3849 | no |
| `green-leaf-of-grapes-in-the-garden-free-photo.jpg` | UNKNOWN | Healthy | 0.9082 | 0.4262 | -0.4820 | no |
| `images (6).jfif` | UNKNOWN | Healthy | 0.8703 | 0.3616 | -0.5087 | no |
| `images (4).jfif` | UNKNOWN | Black_Rot | 0.7474 | 0.2259 | -0.5215 | no |
| `images (1).jfif` | UNKNOWN | Healthy | 0.8425 | 0.3189 | -0.5236 | no |
| `images (11).jfif` | UNKNOWN | Healthy | 0.8766 | 0.3094 | -0.5672 | no |
| `images (5).jfif` | UNKNOWN | Healthy | 0.9081 | 0.3388 | -0.5692 | no |

**Observations:**

- `grape leaf.jpg` is the only image with a positive margin, and the only image that lies
  beyond the maximum margin of **all 831** Healthy training images (0.4923 vs Healthy
  training max 0.3441). This reproduces Steps 5.1 and 5.2 exactly and extends the
  comparison to the rest of the external set.
- Exactly **one** other image, `images (9).jfif`, has a positive margin (0.1288) - but that value
  sits comfortably **inside** the Healthy training range, so it does not
  reproduce the verified query's position.
- The remaining 15 images have negative margins, i.e. they sit on the Healthy
  side of the centroid split.

**Limits that must be stated.** Feature-space position is **not ground truth** and
is never used as a label. It cannot tell us whether any of the 16 UNKNOWN images
is correct or incorrect, and a negative margin does not mean an image is healthy.
Similarity-to-training-data was deliberately excluded as labelling evidence
(`E07`); this section uses it only as a diagnostic comparison, which is the
allowed use. No clustering or dimensionality reduction is offered as evidence of
class correctness.

---

## 12. Sample-Size Limitations

This is the most important section of the report.

| limitation | consequence |
|---|---|
| VERIFIED subset is 1 image | `0/1` is an observation about one file, not a rate. No external accuracy can be inferred from it. |
| 16 of 17 images UNKNOWN | 94% of the external set cannot be scored at all. |
| 0 verified BacterialSpot images | the BacterialSpot -> Healthy direction is completely unmeasured. |
| 0 labelled images for the 5 other classes | no external error rate can be computed for any other class. |
| no reverse image search | the most likely route to provenance for the 16 UNKNOWN images was unavailable. |
| no analyst visual inspection | no independent visual corroboration was possible. |

For scale, the held-out test set contains **1,189** images in total, of which 140 are
BacterialSpot. The external set has **17** images, of which **1** can be scored.
These are not statistically comparable quantities and are not compared anywhere in
this report. The 1,189-image test set is drawn from the same curated source
distribution the model was trained on, whereas these 17 images are real-world
downloads; even with perfect labels, 1 scored image could not stand in for it.

The specific prohibited inference, stated so it is unmistakable: **this report
does not say, and must not be read as saying, anything like "the model has a 100%
error rate on real-world healthy leaves".** One verified image was misclassified.
That is the entire external performance measurement available.

A binomial 95% interval on 0/1 spans effectively the whole range, so even the
appearance of a precise figure would be misleading. None is given.

---

## 13. Findings

1. **Independent ground truth is almost entirely unavailable.** 16 of 17 external
   images are UNKNOWN. This is not a search that gave up early: 10 provenance
   attempts were logged, including failures, and the exhaustive metadata scan
   found zero provenance carriers in all 16 UNKNOWN files.
2. **Only one external file carries any provenance at all** (`grape leaf.jpg`), and its
   source page establishes image identity but makes **no** statement about
   health or disease.
3. **The VERIFIED external performance is 0/1** - the one verified image is
   misclassified. Precision and macro F1 are undefined because the model never
   predicted Healthy on a verified image.
4. **The verified error is the model's most confident external prediction
   (0.8788), not a borderline call.**
5. **Healthy -> BacterialSpot vs BacterialSpot -> Healthy cannot be separated.**
   There is 1 verified Healthy image and 0 verified BacterialSpot images, so
   neither "isolated" nor "recurring" is supportable.
6. **In the 64-D diagnostic, no second image reproduces the verified query's
   position.** `grape leaf.jpg` is the only external image beyond the maximum margin of all
   831 Healthy training images; one other image has a positive margin but sits
   inside that range. This is diagnostic, not ground truth.
7. **No class-level external conclusion is possible for the other five classes** -
   none has a single labelled external image.
8. **Step 5.2's conclusion is neither confirmed nor overturned.** Step 5.2 found
   illumination sensitivity insufficient to explain the failure (43 variants, no
   flip, worst margin 0.310429). Step 5.3 did not assume illumination was the cause,
   and produced no evidence either way.
9. **Step 5.1's conclusion is reproduced, not overturned.** `grape leaf.jpg` remains beyond
   the entire Healthy training range (0.4923 vs max 0.3441 over 831 training images).
10. **The binding constraint on this investigation is human, not computational.**
    The models are reproducible and the integrity checks pass. What is missing is
    provenance and expert labelling for 16 files - work that cannot be automated
    into existence.

---

## 14. Decision Category

**CATEGORY_A - insufficient ground truth.**

| | |
|---|---|
| Category | **A** |
| Meaning | Too many images remain UNKNOWN/DISPUTED for a reliable external conclusion |
| Action | **Do NOT retrain** |

### 14.1 The rule was declared before the outcome was seen

The rule was encoded in the analysis script before the results were inspected,
so the category could not be chosen post hoc.

| criterion | value | fired? |
|---|---|---|
| A: fewer than 5 labelled images | 1 labelled | **yes** |
| A: fewer than 2 labelled images in either direction | 1 Healthy, 0 BacterialSpot | **yes** |
| C: >= 2 labelled errors in one direction | 1 Healthy->BS, 0 BS->H | no |
| D: >= 3 classes systematically failing | 0 | no |

`A takes precedence when the labelled set is too small to separate B/C/D.`

**A** is not a failure of the analysis; it is the correct answer to the question
asked. The question was "isolated, recurring, broader, or undecidable?", and the
evidence supports the fourth.

### 14.2 Why not B, C or D

- **Not B (isolated).** B requires at least two reliably labelled external
  Healthy images, so that "the others do not show the same pattern" can be
  tested. There is one.
- **Not C (recurring).** C requires recurrence across multiple independently
  verified images. One image cannot recur.
- **Not D (broader domain shift).** D requires at least three ground-truth
  classes showing systematic external error. No other class has a single
  labelled image.

Choosing B, C or D here would mean asserting an external failure pattern from a
single labelled example. That is precisely the over-reading this step exists to
prevent.

---

## 15. Whether Training Is Justified

**Training intervention: NOT JUSTIFIED.**

| question | answer |
|---|---|
| Is there evidence of a recurring failure? | No - recurrence is unassessable at n=1 |
| Is there evidence of a broader domain shift? | No - unassessable, 0 labelled images for other classes |
| Is there a specific, evidence-supported problem to target? | No |
| Was illumination identified as the cause? | No - Step 5.2 found it insufficient |
| Was a training-distribution gap identified? | Partially, on 1 image - Step 5.1/5.2, unchanged by Step 5.3 |

### 15.1 Why not justified

The evidence that would justify a targeted training intervention is a
**repeated, independently verified** failure pattern. This step produced exactly
one verified observation. Targeting training at a pattern seen once is not
evidence-driven; it is guessing with a compute budget.

Three specific traps are avoided:

1. **Retraining to fix one image.** Adding or upweighting `grape leaf.jpg` would
   make this step's own verified error disappear while leaving every real-world
   failure unmeasured. That is the definition of fitting the test set.
2. **Using the 16 UNKNOWN images as labels.** Labelling them from model output
   and then training on them is circular: it would train the model to agree with
   itself and would manufacture exactly the ground truth this step was built to
   obtain independently.
3. **Treating the 64-D outlier as a target.** `grape leaf.jpg` is beyond the Healthy training
   range, but a single point beyond a range is evidence of a coverage gap in
   *one* direction on *one* image. It is not a measured training deficit.

### 15.2 What would justify a future training investigation

- **First, close the ground-truth gap.** Provenance or expert labelling for the
  16 UNKNOWN images. Until at least a handful of verified external images per
  direction exist, no training question can be formulated.
- **Then, only if verified external images reproduce the pattern.** If several
  verified Healthy leaves are predicted BacterialSpot, that supports Category C
  and a targeted investigation of the Healthy training distribution's coverage.
- **Only then** consider data acquisition or sampling changes, with the external
  set held out and never trained on.

### 15.3 Recommendation

The highest-value next action is **not** a training run. It is to obtain
provenance or expert labels for the 16 UNKNOWN images - by reverse image search,
by identifying their source collection, or by expert examination. That single
step would convert this from undecidable to decidable. Step 5.2 reached the same
conclusion from the mechanism side: verify labels first, then re-measure.

---

## 16. What Should NOT Change

Nothing in the model, data or pipeline was modified, and this step recommends
none of the following:

| item | state | changed? |
|---|---|---|
| `best_iwnet.pth` | frozen checkpoint, SHA-256 `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` | **no** |
| model architecture / parameters | 10,161,307 parameters | **no** |
| class list | 7 classes | **no** |
| production preprocessing | Resize -> CenterCrop -> ToTensor -> Normalize | **no** |
| classification threshold | none invented or changed | **no** |
| training data / labels | untouched | **no** |
| `split_manifest.csv` | `d25542dab5673810` | **no** |
| baseline commit `e101607` | not amended, reset or squashed | **no** |
| the 17 external images | byte-identical | **no** |
| Step 1-5.2 artefacts | 72 files unchanged | **no** |
| labels in any dataset | unchanged; no automatic relabelling | **no** |
| Git history | no commit created | **no** |

**No automatic relabelling occurred.** Step 1's existing `manifest.csv` already
marked these images `ground_truth_status = UNKNOWN` and that is exactly where
they remain. Where a potential label issue might exist it is recorded only as
`CANDIDATE FOR MANUAL REVIEW` and nothing more - and in this step no such
candidate was raised, because there was no evidence to raise one.

No illumination-correction stage was added to production. No augmentation,
loss, sampling or architecture change was made or proposed.

---

## 17. Integrity Verification

Full log with every individual check: `verification_log.txt`.

| check | result |
|---|---|
| checkpoint SHA-256 unchanged | **PASS** |
| checkpoint size unchanged (41,213,897 bytes) | **PASS** |
| `Balanced_From_Sources` tree fingerprint unchanged | **PASS** |
| `Balanced_Final_Split` tree fingerprint unchanged | **PASS** |
| dataset class counts unchanged | **PASS** |
| `split_manifest.csv` unchanged | **PASS** |
| class list unchanged | **PASS** |
| parameter count unchanged | **PASS** |
| production preprocessing unchanged | **PASS** |
| baseline prediction reproduced | **PASS** |
| all 17 external images byte-identical | **PASS** |
| 72 Step 1-5.2 artefacts unchanged | **PASS** |
| hooked forward vs production path, max prob delta | 0.0 (read-only proof) |
| no file written outside the Step 5.3 directory | **PASS** |
| accuracy computed over UNKNOWN images | **none - deliberately absent** |
| UNKNOWN images carrying a proposed label | **none** |
| Git HEAD unchanged, nothing committed | **PASS** |

Dataset fingerprints:

| dataset | before | after |
|---|---|---|
| `Balanced_From_Sources` | `35c72a0ae283f7e8` | `35c72a0ae283f7e8` |
| `Balanced_Final_Split` | `389b24a8beac8ad5` | `389b24a8beac8ad5` |

---

## 18. Test Results

The existing test suite was run after all Step 5.3 work.

| | expected baseline | observed |
|---|---|---|
| passed | 161 | **161** |
| failed | 0 | **0** |
| errors | 0 | **0** |
| skipped | 0 | **0** |

No test was deleted, weakened, skipped or rewritten. No production code was
modified to make a test pass. This step created no new tests, because it
introduced no new production behaviour to test - the artefacts it produced are
validation records, and their internal consistency is checked in
`verification_log.txt` instead.

Runtime: **73.185 s**.

---

## 19. Reproducibility

| item | value |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| timm | 1.0.30 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| OS | Windows-11-10.0.26200-SP0 |
| Device | CPU (`cuda_available = False`) |
| Checkpoint SHA-256 | `c1fae27c83d68b613034ca089a27e8fa629f9478bec307ae5e0ff74e75a27c0b` |
| Checkpoint size | 41,213,897 bytes |
| Architecture | IWNET / EfficientNet-B3, 10,161,307 parameters |
| Preprocessing | Resize(255, BILINEAR) -> CenterCrop(224,224) -> ToTensor -> Normalize |
| Representation | 64-D `head[8]` bottleneck (read-only forward hook) |
| Centroids | frozen Step 5.1 cache, Healthy n=831, BacterialSpot n=700 |
| External images | 17 |
| VERIFIED / SUPPORTED / DISPUTED / UNKNOWN | 1 / 0 / 0 / 16 |
| Evidence attempts logged | 10 |
| Inference time per image | 47.44 - 74.40 ms (mean 57.45 ms) |
| Integrity capture time | 23.13 s each |
| Random seed | none - no stochastic step exists in this pipeline |
| Analysis date | 2026-09-30 |

**On the seed:** no randomness is used anywhere in this step - no augmentation, no
sampling, no shuffling, no dropout at inference. There is therefore nothing for a
seed to control, and recording an invented one would be misleading. The
reproducibility guarantee rests on the frozen checkpoint digest, the fixed
preprocessing, and the deterministic CPU inference path, all of which are
recorded above and re-verified in `integrity_after.json`.

### Artefacts

| file | contents |
|---|---|
| `STEP5_3_REPORT.md` | this report |
| `external_ground_truth.csv` | per-image adjudication: status, evidence level, source, dates, verifier, notes |
| `external_predictions.csv` | all 17 images: prediction, top-1/top-2, margin, all class probabilities, 64-D position |
| `external_validation_summary.json` | metrics, decision rule and its evaluation, feature-space and confidence analyses |
| `external_metadata.json` | full read-only metadata extraction for all 17 files |
| `external_predictions.json` | raw prediction records incl. all seven probabilities and vector norms |
| `confusion_matrix.csv` | confusion matrices and metrics for VERIFIED and VERIFIED+SUPPORTED |
| `verified_errors.csv` | the one verified error, fully detailed |
| `evidence_log.csv` | all 10 provenance attempts including failures and deliberate exclusions |
| `integrity_before.json` / `integrity_after.json` | frozen-baseline capture and re-capture |
| `verification_log.txt` | full pass/fail integrity and consistency log |
| `reference_scales.json` | frozen centroid and training-margin reference ranges |
| `visualizations/` | 3 figures + README |

### Reproduction order

1. `step53_integrity.py` with `S53_PHASE=before`
2. `step53_probe.py` - metadata extraction + production inference
3. `step53_groundtruth.py` - adjudication and metrics
4. `step53_viz.py` - figures
5. `step53_integrity.py` with `S53_PHASE=after`
6. `pytest`, then `step53_verify.py`

Scripts are held outside the repository in `%TEMP%` so no temporary analysis
file is scattered through the repo.


---

## 20. Required Final Questions

**A. How many of the 17 external images now have reliable ground truth?**
1 (VERIFIED) + 0 (SUPPORTED) = **1 of 17**.

**B. VERIFIED?** **1** - `grape leaf.jpg`.

**C. SUPPORTED?** **0**.

**D. UNKNOWN?** **16**.

**E. DISPUTED?** **0**.

**F. External performance on VERIFIED images?**
**0/1 correct (accuracy 0.0000)**, on a single image. Healthy precision and
macro F1 are undefined because the model never predicted Healthy on a verified
image. Not a rate.

**G. Performance on VERIFIED + SUPPORTED?**
**0/1** - the same single image, because SUPPORTED is empty. No broader
lower-certainty evaluation could be produced.

**H. Is Healthy -> BacterialSpot a recurring external failure?**
**Cannot be determined.** 1 verified Healthy image misclassified as
BacterialSpot; 0 verified BacterialSpot images. Recurrence requires at least two
labelled images in a direction. One observation cannot recur.

**I. Is BacterialSpot -> Healthy a recurring external failure?**
**Cannot be determined.** There are **no** verified BacterialSpot images, so this
direction is entirely unmeasured.

**J. Are other disease classes also failing?**
**Cannot be determined.** Black_Rot, DownyMildew, Esca, Irrelavant and
PowderyMildew have **zero** labelled external images each. Broader domain shift is
neither confirmed nor excluded.

**K. Is `grape leaf.jpg` isolated or part of a broader pattern?**
**Undecidable on the available evidence.** In the 64-D diagnostic it is the only
external image beyond the maximum margin of all 831 Healthy training images
(0.4923 vs 0.3441), and no second image reproduces that position - but feature position
is not ground truth, and the ground truth for the other 16 images does not exist.

**L. Does the evidence justify a training intervention?**
**No.**

**M. Why not?**
The evidence that would justify it - repeated, independently verified failure -
was not found, because it could not be looked for: 16 of 17 images have no
independent ground truth. Retraining now would target a pattern observed once,
would risk fitting the very image being validated, and would leave every real
failure still unmeasured.

**N. If training were justified, what would be targeted?**
Not applicable under the current evidence. Should verified external images later
reproduce the pattern, the evidence-supported target would be the **coverage gap
in the Healthy training distribution's feature range** - grape leaf.jpg lies beyond the
maximum margin of all 831 Healthy training images - and the investigation would
start with data coverage, not with illumination augmentation, which Step 5.2
found insufficient.


---

## 21. Terminal Summary

```
Step 5.3 complete.

External images:          17
VERIFIED:                 1
SUPPORTED:                0
DISPUTED:                 0
UNKNOWN:                  16

Verified external accuracy:        0/1
Verified + Supported accuracy:     0/1 or NOT COMPUTABLE

Healthy -> BacterialSpot verified errors:      1
BacterialSpot -> Healthy verified errors:      0
Other verified external errors:                0

Main finding:
  16 of 17 external images have no independently verifiable ground truth;
  only 1 image is VERIFIED, and it is misclassified, so neither an isolated
  nor a recurring external failure can be distinguished from the evidence.

Decision category:      A
Training justified:     NO
Checkpoint:             UNCHANGED
Dataset:                UNCHANGED
External images:        UNCHANGED
Tests:                  161 passed / 0 failed / 0 errors / 0 skipped
Git:                    DO NOT COMMIT
Output:                 RealWorldValidation\step5_3_external_validation\
```

---

## 22. Explicit Non-Claims

To be unambiguous about what this report does **not** say:

- It does **not** prove lighting caused the failure. Step 5.2 found illumination
  sensitivity insufficient to explain it.
- It does **not** show that illumination is irrelevant.
- It does **not** establish external accuracy. `0/1` on one image is not a rate.
- It does **not** show the 16 UNKNOWN images are misclassified or correct.
- It does **not** identify the true cause of the `grape leaf.jpg` failure.
- It does **not** generalise beyond this image, this model and this checkpoint.
- It does **not** treat the 64-D position as evidence of class correctness.
- It does **not** treat model agreement as verification.
- It does **not** convert visual impression into a label.
- It does **not** prove the failure is isolated, and does **not** prove it recurs.
- It does **not** justify training, and no threshold, architecture or dataset
  change is recommended.
- No evidence level in this report is described as proven.