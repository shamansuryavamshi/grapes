# Step 5.3 visualisations

| file | what it shows |
|---|---|
| `external_predictions.png` | all 17 external images, frozen-model prediction, top-2, margin, P(BacterialSpot)/P(Healthy) and 64-D centroid margin. Border colour encodes ground-truth status: green = VERIFIED, grey = UNKNOWN. |
| `confusion_matrix.png` | the VERIFIED confusion matrix (a single cell) beside the status breakdown of all 17 images. |
| `verified_error_contact_sheet.png` | the one verified error at full size with all seven softmax probabilities. |

## Captions and scope

Every caption is read from the recorded artefacts by filename, so a caption cannot disagree with the data.

**No visual, appearance or disease judgement is made in any figure.** The environment used for Step 5.3 cannot display images to the analyst, so nothing was visually inspected and no such claim is recorded.

Colours encode only values present in the artefacts: ground-truth status and predicted class. Nothing is encoded from visual appearance.

Structural validation (valid PNG, decodable, dimensions recorded) is in `../verification_log.txt`.
