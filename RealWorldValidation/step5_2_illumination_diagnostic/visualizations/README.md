# Step 5.2 visualisations

Generated from the same deterministic transformation functions used by the
experiment, with every caption value read from `prediction_details.json` by
`variant_id`. Images and numbers therefore cannot be mismatched.

- `original.png` - the protected original, unmodified
- `brightness_grid.png` / `contrast_grid.png` / `gamma_grid.png`
- `directional_grid.png` - 4 light directions x 3 strengths
- `shadow_grid.png` - 2 shadow centres x 3 strengths
- `correction_grid.png` - gray-world, flat-field and retinex-style corrections
- `illumination_grid.png` - all 43 variants in one overview
- `response_curves.png` - P(BacterialSpot), P(Healthy) and centroid margin vs factor
- `family_movement.png` - per-variant movement grouped by family

## LIMITATION

**These images have not been visually inspected.** The environment used to
produce them cannot display images to the analyst. They are structurally
validated (valid PNG, decodable, dimensions recorded in the report) only.
Whether the transformations look like plausible real-world lighting is a
question for a human, and no visual claim is made anywhere in Step 5.2.
