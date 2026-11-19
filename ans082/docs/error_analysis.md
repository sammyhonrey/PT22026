# Error analysis (E3, real phone photos)

The 12 most confident wrong predictions (confidence 0.95 to 0.99) were inspected by eye.

- **Label problems (3 photos):** cardboard_005 is a clear plastic egg tray, biological_004 is a drug container
  (outside the food-waste class), trash_023 is a toothpaste tube (not a deliberate "trash" choice).
  Labels were not changed; see real_photo_audit.md.
- **Glossy or transparent plastic (4 photos):** plastic_003 (foil rim), plastic_009 (black glossy bowl) and
  plastic_023 were read as metal, plastic_002 (transparent mug) as glass.
- **Crumpled paper (2 photos):** paper_004 (KFC bag) and paper_007 (handwritten sheet) were read as clothes.
- **Paper vs cardboard (3 photos):** glossy printed boxes (cardboard_013, cardboard_014) were read as paper,
  a stiff paper bag (paper_017) as cardboard.
- **Possible factor, not yet tested:** most real photos have a red carpet or pavement background, unlike the
  plain backgrounds of most training images. Grad-CAM (Phase 8) will show whether the model looks at the background.