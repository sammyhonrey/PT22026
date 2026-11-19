# Grad-CAM notes (E3, six real photos)

Red and yellow mean the model paid more attention to that part. Blue and purple mean less.
The map is made from a 14 x 14 grid of patches, so it is rough and blocky.

## Correct predictions
- cardboard_001: The heat is in the middle of the open box. This looks right.
- clothes_007: The heat is on the shirt, strongest near the neck and the top edge.
- plastic_016: The answer is right, but a lot of heat is on the floor around the bottle and not on the bottle.

## Wrong predictions
- paper_004 (said clothes): The heat is on the crumpled edges and folds, not on the KFC logo.
- cardboard_013 (said paper): The heat is on the edges of the box and small spots, not on the picture in the middle.
- plastic_003 (said metal): The heat is around the shiny rim of the object and on the carpet.

## What I think
In these six photos the heat was not always on the object. Sometimes it was on edges, folds or the background.
Six photos is too few to be sure. Grad-CAM shows where the model reacted, but it does not prove why.